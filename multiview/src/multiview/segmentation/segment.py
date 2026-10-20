#!/usr/bin/env python
"""SAM 3.1 Object Multiplex box-prompted video segmentation (run with ~/sam3env/bin/python).

Step 1 (frames):   segment.py extract --video V.mp4 --start S --end E [--fps 30] --out-dir D
                   -> D/frames/00000.jpg ... (source resolution) + D/frames.json
                   (--fps 0 without --start/--end: every decoded frame + D/frames_pts.json from ingest.py's probe)
Step 2 (segment):  segment.py track --out-dir D --init-dets dets.json [--max-objects 24] [--min-score 0.4]
                   [--dtype bf16|fp16|fp32] [--preview]
  dets.json is detect.py output; the detections of its first frame (which must be D/frames/00000.jpg,
  or pass --init-boxes boxes.json = [[x1,y1,x2,y2], ...] in source pixels) seed one object each.

Outputs in D:
  masks.json   {"video", "width", "height", "fps", "frames": [{"frame_idx", "objects": [{"id", "score",
               "box_xyxy", "rle": {"size": [h, w], "counts": str}}]}]}  (COCO RLE, source pixels;
               objects SAM considers absent on a frame are omitted)
  tracks.npz   ids (N,), boxes (T,N,4) xyxy NaN when absent, present (T,N) bool, scores (T,N) object logits
  stats.json   timings, peak GPU memory, objects alive per frame, multiplex bucket layout
  preview.mp4  (with --preview) masks + ids drawn over the frames

Prompting: each box is turned into a mask on frame 0 with the tracker's interactive SAM head
(box = corner points with labels 2/3). Small players often come back empty at the full-frame 1008 px input, so a
mask under --min-mask-frac of the box area is re-prompted on a ~3x crop around the box resized to the model input
(--crop-prompt), and if that is still too small it is replaced by an ellipse inscribed in the box; then all masks are added together with add_new_masks so the
objects share multiplex buckets (16 slots each) in a single inference state, and propagated.
"""
import argparse, json, os, subprocess, sys, time
from contextlib import nullcontext

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from pycocotools import mask as mask_utils
from safetensors.torch import load_file

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
DEFAULT_CKPT = os.path.expanduser("~/sam3_work/checkpoints/sam3.1/sam3.1_multiplex.safetensors")
IMAGE_SIZE = 1008
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


def cmd_extract(args):
    fdir = os.path.join(args.out_dir, "frames")
    os.makedirs(fdir, exist_ok=True)
    filters = ([f"fps={args.fps}"] if args.fps else []) + ([f"scale={args.width}:-2"] if args.width else [])
    cmd = [FFMPEG, "-v", "error", "-y"]
    if args.start > 0:
        cmd += ["-ss", f"{args.start:.3f}"]
    cmd += ["-i", args.video]
    if args.end > 0:
        cmd += ["-t", f"{args.end - args.start:.3f}"]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    else:
        cmd += ["-fps_mode", "passthrough"]  # one jpg per decoded frame, no duplication / dropping
    subprocess.run(cmd + ["-q:v", "2", "-start_number", "0", os.path.join(fdir, "%05d.jpg")], check=True)
    frames = sorted(f for f in os.listdir(fdir) if f.endswith(".jpg"))
    h, w = cv2.imread(os.path.join(fdir, frames[0])).shape[:2]
    meta_fps = args.fps
    if not args.fps:  # native timing: attach per-frame pts from ingest.py's probe if present
        meta_fps = 30.0
        probe = args.video + ".probe.json"
        if os.path.exists(probe):
            pr = json.load(open(probe))
            meta_fps = round(1.0 / pr["frame_dt"]["median"], 6)
            if args.start <= 0 and args.end <= 0:
                if pr["decoded_frames"] != len(frames):
                    raise RuntimeError(f"extracted {len(frames)} frames but probe decoded {pr['decoded_frames']}")
                json.dump({"pts_s": pr["pts_s"]}, open(os.path.join(args.out_dir, "frames_pts.json"), "w"))
    json.dump({"video": os.path.abspath(args.video), "start": args.start, "end": args.end, "fps": meta_fps,
               "width": w, "height": h, "num_frames": len(frames)},
              open(os.path.join(args.out_dir, "frames.json"), "w"), indent=1)
    print(json.dumps({"frames": len(frames), "width": w, "height": h, "dir": fdir}))


def patch_sdpa_for_turing():
    """sam3/model/decoder.py pins SDPBackend.FLASH_ATTENTION, which PyTorch only ships for sm_80+.
    On Turing (sm_75) that raises "No available kernel"; allow mem-efficient / math fallbacks instead."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import sam3.model.decoder as dec
    if torch.cuda.get_device_capability() >= (8, 0):
        return False
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    dec.sdpa_kernel = lambda *_a, **_k: sdpa_kernel(backends)
    return True


def build_tracker(ckpt_path):
    from sam3.model_builder import build_sam3_multiplex_video_model
    model = build_sam3_multiplex_video_model(checkpoint_path=None, load_from_HF=False, use_fa3=False,
                                             use_rope_real=False, device="cpu")
    if ckpt_path.endswith(".safetensors"):
        st = load_file(ckpt_path)
    else:
        st = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        st = st.get("model", st)
    # merged SAM 3.1 checkpoint: tracker.model.* -> tracker; detector.backbone.vision_backbone.* -> shared ViT
    sd = {}
    for k, v in st.items():
        if k.startswith("tracker.model."):
            sd[k[len("tracker.model."):]] = v
        elif k.startswith("detector.backbone.vision_backbone."):
            sd["backbone." + k[len("detector.backbone."):]] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    # Tracker-only use: _get_image_feature asks for the detector ("sam3") head too, whose flat keys break the
    # clone loop in VideoTrackingDynamicMultiplex.forward_image (upstream only runs this path via the detector).
    orig_forward_image = model.forward_image

    def forward_image(img, need_sam3_out=False, **kw):
        return orig_forward_image(img, need_sam3_out=False, **kw)

    model.forward_image = forward_image
    return model.cuda().eval()


def load_images(fdir, frames):
    """Frames -> (T,3,1008,1008) fp16 on GPU, normalised like sam3.model.io_utils (mean=std=0.5)."""
    out = torch.empty(len(frames), 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float16, device="cuda")
    for i, f in enumerate(frames):
        rgb = cv2.cvtColor(cv2.imread(os.path.join(fdir, f)), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        out[i] = (torch.from_numpy(rgb).cuda().permute(2, 0, 1).half() / 255.0 - 0.5) / 0.5
    return out


def crop_prompt_mask(model, frame_bgr, box, scale=3.0, min_side=96):
    """Box prompt on a square crop around the box (about `scale` x the box), resized to the model input, so a small
    player covers many more model pixels. Returns a full-frame bool mask (numpy)."""
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    side = max(min_side, scale * max(x2 - x1, y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a0, b0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    a1, b1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    crop = frame_bgr[b0:b1, a0:a1]
    ch, cw = crop.shape[:2]
    rgb = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    img = ((torch.from_numpy(rgb).cuda().permute(2, 0, 1).half() / 255.0 - 0.5) / 0.5)[None]
    st = model.init_state(video_height=ch, video_width=cw, num_frames=1)
    st["images"] = img
    pts = torch.tensor([[(x1 - a0) / cw, (y1 - b0) / ch], [(x2 - a0) / cw, (y2 - b0) / ch]], dtype=torch.float32)
    _, _, _, vm = model.add_new_points(st, 0, 1, pts, torch.tensor([2, 3], dtype=torch.int32), clear_old_points=True,
                                       rel_coordinates=True)
    full = np.zeros((H, W), bool)
    full[b0:b1, a0:a1] = (vm[0, 0] > 0).cpu().numpy()
    # keep only the part inside (a slightly padded) box: crops can include neighbours
    keep = np.zeros_like(full)
    keep[int(max(0, y1 - 4)):int(min(H, y2 + 4)), int(max(0, x1 - 4)):int(min(W, x2 + 4))] = True
    return full & keep


def color(i):
    rng = np.random.default_rng(int(i) * 7919 + 1)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def mask_box(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def pick_boxes(args, meta):
    if args.init_boxes:
        boxes = [list(map(float, b)) for b in json.load(open(args.init_boxes))]
        return boxes, [1.0] * len(boxes)
    dets = json.load(open(args.init_dets))
    f0 = dets["frames"][0]
    if (f0["width"], f0["height"]) != (meta["width"], meta["height"]):
        raise ValueError(f"init dets are {f0['width']}x{f0['height']}, frames are {meta['width']}x{meta['height']}")
    d = [x for x in f0["detections"] if x["class_name"] == "person" and x["score"] >= args.min_score]
    d = sorted(d, key=lambda x: -x["score"])[: args.max_objects]
    return [x["box_xyxy"] for x in d], [x["score"] for x in d]


def write_preview(path, fdir, frames, per_frame, fps, width=1280):
    h0, w0 = cv2.imread(os.path.join(fdir, frames[0])).shape[:2]
    w, h = width, int(round(h0 * width / w0 / 2) * 2)
    proc = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", path],
                            stdin=subprocess.PIPE)
    for fi, f in enumerate(frames):
        img = cv2.imread(os.path.join(fdir, f))
        over = img.copy()
        for obj in per_frame[fi]:
            m = mask_utils.decode(obj["rle"]).astype(bool)
            over[m] = color(obj["id"])
        img = cv2.addWeighted(over, 0.45, img, 0.55, 0)
        for obj in per_frame[fi]:
            x1, y1, x2, y2 = obj["box_xyxy"]
            cv2.rectangle(img, (x1, y1), (x2, y2), color(obj["id"]), 2)
            cv2.putText(img, str(obj["id"]), (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"frame {fi}  objects {len(per_frame[fi])}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        proc.stdin.write(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).tobytes())
    proc.stdin.close()
    proc.wait()


def cmd_track(args):
    meta = json.load(open(os.path.join(args.out_dir, "frames.json")))
    fdir = os.path.join(args.out_dir, "frames")
    frames = sorted(f for f in os.listdir(fdir) if f.endswith(".jpg"))[: args.max_frames or None]
    H, W, T = meta["height"], meta["width"], len(frames)
    boxes, det_scores = pick_boxes(args, meta)
    obj_ids = list(range(1, len(boxes) + 1))
    stats = {"dtype": args.dtype, "num_frames": T, "width": W, "height": H, "num_init_objects": len(boxes),
             "init_det_scores": det_scores, "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}

    stats["sdpa_patched_for_pre_ampere"] = patch_sdpa_for_turing()
    t0 = time.perf_counter()
    model = build_tracker(args.ckpt)
    stats["model_load_s"] = round(time.perf_counter() - t0, 2)
    stats["multiplex_count"] = model.multiplex_controller.multiplex_count

    ac = DTYPES[args.dtype]
    ctx = torch.autocast(device_type="cuda", dtype=ac) if ac is not None else nullcontext()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), ctx:
        t0 = time.perf_counter()
        images = load_images(fdir, frames)
        torch.cuda.synchronize(); stats["frame_load_s"] = round(time.perf_counter() - t0, 2)

        def new_state():
            st = model.init_state(video_height=H, video_width=W, num_frames=T)
            st["images"] = images
            return st

        # 1) box -> mask on frame 0 via the interactive head (scratch state, one object at a time)
        t0 = time.perf_counter()
        scratch = new_state()
        init_masks, fallback_ids, crop_ids = [], [], []
        frame0_bgr = cv2.imread(os.path.join(fdir, frames[0]))
        for oid, (x1, y1, x2, y2) in zip(obj_ids, boxes):
            pts = torch.tensor([[x1 / W, y1 / H], [x2 / W, y2 / H]], dtype=torch.float32)
            lbl = torch.tensor([2, 3], dtype=torch.int32)
            _, out_ids, _, vmasks = model.add_new_points(scratch, 0, oid, pts, lbl, clear_old_points=True,
                                                         rel_coordinates=True)
            k = list(out_ids).index(oid)
            m = vmasks[k, 0] > 0
            box_area = max(1.0, (x2 - x1) * (y2 - y1))
            if args.crop_prompt and float(m.sum()) < args.min_mask_frac * box_area:
                mc = crop_prompt_mask(model, frame0_bgr, (x1, y1, x2, y2))
                if mc.sum() >= args.min_mask_frac * box_area:
                    m = torch.from_numpy(mc).to(m.device)
                    crop_ids.append(oid)
            if args.min_mask_frac > 0 and float(m.sum()) < args.min_mask_frac * box_area:
                # small players (~20-45 px wide at 1080p, ~10-25 px at the model's 1008 px input) often get an empty
                # or tiny mask from the box prompt; seed them with an ellipse inscribed in the box instead
                e = np.zeros((H, W), np.uint8)
                cv2.ellipse(e, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            (max(1, int((x2 - x1) * 0.42)), max(1, int((y2 - y1) * 0.47))), 0, 0, 360, 1, -1)
                m = torch.from_numpy(e.astype(bool)).to(m.device)
                fallback_ids.append(oid)
            init_masks.append(m)
        del scratch
        init_masks = torch.stack(init_masks)
        # 2) all masks into one state together -> shared multiplex buckets
        state = new_state()
        model.add_new_masks(state, 0, obj_ids, init_masks)
        model.propagate_in_video_preflight(state, run_mem_encoder=True)
        ms = state["multiplex_state"]
        stats["buckets"] = [[int(x) for x in b] for b in ms.assignments]
        torch.cuda.synchronize(); stats["prompt_s"] = round(time.perf_counter() - t0, 2)
        stats["init_mask_areas"] = [int(m.sum()) for m in init_masks]
        stats["ellipse_fallback_ids"] = fallback_ids
        stats["crop_prompt_ids"] = crop_ids

        # 3) propagate
        per_frame = [None] * T
        ids_arr = np.array(obj_ids)
        tboxes = np.full((T, len(obj_ids), 4), np.nan, np.float32)
        present = np.zeros((T, len(obj_ids)), bool)
        scores = np.full((T, len(obj_ids)), np.nan, np.float32)
        frame_times = []
        torch.cuda.synchronize(); t_prev = time.perf_counter(); t_start = t_prev
        for fidx, out_ids, _, vmasks, oscores in model.propagate_in_video(
                state, start_frame_idx=0, max_frame_num_to_track=T, reverse=False, tqdm_disable=True):
            bm = (vmasks[:, 0] > 0).cpu().numpy()
            sc = oscores.float().view(-1).cpu().numpy()
            torch.cuda.synchronize(); now = time.perf_counter()
            frame_times.append(now - t_prev)
            objs = []
            for k, oid in enumerate(out_ids):
                j = obj_ids.index(oid)
                scores[fidx, j] = sc[k]
                b = mask_box(bm[k])
                if sc[k] <= 0 or b is None:
                    continue
                present[fidx, j] = True
                tboxes[fidx, j] = b
                rle = mask_utils.encode(np.asfortranarray(bm[k].astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                objs.append({"id": int(oid), "score": round(float(sc[k]), 3), "box_xyxy": b, "rle": rle})
            per_frame[fidx] = objs
            t_prev = time.perf_counter()
        torch.cuda.synchronize()
        prop_s = time.perf_counter() - t_start

    stats["propagate_s"] = round(prop_s, 2)
    stats["propagate_s_per_frame_mean"] = round(float(np.mean(frame_times[1:] if len(frame_times) > 1 else frame_times)), 4)
    stats["propagate_s_per_frame_median"] = round(float(np.median(frame_times)), 4)
    stats["peak_gpu_mem_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    stats["peak_gpu_mem_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    alive = present.sum(1)
    stats["objects_alive_first_frame"] = int(alive[0])
    stats["objects_alive_last_frame"] = int(alive[-1])
    stats["objects_alive_min"] = int(alive.min())
    stats["objects_alive_per_frame"] = alive.tolist()
    stats["objects_present_every_frame"] = int(present.all(0).sum())
    stats["empty_frames"] = [i for i, p in enumerate(per_frame) if p is None]

    json.dump({"video": meta["video"], "start": meta["start"], "width": W, "height": H, "fps": meta["fps"],
               "frames": [{"frame_idx": i, "objects": p or []} for i, p in enumerate(per_frame)]},
              open(os.path.join(args.out_dir, "masks.json"), "w"))
    np.savez_compressed(os.path.join(args.out_dir, "tracks.npz"), ids=ids_arr, boxes=tboxes, present=present,
                        scores=scores)
    if args.preview:
        write_preview(os.path.join(args.out_dir, "preview.mp4"), fdir, frames, [p or [] for p in per_frame],
                      meta["fps"])
    json.dump(stats, open(os.path.join(args.out_dir, "stats.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("objects_alive_per_frame", "init_mask_areas",
                                                                   "init_det_scores", "buckets")}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--video", required=True)
    e.add_argument("--start", type=float, default=0.0, help="0 = from the first frame")
    e.add_argument("--end", type=float, default=0.0, help="0 = to the last frame")
    e.add_argument("--fps", type=float, default=30, help="0 = native frames/timing (writes frames_pts.json)")
    e.add_argument("--width", type=int, default=0, help="0 = keep source resolution")
    e.add_argument("--out-dir", required=True)
    t = sub.add_parser("track")
    t.add_argument("--out-dir", required=True)
    g = t.add_mutually_exclusive_group(required=True)
    g.add_argument("--init-dets")
    g.add_argument("--init-boxes")
    t.add_argument("--max-objects", type=int, default=24)
    t.add_argument("--min-score", type=float, default=0.4)
    t.add_argument("--max-frames", type=int, default=0)
    t.add_argument("--dtype", choices=DTYPES, default="bf16")
    t.add_argument("--ckpt", default=DEFAULT_CKPT)
    t.add_argument("--preview", action="store_true")
    t.add_argument("--crop-prompt", type=int, default=1,
                   help="1: re-prompt small/empty box masks on a 3x crop resized to the model input, before the ellipse fallback")
    t.add_argument("--min-mask-frac", type=float, default=0.25,
                   help="box-prompt masks smaller than this fraction of the box are replaced by an inscribed ellipse (0=off)")
    args = ap.parse_args()
    {"extract": cmd_extract, "track": cmd_track}[args.cmd](args)


if __name__ == "__main__":
    main()
