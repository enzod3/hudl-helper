#!/usr/bin/env python
"""Ground-contact head (rfdetr_env python, from ~/sam3_work).

  train_foot.py --synth foot/foot_synth_wide.npz --real foot/real_dots.npz --out foot/run1

Model: small CNN on the 64x128 crop -> (u, v, log_sigma_u, log_sigma_v), Gaussian NLL (so it predicts its own
uncertainty). Training-time augmentation: random crop shift / scale (stands in for detector-box jitter, target moved
accordingly), brightness / contrast, blur. Stage 1: synthetic (5% held out). Stage 2: fine-tune on the real dots of
even clip ids. Evaluation on real dots of odd clip ids, in pixels and yards (through each frame's H):
box bottom-centre, mask bottom (current pipeline rule, where a mask exists), head (synthetic only), head (fine-tuned).
Writes OUT/eval.json and OUT/head.pt.
"""
import argparse, json, os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CW, CH = 64, 128


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 96), blk(96, 128))   # 128x64 -> 8x4
        self.h = nn.Sequential(nn.Flatten(), nn.Linear(128 * 8 * 4, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 4))

    def forward(self, x):
        o = self.h(self.f(x))
        return o[:, :2], o[:, 2:].clamp(-6, 2)


def augment(x, y, train):
    """x: (B,3,H,W) float 0..1, y: (B,2) crop-normalised. Random affine (shift/scale) with target update."""
    if not train:
        return x, y
    B = x.shape[0]
    s = torch.empty(B, device=x.device).uniform_(0.88, 1.12)
    tx = torch.empty(B, device=x.device).uniform_(-0.08, 0.08)
    ty = torch.empty(B, device=x.device).uniform_(-0.08, 0.08)
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = s; theta[:, 1, 1] = s; theta[:, 0, 2] = tx; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    # output pixel p samples input at s*p + t (normalised -1..1) -> target maps as p = (q - t) / s
    q = y * 2 - 1
    q = torch.stack([(q[:, 0] - tx) / s, (q[:, 1] - ty) / s], 1)
    y = (q + 1) / 2
    x = x * torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.7, 1.3) + torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.1, 0.1)
    if torch.rand(1).item() < 0.3:
        x = F.avg_pool2d(x, 3, 1, 1)
    return x.clamp(0, 1), y


def nll(mu, ls, y):
    return (0.5 * ((y - mu) / ls.exp()) ** 2 + ls).sum(1).mean()


def run(model, X, Y, epochs, lr, dev, bs=256, val=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * ((len(X) + bs - 1) // bs))
    Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y)
    log = []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            x = Xt[idx].to(dev).permute(0, 3, 1, 2).float() / 255.0
            y = Yt[idx].to(dev)
            x, y = augment(x, y, True)
            mu, ls = model(x)
            loss = nll(mu, ls, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(idx)
        msg = {"epoch": ep, "train_nll": round(tot / len(X), 4)}
        if val is not None:
            mu, _ = predict(model, val[0], dev)
            msg["val_err_crop_norm"] = round(float(np.median(np.linalg.norm(mu - val[1], axis=1))), 4)
        log.append(msg); print(msg, flush=True)
    return log


@torch.no_grad()
def predict(model, X, dev, bs=512):
    model.eval(); MU, SG = [], []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        mu, ls = model(x); MU.append(mu.cpu().numpy()); SG.append(ls.exp().cpu().numpy())
    return np.concatenate(MU), np.concatenate(SG)


def crop_box(box):
    x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1
    return x1 - 0.10 * w, y1 - 0.15 * h, x2 + 0.10 * w, y2 + 0.30 * h


def from_norm(box, uv):
    X1, Y1, X2, Y2 = crop_box(box)
    return np.array([X1 + uv[0] * (X2 - X1), Y1 + uv[1] * (Y2 - Y1)])


def to_field(H, p):
    q = np.linalg.inv(np.array(H)) @ np.array([p[0], p[1], 1.0])
    return q[:2] / q[2]


def stats(d):
    d = np.asarray(d)
    return {"n": int(len(d)), "median": round(float(np.median(d)), 3), "p95": round(float(np.percentile(d, 95)), 3),
            "mean": round(float(np.mean(d)), 3)} if len(d) else {"n": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True); ap.add_argument("--real", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--ft-epochs", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    S = np.load(a.synth); Xs, Ys = S["X"], S["Y"]
    n_val = len(Xs) // 20
    perm = np.random.permutation(len(Xs)); vi, ti = perm[:n_val], perm[n_val:]
    model = Head().to(dev)
    log1 = run(model, Xs[ti], Ys[ti], a.epochs, 2e-3, dev, val=(Xs[vi], Ys[vi]))
    R = np.load(a.real); Xr, Yr = R["X"], R["Y"]
    meta = json.load(open(a.real.replace(".npz", "_meta.json")))
    even = np.array([int(m["clip"]) % 2 == 0 for m in meta]); odd = ~even
    mu_s, sg_s = predict(model, Xr[odd], dev)
    torch.save(model.state_dict(), os.path.join(a.out, "head_synth.pt"))
    log2 = run(model, Xr[even], Yr[even], a.ft_epochs, 3e-4, dev, bs=64)
    mu_f, sg_f = predict(model, Xr[odd], dev)
    torch.save(model.state_dict(), os.path.join(a.out, "head.pt"))
    err = {k: [] for k in ("box_bottom", "mask_bottom", "mask_or_box", "head_synth", "head_finetuned")}
    err_px = {k: [] for k in err}
    calib = []
    for j, (m, i) in enumerate(zip([m for m, o in zip(meta, odd) if o], range(odd.sum()))):
        box, dot_yd, H = m["box"], np.array(m["dot_yd"]), m["H"]
        dot_px = np.array(m["dot_px"])
        cands = {"box_bottom": np.array([(box[0] + box[2]) / 2, box[3]]),
                 "head_synth": from_norm(box, mu_s[i]), "head_finetuned": from_norm(box, mu_f[i])}
        if m["mask_foot_px"]:
            cands["mask_bottom"] = np.array(m["mask_foot_px"])
        cands["mask_or_box"] = cands.get("mask_bottom", cands["box_bottom"])
        for k, pxy in cands.items():
            err[k].append(float(np.linalg.norm(to_field(H, pxy) - dot_yd)))
            err_px[k].append(float(np.linalg.norm(pxy - dot_px)))
        # uncertainty calibration: normalised error of the fine-tuned head in crop units
        z = (Yr[odd][i] - mu_f[i]) / np.maximum(sg_f[i], 1e-4)
        calib.append(float(np.sum(z ** 2)))
    res = {"real_dots_eval_odd": int(odd.sum()), "real_dots_finetune_even": int(even.sum()), "synthetic_train": int(len(ti)),
           "yards": {k: stats(v) for k, v in err.items()}, "pixels": {k: stats(v) for k, v in err_px.items()},
           "uncertainty_check": {"median_chi2_2dof": round(float(np.median(calib)), 3), "ideal": 1.386,
                                 "frac_within_95pct": round(float(np.mean(np.array(calib) <= 5.991)), 3)},
           "synth_log": log1[-1], "ft_log": log2[-1]}
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
