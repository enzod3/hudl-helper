#!/usr/bin/env python
"""Fine-tune RF-DETR Large on a Roboflow-YOLO dataset (rfdetr_env python, from ~/sam3_work).
  train_ft.py DATASET_DIR OUT_DIR [--epochs 15] [--batch 4] [--accum 4] [--lr 1e-4]
Starts from the COCO RFDETRLarge weights; class list from DATASET_DIR/data.yaml. Takes the GPU lock.
"""
import argparse, fcntl, json, os, time

os.environ.setdefault("RF_HOME", os.path.expanduser("~/sam3_work/rfdetr_weights"))
from rfdetr import RFDETRLarge  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset"); ap.add_argument("out")
ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--accum", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-4)
a = ap.parse_args()
lock = open(os.path.expanduser("~/sam3_work/logs/gpu.lock"), "w")
fcntl.flock(lock, fcntl.LOCK_EX)
t0 = time.time()
model = RFDETRLarge()
model.train(dataset_dir=a.dataset, output_dir=a.out, epochs=a.epochs, batch_size=a.batch, grad_accum_steps=a.accum,
            lr=a.lr, device="cuda", notes={"dataset": a.dataset})
json.dump({"seconds": round(time.time() - t0, 1), "epochs": a.epochs, "dataset": a.dataset},
          open(os.path.join(a.out, "train_done.json"), "w"))
print("TRAIN_DONE", round(time.time() - t0, 1))
