"""Put every src/multiview/* module dir on sys.path (the pipeline uses flat sibling imports) and stub the optional
runtime-only dependencies (ffmpeg wrapper, COCO RLE) so the pure-geometry tests run without them."""
import glob, os, sys, types

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = glob.glob(os.path.join(ROOT, "src", "multiview", "*", ""))
for name in ("imageio_ffmpeg", "pycocotools", "pycocotools.mask"):
    try:
        __import__(name)
    except ImportError:
        mod = types.ModuleType(name)
        mod.get_ffmpeg_exe = lambda: "ffmpeg"
        sys.modules[name] = mod
        if name == "pycocotools.mask":
            sys.modules["pycocotools"].mask = mod
