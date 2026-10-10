"""Helpers for inspecting / editing the AMM animated GIF assets."""
from PIL import Image, ImageSequence
import numpy as np

CHARS = " .:-=+*#%@"


def load_frames(path):
    im = Image.open(path)
    fr = []
    dur = []
    for f in ImageSequence.Iterator(im):
        fr.append(np.array(f.convert('RGB'), dtype=np.float32))
        dur.append(f.info.get('duration', 40))
    return np.stack(fr), dur, im.info.get('loop', 0), im.size


def art(arr, w, chars=CHARS, aspect=0.5, lo=None, hi=None):
    a = np.asarray(arr, dtype=np.float32)
    if lo is None:
        lo = a.min()
    if hi is None:
        hi = a.max()
    if hi > lo:
        a = (a - lo) / (hi - lo)
    else:
        a = np.zeros_like(a)
    h = max(1, int(a.shape[0] / a.shape[1] * w * aspect))
    src = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    res = Image.BILINEAR if w > a.shape[1] else Image.BOX
    im = src.resize((w, h), res)
    b = np.array(im, dtype=np.float32) / 255.0
    idx = np.clip((b * (len(chars) - 1)).astype(int), 0, len(chars) - 1)
    return "\n".join("".join(chars[i] for i in row) for row in idx)


def show(name, arr, w=140, **kw):
    print(f"--- {name} ---")
    print(art(arr, w, **kw))
