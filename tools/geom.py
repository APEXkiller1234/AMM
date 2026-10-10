"""Locate the letters of the big 'AMV' text and the URL text in both assets."""
import os, sys
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from imgutils import load_frames

BIG_INK = np.array([250.0, 237.0, 229.0])
URL_INK = np.array([253.0, 251.0, 249.0])
ROOT = os.path.join(HERE, '..')


def median(path):
    a, dur, loop, size = load_frames(path)
    return np.median(a, axis=0), dur, loop, size


def key_mask(med, ink, tol):
    return np.linalg.norm(med - ink, axis=2) < tol


def blobs(mask, min_area=200, close=5):
    mm = ndimage.binary_closing(mask, np.ones((close, close)))
    mm = ndimage.binary_opening(mm, np.ones((3, 3)))
    lab, n = ndimage.label(mm)
    objs = ndimage.find_objects(lab)
    sizes = ndimage.sum(mm, lab, range(1, n + 1))
    out = []
    for i in range(n):
        if sizes[i] < min_area:
            continue
        sl = objs[i]
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        out.append(dict(x0=x0, x1=x1, y0=y0, y1=y1, area=int(sizes[i]),
                        mask=(lab == i + 1)))
    out.sort(key=lambda d: d['x0'])
    return out, mm


if __name__ == '__main__':
    for name in ['a_ed4953007e2155c7cfeaf4a4e3283554.gif',
                 'a_76167db1a951167e537ce18edd2b3d63.gif']:
        med, dur, loop, size = median(os.path.join(ROOT, name))
        print('====', name, size)
        for tol in (30, 50, 70):
            m = key_mask(med, BIG_INK, tol)
            bs, mm = blobs(m, min_area=1500)
            print(f'  big-ink tol={tol}: {len(bs)} blobs')
            for b in bs:
                print(f'     x{b["x0"]}-{b["x1"]} y{b["y0"]}-{b["y1"]} '
                      f'w{b["x1"]-b["x0"]} h{b["y1"]-b["y0"]} area{b["area"]}')
