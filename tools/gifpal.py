"""Extract / apply per-frame GIF colour tables.

The banner GIF carries a 256-colour LOCAL colour table on every frame, so a
re-encode only stays pixel-exact if each frame is requantised with *its own*
table.  PIL's ``Image.getpalette()`` returns the (2-colour) global table for
such files, which is why a naive re-encode shifts every pixel.
"""
from PIL import Image
import numpy as np


def local_palettes(path):
    """Return [bytes(768), ...], one per frame (local table, else global)."""
    b = open(path, 'rb').read()
    assert b[:6] in (b'GIF87a', b'GIF89a')
    gct_flag = bool(b[10] & 0x80)
    gct_size = 2 ** ((b[10] & 7) + 1)
    i = 13 + (3 * gct_size if gct_flag else 0)
    gct = b[i - 3 * gct_size:i] if gct_flag else None
    pals = []
    while i < len(b):
        t = b[i]
        if t == 0x21:                      # extension
            i += 2
            while b[i] != 0:
                i += 1 + b[i]
            i += 1
        elif t == 0x2C:                    # image descriptor
            packed = b[i + 9]
            i += 10
            if packed & 0x80:              # local colour table
                n = 2 ** ((packed & 7) + 1)
                pals.append(b[i:i + 3 * n])
                i += 3 * n
            else:
                pals.append(gct)
            i += 1                         # LZW min code size
            while b[i] != 0:
                i += 1 + b[i]
            i += 1
        elif t == 0x3B:
            break
        else:
            i += 1
    return pals


def index_map(rgb, pal):
    """Map an RGB frame onto `pal` (N,3).

    Colours that already exist in the table map to their exact index, so
    untouched pixels survive a re-encode byte-for-byte; anything new (pixels
    the edit introduced) falls back to nearest colour.
    """
    pal = np.frombuffer(bytes(pal)[:768 * (len(bytes(pal)) // 768)],
                        dtype=np.uint8).astype(np.int16).reshape(-1, 3)
    lut = {}
    for j, c in enumerate(pal):
        lut.setdefault((int(c[0]), int(c[1]), int(c[2])), j)
    flat = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    cols, inv = np.unique(flat, axis=0, return_inverse=True)
    idx = np.empty(len(cols), dtype=np.int64)
    for k, c in enumerate(cols):
        t = (int(c[0]), int(c[1]), int(c[2]))
        j = lut.get(t)
        if j is None:                      # new colour -> nearest (int32, no overflow)
            j = int(np.argmin(((pal.astype(np.int32) -
                                np.asarray(c, np.int32)) ** 2).sum(1)))
        idx[k] = j
    return idx[np.asarray(inv).reshape(-1)].reshape(rgb.shape[:2])


def save_exact(frames, path, src, dur=None, loop=0):
    """Re-encode `frames` reusing every frame's own colour table."""
    pals = local_palettes(src)
    assert len(pals) == len(frames), (len(pals), len(frames))
    imgs = []
    for i, f in enumerate(frames):
        p = pals[i]
        im = Image.fromarray(index_map(f, p).astype(np.uint8), mode='P')
        im.putpalette(list(p) + [0] * (768 - len(p)))
        imgs.append(im)
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=list(dur) if dur else 40, loop=loop, optimize=False)
    return path


if __name__ == '__main__':
    import sys, os
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/rt.gif'
    im = Image.open(src)
    orig, durs = [], []
    for i in range(im.n_frames):
        im.seek(i)
        orig.append(np.asarray(im.convert('RGB')).astype(np.uint8))
        durs.append(im.info.get('duration', 40))
    orig = np.stack(orig)
    save_exact(orig, out, src, durs, im.info.get('loop', 0))
    from PIL import ImageSequence
    im2 = Image.open(out)
    back = []
    for i in range(im2.n_frames):
        im2.seek(i)
        back.append(np.asarray(im2.convert('RGB')).astype(np.uint8))
    back = np.stack(back)
    d = np.abs(back.astype(int) - orig.astype(int)).mean(3)
    print('round-trip %s: frames %d  mean %.5f  max %.1f  px>0 %d'
          % (src, len(orig), d.mean(), d.max(), (d > 0).sum()))
