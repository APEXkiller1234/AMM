"""AMV -> AMM for the 256x256 avatar GIF (a_76167db1a951167e537ce18edd2b3d63.gif).

The avatar wordmark is the same face as the banner's, at ~0.57x.  The three
glyphs are harvested as connected ink blobs (min-channel > 195, 3x3 close):

    A  x 20- 92   y126-213      M  x 89-162  y133-213
    V  x167-239   y131-219

The V blob is erased (cv2 TELEA inpaint) and a copy of the M blob -- including
its ~3px dark outline -- is stamped in its place, cap-aligned to the source M.

Per frame the *residual* (frame - median) is used rather than the pixels, so the
animated illustration behind the wordmark keeps moving; inside the erased slot
the residual field is itself inpainted from the surrounding background, and
inside the stamped M the source M's own residual is used.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
from scipy import ndimage
from PIL import Image

from imgutils import load_frames
import edit_banner as E
import render_banner as R
import verify

SRC = 'a_76167db1a951167e537ce18edd2b3d63.gif'
AV_INK = 195          # min-channel cut that keeps the cream wordmark
OUTLINE = 3           # glyph ink -> outer edge of the dark outline
ERASE = 4             # dilation of the V ink used as the removal mask

# generous harvest boxes around each glyph (y0, y1, x0, x1)
BOX = dict(A=(126, 214, 14, 92),
           M=(126, 220, 84, 166),
           V=(124, 220, 162, 244))

_f = _med = None


def frames():
    global _f, _med
    if _f is None:
        _f, _dur, _loop, _size = load_frames(SRC)
        _med = np.median(_f, axis=0)
    return _f, _med


def ink(med):
    return ndimage.binary_closing(med.min(axis=2) > AV_INK, np.ones((3, 3)))


def blob(med, name):
    """Largest connected ink blob inside BOX[name].

    Returns (mask_cropped, rgb_cropped, y0, x0, mask_full).
    """
    y0, y1, x0, x1 = BOX[name]
    sub = ink(med)[y0:y1, x0:x1]
    lab, n = ndimage.label(sub)
    if n == 0:
        raise RuntimeError('no ink in box %s' % name)
    sz = ndimage.sum(sub, lab, range(1, n + 1))
    keep = lab == (int(np.argmax(sz)) + 1)
    ys, xs = np.nonzero(keep)
    by0, by1, bx0, bx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = keep[by0:by1, bx0:bx1]
    full = np.zeros(med.shape[:2], bool)
    full[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1] = m
    return m, med[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1].copy(), \
        y0 + by0, x0 + bx0, full


def dil(mask, k):
    return ndimage.binary_dilation(mask, np.ones((3, 3)), iterations=k)


def soft(mask, sigma=0.8):
    """Anti-aliased coverage alpha; keeps the glyph's interior + outline."""
    return np.clip(ndimage.gaussian_filter(mask.astype(np.float32), sigma), 0, 1)


# ------------------------------------------------------------------ editing
def stamp(med, dy=2, dx=0, r=9):
    """Erase the V, stamp the M where it was.

    Returns (edited_median, erase_mask, newM_core_mask, paste_box).
    """
    vm, vrgb, vy0, vx0, vf = blob(med, 'V')
    mm, mrgb, my0, mx0, mf = blob(med, 'M')

    erase = dil(vf, ERASE)
    edit = E.inpaint(med, erase, r=r)

    # paste rect = the M's ink grown by its outline
    reg = dil(mf, OUTLINE)
    ys, xs = np.nonzero(reg)
    gy0, gy1, gx0, gx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    src_rgb = med[gy0:gy1, gx0:gx1].copy()
    src_a = soft(reg[gy0:gy1, gx0:gx1])

    # destination: align the new M's ink-core top-left with the V's ink top-left
    ty0, tx0 = vy0 + dy, vx0 + dx
    dy0, dx0 = ty0 - (my0 - gy0), tx0 - (mx0 - gx0)
    H = min(src_rgb.shape[0], med.shape[0] - dy0)
    W = min(src_rgb.shape[1], med.shape[1] - dx0)

    a = np.zeros(med.shape[:2], np.float32)
    s = np.zeros_like(edit)
    a[dy0:dy0 + H, dx0:dx0 + W] = src_a[:H, :W]
    s[dy0:dy0 + H, dx0:dx0 + W] = src_rgb[:H, :W]
    edit = edit * (1 - a[..., None]) + s * a[..., None]

    newm = np.zeros(med.shape[:2], bool)
    newm[ty0:ty0 + mm.shape[0], tx0:tx0 + mm.shape[1]] = mm
    return np.clip(edit, 0, 255), erase, newm, (dy0, dx0, H, W)


def save_like_original(frames, path, dur=40, loop=0):
    """Re-encode reusing the source colour table -> untouched pixels identical."""
    import gifpal
    return gifpal.save_exact(frames, path, SRC,
                             [dur] * len(frames) if np.isscalar(dur) else list(dur),
                             loop)


def render(dy=2, dx=0, out_path=None, verbose=True):
    fr, med = frames()
    edited, erase, newm, box = stamp(med, dy=dy, dx=dx)
    mm, _, my0, mx0, _ = blob(med, 'M')
    vm, _, vy0, vx0, _ = blob(med, 'V')
    m = (dil(erase, 2) * 255).astype(np.uint8)

    out = []
    for i in range(len(fr)):
        res = fr[i] - med
        # inpaint the residual field so the erased slot keeps the background motion
        ri = np.clip((res + 255.0) * 0.5, 0, 255).astype(np.uint8)
        ri = cv2.inpaint(ri, m, 9, cv2.INPAINT_TELEA).astype(np.float32) * 2.0 - 255.0
        rf = res.copy()
        rf[erase] = ri[erase]
        # the stamped M keeps the source M's own temporal behaviour
        h = min(mm.shape[0], med.shape[0] - (vy0 + dy))
        w = min(mm.shape[1], med.shape[1] - (vx0 + dx))
        sub = rf[vy0 + dy:vy0 + dy + h, vx0 + dx:vx0 + dx + w]
        sub[mm[:h, :w]] = res[my0:my0 + h, mx0:mx0 + w][mm[:h, :w]]
        out.append(np.clip(edited + rf, 0, 255))
    out = np.clip(np.stack(out), 0, 255).astype(np.uint8)
    if out_path:
        save_like_original(out, out_path)
    return out, med, edited, box


if __name__ == '__main__':
    import sys as _s
    out = _s.argv[1] if len(_s.argv) > 1 else '/tmp/avatar_v1.gif'
    fr, med = frames()
    for n in 'AMV':
        mk, rgb, y0, x0, _ = blob(med, n)
        h, w = mk.shape
        print('%s: x%d-%d y%d-%d  %dx%d  area %d  ink mean %.1f'
              % (n, x0, x0 + w, y0, y0 + h, w, h, mk.sum(), rgb[mk].mean()))
    edited, erase, newm, box = stamp(med)
    print('paste box (y0,x0,H,W):', box)
    print('BEFORE:', verify.ocr(med, 3, 'avb'))
    print('AFTER :', verify.ocr(edited, 3, 'ava'))
    d = np.abs(edited - med).mean(axis=2)
    ys, xs = np.nonzero(d > 8)
    print('median edit region: y%d-%d x%d-%d (%d px)'
          % (ys.min(), ys.max(), xs.min(), xs.max(), (d > 8).sum()))
    outf, med, edited, box = render(out_path=out)
    print('wrote', out, outf.shape)
