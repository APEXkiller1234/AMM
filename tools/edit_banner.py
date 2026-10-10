"""Rebuild the banner: big 'AMV' -> 'AMM' and URL 'VALUE' -> 'MARKET'."""
import os, sys
import numpy as np
from scipy import ndimage
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from imgutils import load_frames
from geom import median, ROOT

SRC = os.path.join(ROOT, 'a_ed4953007e2155c7cfeaf4a4e3283554.gif')
BIG_INK = np.array([250.0, 237.0, 229.0])
URL_INK = np.array([253.0, 251.0, 249.0])
URL_ROW = 300                      # med[300:385] holds the url text


# ---------------------------------------------------------------- helpers
def soft_alpha(med, ink, lo, hi):
    d = np.linalg.norm(med - ink, axis=2)
    return np.clip((hi - d) / (hi - lo), 0.0, 1.0)


def inpaint(rgb, mask, r=7):
    m = (ndimage.binary_dilation(mask, np.ones((r, r))) * 255).astype(np.uint8)
    return cv2.inpaint(np.clip(rgb, 0, 255).astype(np.uint8), m, r,
                       cv2.INPAINT_TELEA).astype(np.float32)


def trim(mask, frac=0.30):
    m = mask.copy()
    m[mask.sum(axis=1) < mask.sum(axis=1).max() * frac, :] = False
    m[:, mask.sum(axis=0) < mask.sum(axis=0).max() * frac] = False
    return m


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), xs.max() + 1, ys.min(), ys.max() + 1


# ---------------------------------------------------------------- geometry
def big_letters(med):
    m = np.linalg.norm(med - BIG_INK, axis=2) < 30
    keep = np.zeros_like(m); keep[100:300, 0:445] = True
    m &= keep
    m = ndimage.binary_closing(m, np.ones((5, 5)))
    m = ndimage.binary_opening(m, np.ones((3, 3)))
    lab, n = ndimage.label(m)
    objs = ndimage.find_objects(lab)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    out = []
    for i in range(n):
        if sizes[i] < 3000:
            continue
        sl = objs[i]
        g = np.zeros(m.shape, bool); g[sl] = (lab == i + 1)[sl]
        out.append(dict(x0=sl[1].start, x1=sl[1].stop, y0=sl[0].start, y1=sl[0].stop, g=g))
    out.sort(key=lambda d: d['x0'])
    return out


def url_glyphs(med):
    sub = med[URL_ROW:URL_ROW + 85, 0:450]
    m = sub.min(axis=2) > 200
    m = ndimage.binary_closing(m, np.ones((3, 3)))
    m = ndimage.binary_opening(m, np.ones((2, 2)))
    lab, n = ndimage.label(m)
    objs = ndimage.find_objects(lab)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    out = []
    for i in range(n):
        if sizes[i] < 90:
            continue
        sl = objs[i]
        x0, x1 = sl[1].start, sl[1].stop
        if x1 - x0 > 40:
            continue
        out.append(dict(x0=x0, x1=x1, y0=sl[0].start, y1=sl[0].stop,
                        g=(lab == i + 1)))
    out.sort(key=lambda d: d['x0'])
    return out


# ---------------------------------------------------------------- synth R/K
def synth(stem, H, W, kind, src=None):
    im = Image.new('L', (W, H), 0)
    dr = ImageDraw.Draw(im)
    if kind == 'R':
        ph, pw = src.shape
        im.paste(Image.fromarray((src * 255).astype(np.uint8)), (0, H - ph))
        dr.line([(int(0.46 * W), int(0.46 * H)), (W - 2, H - 1)], 255,
                width=int(round(stem)))
    else:
        dr.rectangle([0, 0, int(round(stem)), H], fill=255)
        jy = int(round(0.55 * H)); jx = int(round(stem * 0.4))
        dr.line([(jx, jy), (W - 1, 0)], 255, width=int(round(stem)))
        dr.line([(jx, jy), (W - 1, H - 1)], 255, width=int(round(stem)))
    a = np.array(im, dtype=np.float32) / 255.0
    a = ndimage.gaussian_filter(a, 0.65)
    return np.clip(a * 1.30, 0, 1)


# ---------------------------------------------------------------- plan
def build(med, verbose=True):
    L = big_letters(med)
    A, M, V = L[0], L[1], L[2]
    vm = V['g'].copy(); vm[:123, :] = False
    vx0, vx1, vy0, vy1 = bbox(vm)
    mmask = M['g']
    mx0, mx1, my0, my1 = bbox(mmask)
    # carry the glyph's ~5px dark outline along with it, and erase the V's too,
    # otherwise the old letter leaves a dark ghost behind the new one
    PAD = 7
    mm_full = mmask                            # already full-image sized
    grown = ndimage.binary_dilation(mm_full, np.ones((3, 3)), iterations=4)
    gy0, gy1, gx0, gx1 = my0 - PAD, my1 + PAD, mx0 - PAD, mx1 + PAD
    src_rgb = med[gy0:gy1, gx0:gx1]
    src_a = np.clip(ndimage.gaussian_filter(
        grown[gy0:gy1, gx0:gx1].astype(np.float32), 1.2), 0.0, 1.0)
    nx0, ny0 = vx0 - PAD, my0 - PAD           # same cap height / baseline as the real M
    big_remove = ndimage.binary_dilation(vm, np.ones((3, 3)), iterations=6)

    U = url_glyphs(med)
    a_url = soft_alpha(med, URL_INK, 150.0, 205.0)
    P = {}
    for c, i in [('M', 8), ('A', 11), ('E', 9), ('T', 7), ('P', 6), ('L', 12)]:
        d = U[i]
        P[c] = dict(g=d['g'][d['y0']:d['y1'], d['x0']:d['x1']],
                    a=a_url[URL_ROW + d['y0']:URL_ROW + d['y1'], d['x0']:d['x1']]
                    * d['g'][d['y0']:d['y1'], d['x0']:d['x1']],
                    y0=d['y0'], y1=d['y1'], x0=d['x0'], x1=d['x1'])
    Hh = P['M']['y1'] - P['M']['y0']
    stem = 6
    P['R'] = dict(a=synth(stem, Hh, 21, 'R', src=P['P']['g']), y0=P['M']['y0'], y1=P['M']['y1'])
    P['K'] = dict(a=synth(stem, Hh, 21, 'K'), y0=P['M']['y0'], y1=P['M']['y1'])

    # baseline drift: fit y1 against glyph centre
    cx = np.array([(d['x0'] + d['x1']) / 2 for d in U], float)
    by = np.array([d['y1'] for d in U], float)
    bcoef = np.polyfit(cx, by, 1)

    adv = dict(M=35, A=19, R=21, K=21, E=21, T=21)
    x = U[10]['x0']
    plan = []
    for c in 'MARKET':
        gw = P[c]['a'].shape[1]
        y1 = int(round(np.polyval(bcoef, x + gw / 2.0)))
        y0 = y1 - P[c]['a'].shape[0]
        plan.append(dict(c=c, x=x, y0=y0, y1=y1))
        x += adv[c]

    url_remove = np.zeros((85, med.shape[1]), bool)
    for i in (10, 11, 12, 13, 14):
        d = U[i]
        url_remove[d['y0']:d['y1'], d['x0']:d['x1']] |= d['g'][d['y0']:d['y1'], d['x0']:d['x1']]
    url_ink_all = np.zeros((85, med.shape[1]), bool)
    for d in U:
        url_ink_all[d['y0']:d['y1'], d['x0']:d['x1']] |= d['g'][d['y0']:d['y1'], d['x0']:d['x1']]

    if verbose:
        print('big letters', [(d['x0'], d['x1'], d['y0'], d['y1']) for d in L])
        print('V ink', vx0, vx1, vy0, vy1, ' M ink', mx0, mx1, my0, my1)
        print('new M at', nx0, ny0, src_rgb.shape)
        print('url plan', [(p['c'], p['x'], p['y0']) for p in plan], 'end', x)
    return dict(big=dict(src_rgb=src_rgb, src_a=src_a, nx0=nx0, ny0=ny0, remove=big_remove,
                         ink_all=(L[0]['g'] | L[1]['g'] | vm)),
                url=dict(P=P, plan=plan, remove=url_remove, ink_all=url_ink_all))


# ---------------------------------------------------------------- compositing
def ratio_field(frame, med, ink_mask, pad=8, cell=(26, 18)):
    ys, xs = np.nonzero(ink_mask)
    y0, y1 = max(0, ys.min() - pad), min(frame.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(frame.shape[1], xs.max() + 1 + pad)
    H, W = y1 - y0, x1 - x0
    gh, gw = max(1, H // cell[1]), max(1, W // cell[0])
    f = frame[y0:y1, x0:x1].max(axis=2)
    m = med[y0:y1, x0:x1].max(axis=2)
    ok = ink_mask[y0:y1, x0:x1] & (m > 120)
    r = np.where(ok, f / np.maximum(m, 1.0), np.nan)
    rh, rw = H / gh, W / gw
    grid = np.full((gh, gw), np.nan)
    for i in range(gh):
        for j in range(gw):
            v = r[int(i * rh):int((i + 1) * rh), int(j * rw):int((j + 1) * rw)]
            v = v[~np.isnan(v)]
            if v.size >= 6:
                grid[i, j] = np.median(v)
    g = np.nanmedian(r[~np.isnan(r)]) if np.any(~np.isnan(r)) else 1.0
    if not np.isfinite(g):
        g = 1.0
    grid = np.where(np.isnan(grid), g, grid)
    sm = np.array(Image.fromarray((np.clip(grid, 0.25, 1.8) * 200).astype(np.uint16))
                  .resize((W, H), Image.BILINEAR), dtype=np.float32) / 200.0
    sm = ndimage.gaussian_filter(sm, 7)
    out = np.ones(frame.shape[:2], np.float32)
    out[y0:y1, x0:x1] = sm
    return out


def apply_edit(frame, med, plan, ratio=None):
    out = frame.copy()
    # ---- big word
    b = plan['big']
    bg = inpaint(frame, b['remove'], 8)
    if ratio is not None:
        k = ratio['big'][b['ny0']:b['ny0'] + b['src_rgb'].shape[0],
                         b['nx0']:b['nx0'] + b['src_rgb'].shape[1]]
    else:
        k = 1.0
    h, w = b['src_rgb'].shape[:2]
    a = (b['src_a'] * np.clip(k, 0.3, 1.25))[..., None]
    reg = out[b['ny0']:b['ny0'] + h, b['nx0']:b['nx0'] + w]
    out[b['ny0']:b['ny0'] + h, b['nx0']:b['nx0'] + w] = reg * (1 - a) + b['src_rgb'] * a
    # ---- url
    u = plan['url']
    rmfull = np.zeros(frame.shape[:2], bool)
    rmfull[URL_ROW:URL_ROW + 85] |= u['remove']
    bg2 = inpaint(out, rmfull, 5)
    out = bg2
    for p in u['plan']:
        g = u['P'][p['c']]
        a = g['a']
        if ratio is not None:
            kk = ratio['url'][URL_ROW + p['y0']:URL_ROW + p['y0'] + a.shape[0],
                              p['x']:p['x'] + a.shape[1]]
        else:
            kk = 1.0
        col = np.clip(URL_INK * np.clip(kk, 0.3, 1.25)[..., None], 0, 255)
        reg = out[URL_ROW + p['y0']:URL_ROW + p['y0'] + a.shape[0], p['x']:p['x'] + a.shape[1]]
        out[URL_ROW + p['y0']:URL_ROW + p['y0'] + a.shape[0],
            p['x']:p['x'] + a.shape[1]] = reg * (1 - a[..., None]) + col * a[..., None]
    return np.clip(out, 0, 255)


if __name__ == '__main__':
    frames, dur, loop, size = load_frames(SRC)
    med = np.median(frames, axis=0)
    plan = build(med)
    edited = apply_edit(med, med, plan)
    from imgutils import show
    d = np.linalg.norm(edited - BIG_INK, axis=2)
    show('BIG WORD ink mask after', (d < 30)[100:300, 0:460].astype(float), 150, aspect=0.5)
    sub = edited[295:365, 0:450]
    show('URL after (min-channel>200)', (sub.min(axis=2) > 200).astype(float), 150, aspect=0.55)
