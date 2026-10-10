"""Render the edited banner GIF:  'AMV'->'AMM'  and  '.GG/ADOPTMEVALUE'->'.GG/ADOPTMEMARKET'."""
import os, sys
import numpy as np
from PIL import Image, ImageSequence
import edit_banner as E

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, '..')


SRC = os.environ.get('AMM_SRC') or E.SRC


def main(out_path=None, dry=False):
    frames, dur, loop, size = E.load_frames(SRC)
    med = np.median(frames, axis=0)
    plan = E.build(med)
    b, u = plan['big'], plan['url']

    url_rm = np.zeros(med.shape[:2], bool)
    url_rm[E.URL_ROW:E.URL_ROW + 85] |= u['remove']

    h, w = b['src_rgb'].shape[:2]
    y0, x0 = b['ny0'], b['nx0']

    # residual check inside the blocks (outside the ink) -> is the bg animated?
    for nm, m in (('big', b['remove']), ('url', url_rm)):
        rr = [np.abs(frames[i] - med)[m].mean() for i in range(0, 75, 18)]
        print(f'  |f-med| inside {nm} ink: ' + ' '.join(f'{v:.1f}' for v in rr))

    out = []
    for i in range(len(frames)):
        f = frames[i]
        o = f.copy()
        # big word: remove the old letter from this frame, redraw with this frame's shimmer
        big_bg = E.inpaint(f, b['remove'], 8)
        kb = np.clip(E.ratio_field(f, med, b['ink_all']), 0.35, 1.20)
        a = b['src_a'] * kb[y0:y0 + h, x0:x0 + w]
        a = a[..., None]
        blk = big_bg[y0:y0 + h, x0:x0 + w]
        o[y0:y0 + h, x0:x0 + w] = blk * (1 - a) + np.clip(b['src_rgb'] *
                                                          kb[y0:y0 + h, x0:x0 + w][..., None], 0, 255) * a
        # url: same treatment
        o[E.URL_ROW:E.URL_ROW + 85] = E.inpaint(f, url_rm, 5)[E.URL_ROW:E.URL_ROW + 85]
        kf = E.ratio_field(f, med, u['ink_all'])
        for p in u['plan']:
            g = u['P'][p['c']]['a']
            kk = np.clip(kf[E.URL_ROW + p['y0']:E.URL_ROW + p['y0'] + g.shape[0],
                             p['x']:p['x'] + g.shape[1]], 0.30, 1.20)[..., None]
            col = np.clip(E.URL_INK * kk, 0, 255)
            reg = o[E.URL_ROW + p['y0']:E.URL_ROW + p['y0'] + g.shape[0],
                    p['x']:p['x'] + g.shape[1]]
            o[E.URL_ROW + p['y0']:E.URL_ROW + p['y0'] + g.shape[0],
              p['x']:p['x'] + g.shape[1]] = reg * (1 - g[..., None]) + col * g[..., None]
        out.append(np.clip(o, 0, 255).astype(np.uint8))
    out = np.stack(out)

    if dry:
        return out, dur, loop, med
    save_gif(out, dur, loop, out_path)
    return out, dur, loop, med


def save_like_original(frames, dur, loop, path, src=None):
    """Re-encode reusing every source frame's own colour table.

    Untouched pixels therefore survive byte-for-byte; only the edited region
    can differ.
    """
    import gifpal
    return gifpal.save_exact(frames, path, src or SRC, list(dur), loop)


def save_gif(frames, dur, loop, path):
    return save_like_original(frames, dur, loop, path)


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else '/tmp/banner_new.gif'
    out, dur, loop, med = main(p)
    print('wrote', p, os.path.getsize(p) // 1024, 'KB', out.shape, 'dur', sorted(set(dur)))
