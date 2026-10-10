"""OCR check of an edited image (big word + url row)."""
import sys, os
import numpy as np
from PIL import Image
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)


def ocr(arr, scale=3, name=''):
    from rapidocr_onnxruntime import RapidOCR
    eng = RapidOCR()
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    p = f'/tmp/ocr_{name}.png'
    im.save(p)
    res, _ = eng(p)
    if not res:
        return []
    return [(r[1], round(float(r[2]), 2), [int(v / scale) for v in r[0][0]] + [int(v / scale) for v in r[0][2]]) for r in res]


if __name__ == '__main__':
    import importlib
    mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else 'edit_banner')
    frames, dur, loop, size = mod.load_frames(mod.SRC)
    med = np.median(frames, axis=0)
    plan = mod.build(med, verbose=False)
    edited = mod.apply_edit(med, med, plan)
    lum = edited.mean(axis=2)
    print('BIG  before:', ocr(med[110:270, 0:450], 3, 'bigb'))
    print('BIG  after :', ocr(edited[110:270, 0:450], 3, 'biga'))
    print('URL  before:', ocr(med[295:370, 0:450], 4, 'urlb'))
    print('URL  after :', ocr(edited[295:370, 0:450], 4, 'urla'))
