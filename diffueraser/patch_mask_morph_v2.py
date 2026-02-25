import re
from pathlib import Path
path = Path("diffueraser_OR.py")
s = path.read_text()

# 把 _postprocess 里那段“(arr>0) + dilate”替换成“(arr>0) + erode + dilate”
pat = re.compile(
    r"m\s*=\s*\(arr\s*>\s*0\)\.astype\(np\.uint8\)\s*\n"
    r"(?:[ \t]*#.*\n)*"
    r"[ \t]*m\s*=\s*cv2\.dilate\(\s*m\s*,\s*cv2\.getStructuringElement\(\s*cv2\.MORPH_RECT\s*,\s*\(3\s*,\s*3\)\s*\)\s*,\s*iterations\s*=\s*mask_dilation_iter\s*\)\s*\n"
)

rep = (
    "m = (arr > 0).astype(np.uint8)\n"
    "        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)\n"
    "        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=mask_dilation_iter)\n"
)

s2, n = pat.subn(rep, s, count=1)
if n != 1:
    raise SystemExit(f"ERROR: replaced {n} blocks (expected 1).")
path.write_text(s2)
print("OK: patched with official-style inline kernel calls.")
