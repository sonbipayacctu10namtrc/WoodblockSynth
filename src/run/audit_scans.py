"""
audit_scans.py — soát mặt chính (mặt khắc) của cả một thư mục scan 3D trước khi render.

  python src/run/audit_scans.py --scans <thư mục scan> --out outputs/audit
  python src/run/audit_scans.py --scans <thư mục scan> --out ... --manifest audit.csv   # áp chỉ định tay

Mỗi scan: tự xác định mặt khắc (giả định chỉ MỘT mặt mang thông tin) -> <out>/<tên>_faces.jpg (nhìn thẳng các
mặt ứng viên; khung xanh = tự chọn & tin cậy, vàng = cần xem, tím = chỉ định tay) + một dòng trong <out>/audit.csv.
Quy trình: mở ảnh các dòng confident=False, sửa cột face (vd. C+ / C-) và rot90 (0-3, xoay 90°) nếu cần,
rồi dùng chính audit.csv làm --manifest cho run_smoke.py / notebook Kaggle (manifest.csv).
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import cv2
import mocban_render as mr

FIELDS = ["mesh", "texture", "face", "rot90", "confident", "reason", "ratio", "auto_face", "scores", "mesh_path"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", required=True, help="thư mục chứa scan (tìm đệ quy GLB/OBJ/STL/PLY)")
    ap.add_argument("--out", default="outputs/audit")
    ap.add_argument("--manifest", default=None, help="CSV chỉ định tay (vd. audit.csv đã sửa)")
    ap.add_argument("--extent", type=float, default=200.0, help="scale cạnh dài nhất của khối về N mm")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = mr.read_manifest(args.manifest) if args.manifest else {}

    meshes = sorted(p for p in Path(args.scans).rglob("*") if p.suffix.lower() in mr.MESH_EXT)
    if not meshes:
        raise SystemExit(f"Không thấy mesh {mr.MESH_EXT} trong {args.scans}")
    names = [p.name for p in meshes]
    dup = {n for n in names if names.count(n) > 1}
    if dup:  # manifest khoá theo tên file -> tên phải duy nhất
        raise SystemExit(f"Trùng tên file mesh (manifest không phân biệt được): {sorted(dup)}")

    rows = []
    for p in meshes:
        t0 = time.time()
        try:
            info, img, tex = mr.audit_scan(p, args.extent, manifest.get(p.name))
            cv2.imwrite(str(out / f"{p.stem}_faces.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            rows.append({"mesh": p.name, "texture": Path(tex).name if tex else "", "face": info["face"],
                         "rot90": info["rot90"], "confident": info["confident"], "reason": info["reason"],
                         "ratio": info.get("ratio"), "auto_face": info["auto_face"],
                         "scores": json.dumps(info["scores"]), "mesh_path": str(p)})
            flag = "OK " if info["confident"] else "XEM"
            print(f"[{flag}] {p.name}: mặt {info['face']} — {info['reason']} ({time.time() - t0:.1f}s)")
        except Exception as e:  # một scan lỗi không làm dừng cả lô
            rows.append({"mesh": p.name, "face": "auto", "rot90": 0, "confident": False,
                         "reason": f"LỖI: {e!r}"[:300], "mesh_path": str(p)})
            print(f"[LỖI] {p.name}: {e!r}")

    with open(out / "audit.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_bad = sum(not r["confident"] for r in rows)
    print(f"\n{len(rows)} scan, {n_bad} cần xem -> {out / 'audit.csv'}")


if __name__ == "__main__":
    main()
