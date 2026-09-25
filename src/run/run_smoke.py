"""
run_smoke.py — chạy thử pipeline render trên mesh scan 3D (GLB/OBJ/STL/PLY).

  python src/run/run_smoke.py --mesh models3d/block.obj --n 12            # tự tìm texture + mặt khắc
  python src/run/run_smoke.py --mesh models3d --manifest audit.csv --n 8  # cả thư mục, áp chỉ định tay
  python src/run/run_smoke.py --mesh block.obj --texture block_4K.jpg --face C- --rot90 1

Mặt khắc tự xác định (giả định chỉ MỘT mặt mang thông tin); soát trước cả lô bằng audit_scans.py.
"""
import argparse
import os
import sys
import time
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import cv2
import numpy as np
import mocban_render as mr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="file mesh hoặc thư mục (tìm đệ quy GLB/OBJ/STL/PLY)")
    ap.add_argument("--texture", default="auto", help="ảnh texture rời; 'auto' = tự tìm cạnh mesh")
    ap.add_argument("--face", default="auto", help="mặt khắc: auto | C+ | C- | ... (tên trong ảnh soát)")
    ap.add_argument("--rot90", type=int, default=0, help="xoay thêm k x 90° trong mặt phẳng")
    ap.add_argument("--manifest", default=None, help="CSV mesh,texture,face,rot90 (vd. audit.csv đã sửa); ưu tiên hơn cờ")
    ap.add_argument("--out", default="outputs/smoke")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--size", default="1024x768")
    ap.add_argument("--preset", default="mixed", help="mixed | topdown | handheld | raking | closeup")
    ap.add_argument("--extent", type=float, default=200.0, help="scale cạnh lớn nhất của khối về N mm")
    ap.add_argument("--occluders", type=float, default=0.0, help="xác suất thêm vật che 2D (tay/thước/nhãn/lóa/bóng)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    W, H = map(int, args.size.split("x"))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = mr.read_manifest(args.manifest) if args.manifest else {}

    src = Path(args.mesh)
    meshes = sorted(p for p in src.rglob("*") if p.suffix.lower() in mr.MESH_EXT) if src.is_dir() else [src]
    if not meshes:
        raise SystemExit(f"Không thấy mesh {mr.MESH_EXT} trong {src}")

    for p in meshes:
        t0 = time.time()
        ov = manifest.get(p.name, {"texture": args.texture, "face": args.face, "rot90": args.rot90})
        m = mr.load_external_mesh(p, args.extent, texture_path=ov["texture"], face=ov["face"], rot90=ov["rot90"])
        fi = m.metadata["face_info"]
        print(f"[mesh] {p.name}: {len(m.vertices)} đỉnh, khối(mm) {np.round(m.bounding_box.extents, 1)}, "
              f"texture={m.metadata.get('source_texture') or ('nhúng sẵn' if mr._has_texture_image(m) else 'không')}, "
              f"{time.time() - t0:.1f}s")
        print(f"       mặt {fi['face']} ({'tin cậy' if fi['confident'] else 'CẦN XEM'}): {fi['reason']}")
        be = mr.PyrenderBackend(m, W, H)
        nm = p.stem.replace(" ", "_")
        t0 = time.time()
        recs = mr.render_dataset(be, out, args.n, tuple(m.metadata["block_size_mm"][:2]), W, H, args.preset,
                                 seed=args.seed, name=nm, occluder_prob=args.occluders,
                                 extra_meta={"source_mesh": p.name, "face_info": fi})
        be.close()
        print(f"[render] {args.n} ảnh trong {time.time() - t0:.1f}s ({(time.time() - t0) / args.n:.2f}s/ảnh)")
        cv2.imwrite(str(out / f"{nm}_sheet.jpg"), mr.contact_sheet([out / r["file"] for r in recs], cols=4))
    print("[done]", out)


if __name__ == "__main__":
    main()
