"""
check_face_detection.py — kiểm độ ổn định của việc tự xác định mặt khắc trên scan bất kỳ, KHÔNG cần nhãn.

  python tests/check_face_detection.py --mesh <file hoặc thư mục> --k 12

Mỗi scan: xoay + dịch ngẫu nhiên k lần rồi xác định lại mặt khắc. Kết quả đáng tin thì phải:
  - luôn chọn CÙNG một mặt vật lý (lệch pháp tuyến < 3°),
  - không lần nào lật gương (det > 0),
  - độ tin cậy ổn định (tỉ lệ chi tiết không dao động quanh ngưỡng).
Bất biến khi xoay chưa chứng minh là chọn ĐÚNG mặt -> vẫn phải xem ảnh soát (audit_scans.py).
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "core"))
import numpy as np
import trimesh
import mocban_render as mr


def up_normal(out: trimesh.Trimesh) -> tuple[np.ndarray, float]:
    """Pháp tuyến mặt đã chọn (+Z sau khi xoay), trong toạ độ mesh đưa vào prepare_scan; kèm det phép biến đổi."""
    A = np.array(out.metadata["to_render_frame"])[:3, :3]
    n = A.T @ [0.0, 0.0, 1.0]
    return n / np.linalg.norm(n), float(np.linalg.det(A))


def check(path: Path, k: int, rng: np.random.Generator) -> bool:
    raw = trimesh.load(str(path), process=False)
    if isinstance(raw, trimesh.Scene):
        raw = raw.to_mesh()
    raw.visual = trimesh.visual.ColorVisuals(raw)          # bỏ texture: không ảnh hưởng hình học, copy nhẹ hơn
    t0 = time.time()
    m = mr.prepare_scan(raw)
    info = mr.detect_main_face(m)
    n0, _ = up_normal(mr.orient_to_face(m, info["face"]))
    errs, dets, ratios, confs = [], [], [], []
    for _ in range(k):
        Q = trimesh.transformations.random_rotation_matrix(rng.random(3))
        Q[:3, 3] = rng.uniform(-1, 1, 3) * raw.bounding_box.extents.max()
        r = raw.copy(); r.apply_transform(Q)
        mo = mr.prepare_scan(r)
        inf = mr.detect_main_face(mo)
        n, det = up_normal(mr.orient_to_face(mo, inf["face"]))
        errs.append(math.degrees(math.acos(np.clip(np.dot(Q[:3, :3].T @ n, n0), -1, 1))))
        dets.append(det); ratios.append(inf["ratio"]); confs.append(inf["confident"])
    same = max(errs) < 3
    ok = same and min(dets) > 0 and all(c == info["confident"] for c in confs)
    rs = [r for r in ratios if r is not None]
    print(f"[{'OK ' if ok else 'LỖI'}] {path.name}: mặt {info['face']} ({info['reason']})")
    print(f"      {k} lần xoay: lệch tối đa {max(errs):.2f}° | lật gương: {'không' if min(dets) > 0 else 'CÓ'} | "
          f"tin cậy {sum(confs)}/{k}" + (f" | tỉ lệ chi tiết {min(rs):.2f}–{max(rs):.2f}" if rs else "")
          + f" | {time.time() - t0:.0f}s")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="file mesh hoặc thư mục (tìm đệ quy GLB/OBJ/STL/PLY)")
    ap.add_argument("--k", type=int, default=12, help="số lần xoay ngẫu nhiên mỗi scan")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src = Path(args.mesh)
    meshes = sorted(p for p in src.rglob("*") if p.suffix.lower() in mr.MESH_EXT) if src.is_dir() else [src]
    rng = np.random.default_rng(args.seed)
    bad = [p.name for p in meshes if not check(p, args.k, rng)]
    print(f"\n{len(meshes) - len(bad)}/{len(meshes)} scan ổn định" + (f"; KHÔNG ổn định: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
