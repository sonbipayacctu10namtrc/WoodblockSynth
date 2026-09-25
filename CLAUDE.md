# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc trong repo này. Bối cảnh và lịch sử: [CONTEXT.md](CONTEXT.md). Phương pháp và số
liệu: [REPORT.md](REPORT.md). Cài đặt và cách chạy: [README.md](README.md).

## Dự án là gì

Sinh ảnh 2D từ **scan 3D thật** của mộc bản để làm dữ liệu huấn luyện OCR. Repo chỉ làm phần sinh dữ liệu, không có code
OCR. Có ba khối:
- **Bước chung**: chuẩn hoá scan, tự tìm mặt khắc.
- **Hướng A**: render giả ảnh chụp bằng pyrender, cần GPU.
- **Hướng B**: ảnh độ sâu + 6 phương pháp tăng cường hình học, chỉ cần CPU.

## Ngôn ngữ

- Người dùng nói tiếng Việt: trả lời, viết tài liệu, comment, docstring và nội dung notebook bằng **tiếng Việt**.
- Tên hàm, biến, file bằng tiếng Anh, theo đúng kiểu đang có.

## Lệnh

Chạy từ gốc repo, dùng Python của `.venv` (Windows: `.venv\Scripts\python`, Linux: `.venv/bin/python`).

```bash
python notebooks/build_notebook.py                          # sinh lại notebook; tự kiểm AST + docs + trùng tên
python -m pyflakes $(ls src/core/*/s*.py) src/run/*.py tests/*.py notebooks/*.py
python src/run/audit_scans.py --scans <thư mục scan> --out outputs/audit
python src/run/run_smoke.py --mesh <file|thư mục> --out outputs/smoke --n 4 --size 800x600
python tests/check_face_detection.py --mesh <file|thư mục> --k 2
```

Dữ liệu **không nằm trong repo**: đặt scan vào `input/<tên_scan>/` theo README (zip có zip lồng thì giải nén cả hai lớp).
Không ghi tên dữ liệu cụ thể (tên model, tên file, ID nguồn, tên bộ ảnh) vào bất kỳ file nào được commit (tài liệu,
notebook, `notebook_docs.py`, comment). Chi tiết dữ liệu chỉ nằm ở `input/DATA.md`, file local không commit. Mọi thứ trong `input/` (trừ `.gitkeep`) và `outputs/` đều bị `.gitignore` loại, đừng commit.

Pyflakes báo "imported but unused" trong `__init__.py` là **chủ ý** (re-export), vì vậy lệnh trên chỉ lint các file `s*.py`.

## Cấu trúc code

```
src/core/mocban_render/    s01_scan → s02_face → s03_shot → s04_camera → s05_renderer → s06_post → s07_dataset
src/core/mocban_enhance/   s01_projection → s02_height → s03_derivatives → s04_depth → s05_curvature → s06_msii
                           → s07_ao → s08_shading → s09_display → s10_grid
src/run/                   script dòng lệnh (sys.path trỏ vào src/core)
tests/                     check_face_detection.py
notebooks/                 build_notebook.py + notebook_docs.py → mocban_scan3d_kaggle.ipynb
```

- **Tên file `sNN_` = thứ tự pipeline = thứ tự mục trong notebook.** Số trong banner `# N. Tiêu đề` của file trùng với
  `NN` (mục con dùng hậu tố chữ, ví dụ `6b`).
- **`__init__.py` của mỗi package** xuất lại **mọi** tên, kể cả tên bắt đầu bằng `_`, để script gọi được `mr.xxx` /
  `me.xxx`. `FILES = [...]` ở cuối `__init__.py` quy định thứ tự ghép vào notebook.
- **Import giữa các file** dùng `from .sNN_x import tên`. Không được tạo import vòng tròn: `s09_display` (tiện ích) tách
  riêng khỏi `s10_grid` (dùng mọi phương pháp) chính là để tránh vòng.

## Notebook được sinh ra, không sửa tay

`notebooks/mocban_scan3d_kaggle.ipynb` do `build_notebook.py` sinh. Mọi thay đổi phải làm trong `src/core/`,
`notebook_docs.py` hoặc `build_notebook.py`, rồi build lại.

Builder hoạt động như sau:
- `package_src()` đọc các file theo `FILES`, gom import ngoài lên đầu, bỏ `from .x import` và `from __future__`, rồi ghép
  phần còn lại. Trong notebook mọi hàm dùng chung một namespace.
- Mỗi hàm/lớp thành một cell, đứng sau đoạn giải thích lấy từ `notebook_docs.py` (dict `RENDER` / `ENHANCE`, khoá = tên
  hàm; chunk hằng số dùng khoá `const:<tên biến đầu tiên>`; import dùng `import:<package>`).
- Mỗi banner mục cần một khoá trong `SECTION` của `notebook_docs.py`, trùng **nguyên văn** tiêu đề banner.
- Build **dừng** nếu: AST của các cell khác code gốc; có hàm thiếu giải thích; cell chạy gán đè tên của thư viện.
- Cell chạy gọi hàm trực tiếp, không có tiền tố: builder tự xoá `mr.` / `me.` trong cell chạy.

### Thêm hàm

1. Viết vào đúng file `sNN_` theo bước pipeline.
2. Thêm tên vào khối `from .sNN_x import (...)` tương ứng trong `__init__.py`.
3. Viết giải thích trong `notebook_docs.py`: **Làm gì**, **Đầu vào / Đầu ra**, **Cách làm** (có công thức), **Lưu ý**.
4. Chạy `build_notebook.py` và pyflakes.

### Thêm bước mới (file mới)

Chọn số `sNN_` đúng vị trí trong pipeline; nếu phải chèn giữa thì đánh số lại các file sau cho liền mạch. Thêm banner,
khoá `SECTION`, dòng mô tả trong docstring và `FILES` của `__init__.py`.

## Bất biến không được phá

Mỗi dòng dưới đây từng là một lỗi thật trên scan thật (chi tiết ở REPORT.md mục 8). Đừng "đơn giản hoá" lại.

**Nạp scan**
- Đọc mesh với `process=False`, **không** dùng `force="mesh"`: tuỳ chọn này làm mất texture → render xám.
- Ảnh texture nhỏ hơn 16 px là ảnh giữ chỗ 2×2 của trimesh (OBJ không có `.mtl`): coi như không có texture.
- Tìm texture rời lên tới 3 cấp thư mục; vòng tìm xa **bắt buộc** trùng tiền tố tên ≥ 4 ký tự (Kaggle giải nén zip lồng
  làm mesh sâu thêm một cấp).
- Dùng OBB, không dùng AABB. Mọi phép quay phải có **det = +1**; lật gương sẽ đảo chiều chữ khắc.

**Tìm mặt khắc**
- Độ chi tiết là **trung vị** |h − G_σ*h|. Độ lệch chuẩn và rugosity đều đã chọn sai mặt trên scan thật.
- Độ phủ tính bằng **phép đóng** hình thái, không dùng phép giãn. Mặt có độ phủ < 50 % không được xếp hạng.
- Hướng 0°/180° trong mặt phẳng không suy ra được từ hình học: để người chỉnh bằng `rot90`, đừng đoán.
- `manifest.csv` khoá theo **tên file mesh** (không kèm đường dẫn), nên cùng một file dùng được ở local và Kaggle.

**Render (hướng A)**
- Đặt `PYOPENGL_PLATFORM=egl` **trước** lần import OpenGL/pyrender đầu tiên (PyOpenGL khoá nền tảng ngay lần import đó).
- `pyrender` cài `--no-deps` kèm `PyOpenGL==3.1.7`; shim `np.infty = np.inf` phải chạy trước khi import pyrender.
- Texture gắn qua `TextureVisuals(uv, image)`; pyrender không đọc `PBRMaterial` của trimesh 5.
- Hướng "lên" của camera gắn với trục +Y của khối, kèm tham số `roll` riêng.

**Chiếu và tăng cường (hướng B)**
- `project_depth` **rasterize tam giác tại tâm pixel**, lấy tam giác cao nhất, không lọc theo pháp tuyến. **Không** rải
  điểm (splat): scan có nhiều lớp chồng nhau và ~9 % pháp tuyến bị lật, rải điểm tạo hố giả 14–35 mm.
- `_face_height` (chấm điểm mặt) vẫn rải điểm. Chấp nhận được vì nó chỉ lấy trung vị; **không dùng nó để tạo ảnh**.
- Đạo hàm bậc 2 dùng `_dkernel` (ép `Σk = 0`, `Σk·iⁿ/n! = 1`), **không** dùng thẳng `gaussian_filter(order=2)`: nhân đó có
  tổng khác 0, sai số H/K tới 138 % / 467 %. Kiểm trên mặt **có** thành phần hằng (bán cầu), vì mặt không có DC che lỗi.
- Frame **ảnh**: `x = cột`, `y = hàng`, pháp tuyến `normalize(−hx, −hy, 1)`. Copy công thức frame thế giới sẽ lật đèn
  theo chiều dọc, nét chìm trông thành nổi.
- Đơn vị mm, lưới bước `px_mm`; đạo hàm scipy không chia bước lưới, phải tự chia. Không trộn với `np.gradient`.
- Không xấp xỉ `H ≈ ½∇²h` (mép nét dốc tới ~2,4).
- Không chuẩn hoá percentile các đại lượng đã có thang sẵn (AO, `v_r`, shading): phá mốc "½ = phẳng", "1 = hở".
- Đo độ sâu nét bằng p5–p95, không dùng p1–p99 (khe nứt làm phồng số đo).

## Kiểm chứng sau khi sửa

- **Luôn chạy** `build_notebook.py` (nó kiểm AST và docs) và pyflakes.
- Sửa `mocban_render`: chạy `audit_scans.py` + `run_smoke.py` trên scan mẫu. Kết quả mong đợi: mặt `C-`, tin cậy, tỉ số
  ≈ 8,7×; ảnh render có màu gỗ, không xám.
- Sửa bước tìm mặt: chạy thêm `tests/check_face_detection.py`: lệch ≤ 3°, không lật gương.
- Sửa `mocban_enhance`: các bài kiểm giải tích nằm trong notebook (cell `assert`, 20 bài). Chạy các cell thư viện + cell
  kiểm chứng, hoặc tự dựng mặt có đáp án (mặt sin, mặt phẳng nghiêng, bán cầu) để so.
- Muốn chắc notebook chạy được: exec lần lượt mọi cell code của mục 2 trong cùng một namespace.
- Mọi kết luận hiện dựa trên **1 scan, không có chữ**. Ghi rõ điều này khi báo cáo kết quả, đừng khái quát quá mức.

## Quy ước repo

- **Ít file.** Báo cáo có **một** file `REPORT.md`; kết quả mới thì cập nhật vào đó, không tạo báo cáo mới.
- `figures/`: hình gốc sinh ra khi chạy (là output: chỉ ở máy, **không commit**; repo chỉ giữ `figures/.gitkeep`).
- `assets/`: bản đã nén của các hình dùng trong README và REPORT (có commit, không in tên dữ liệu). Tên `hNN_*.jpg|png`
  với NN trùng số "Hình N" trong REPORT; PNG lớn hơn ~300 KB thì nén sang JPG (quality 88). README và REPORT chỉ nhúng
  ảnh từ `assets/`. Thêm hình giữa chừng thì đánh số lại cho khớp.
- Không commit `.venv/`, `outputs/`, `figures/`, `__pycache__/`, scan (kể cả file zip) hay kết quả chạy.
- Commit và push: **không** thêm dòng `Co-Authored-By` của Claude; tác giả chỉ là người dùng.
- Không gọi các nhánh cũ "A1/A2/A3" (đã bỏ). Chỉ dùng "bước chung", "hướng A", "hướng B".
- Xoá file: không xoá vĩnh viễn; chuyển sang một thư mục chờ xoá ngoài repo để người dùng tự xoá.
