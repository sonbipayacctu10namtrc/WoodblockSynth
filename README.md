# WoodblockSynth

Sinh ảnh 2D từ scan 3D mộc bản để làm giàu dữ liệu huấn luyện OCR Hán-Nôm.

*Synthesize 2D training images for woodblock OCR from 3D scans: photo-realistic renders (camera, lighting, occluders)
and light-independent geometric enhancements (depth, AO, curvature).*

| Hướng A: render giả ảnh chụp | Hướng B: tăng cường hình học |
|---|---|
| ![Render hướng A](assets/h02_render_A.jpg) | ![Tăng cường hướng B](assets/h06_tang_cuong_toan_tam.jpg) |

## Làm được gì

Từ **một scan 3D** của khối mộc bản (GLB/GLTF/OBJ/STL/PLY):

1. **Bước chung:** đưa scan về khung chuẩn, **tự tìm mặt khắc**, xoay mặt khắc về phía camera, không bao giờ lật gương.
   Khi máy không chắc chắn, nó xuất ảnh soát để người kiểm tra, và cho phép sửa tay bằng `manifest.csv`.
2. **Hướng A (cần GPU):** render ảnh giống ảnh chụp: texture thật, 4 kiểu chụp (topdown, handheld, raking, closeup),
   đèn và bóng đổ, hiệu ứng máy ảnh (phơi sáng, nhiễu, mờ, JPEG), tuỳ chọn vật che (ngón tay, thước, lóa…).
   Mỗi ảnh kèm bản ghi `meta.json` đủ để tái tạo lại.
3. **Hướng B (chỉ cần CPU):** chiếu scan thành ảnh độ sâu ở góc bất kỳ rồi làm nổi nét khắc bằng 6 phương pháp
   (độ sâu cục bộ, pháp tuyến, độ cong, MSII, ambient occlusion, exaggerated shading). Hướng này **không dùng đèn**, nên
   nét vẫn rõ ở những góc mà ảnh chụp thường bị tối.

> **Trạng thái:** mới kiểm chứng trên **1 scan thật** (khối in hoa văn, **không có chữ**). Các ngưỡng đều là giá trị đặt
> thử. Chi tiết phương pháp, số liệu và hạn chế xem [REPORT.md](REPORT.md).

## Cấu trúc

```
WoodblockSynth/
├── src/
│   ├── core/
│   │   ├── mocban_render/     bước chung + hướng A   (s01_scan → s02_face → … → s07_dataset)
│   │   └── mocban_enhance/    hướng B                (s01_projection → … → s10_grid)
│   └── run/
│       ├── run_smoke.py       render thử hướng A từ dòng lệnh
│       └── audit_scans.py     soát mặt khắc cả thư mục scan → ảnh soát + audit.csv
├── tests/
│   └── check_face_detection.py   kiểm tra bước tìm mặt khắc ổn định khi xoay scan
├── notebooks/
│   ├── mocban_scan3d_kaggle.ipynb   notebook Kaggle (được SINH RA, không sửa tay)
│   ├── build_notebook.py            sinh notebook từ src/core
│   └── notebook_docs.py             giải thích từng hàm, chèn vào notebook
├── input/                     nơi đặt scan tải về (không đưa lên repo)
├── figures/                   hình gốc sinh ra khi chạy (output, chỉ ở máy; trên repo để trống)
├── assets/                    ảnh đã nén cho README và REPORT
├── REPORT.md                  báo cáo phương pháp và kết quả
├── CONTEXT.md                 bối cảnh, lịch sử, quyết định thiết kế
├── CLAUDE.md                  hướng dẫn cho Claude Code khi làm việc với repo
└── LICENSE                    MIT
```

Tên file `sNN_` đi theo **thứ tự pipeline**, và số mục trong notebook trùng với số file.

## Cài đặt

Yêu cầu Python **3.10–3.12**. Hướng A cần máy có OpenGL (GPU); hướng B chỉ cần CPU.

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install --no-deps pyrender
```

**Linux / macOS**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps pyrender
```

`pyrender` **phải** cài bằng `--no-deps`. Bản 0.1.45 đòi `PyOpenGL==3.1.0`, và bản đó lỗi `glGenTextures` trên Python mới.
`requirements.txt` đã ghim sẵn `PyOpenGL==3.1.7`, bản chạy được.

Trên Linux không có màn hình, các script tự đặt `PYOPENGL_PLATFORM=egl`.

## Chạy thử với một scan

**1. Chuẩn bị scan.** Repo không kèm dữ liệu. Đặt scan 3D (GLB/GLTF/OBJ/STL/PLY, kèm texture nếu
có) vào một thư mục con của `input/`, ví dụ `input/<tên_scan>/`. Nếu scan tải về dạng zip thì giải nén trước; zip có
zip lồng bên trong thì giải nén tiếp cả zip đó:

```bash
python -m zipfile -e input/<tên_scan>.zip input/<tên_scan>
```

Texture là ảnh rời thì để cùng thư mục với mesh hoặc trong thư mục `textures/` bên cạnh. Pipeline tự tìm texture theo
tên, không cần chỉ định.

**2. Soát mặt khắc**, xem máy chọn đúng mặt chưa:

```bash
python src/run/audit_scans.py --scans input/<tên_scan> --out outputs/audit
```

Mở `outputs/audit/*_faces.jpg`: khung xanh là mặt được chọn và tin cậy, khung vàng là cần xem lại.

**3. Render thử hướng A:**

```bash
python src/run/run_smoke.py --mesh input/<tên_scan> --out outputs/smoke --n 12 --size 1600x1200
```

Tuỳ chọn hay dùng: `--preset topdown|handheld|raking|closeup` (mặc định `mixed`), `--occluders 0.5` (xác suất thêm vật
che), `--manifest manifest.csv` (chỉ định tay mặt khắc).

**4. Kiểm tra độ ổn định** của bước tìm mặt khắc:

```bash
python tests/check_face_detection.py --mesh input/<tên_scan> --k 6
```

Các lệnh trên dùng `python` của `.venv`: kích hoạt venv trước, hoặc gọi thẳng `.venv\Scripts\python` (Windows) hay
`.venv/bin/python` (Linux).

Nạp scan 1,42 triệu tam giác mất khoảng 20 s. Sau đó mỗi ảnh render mất khoảng 0,6–0,8 s trên GTX 1660 Ti.

## Chạy trên Kaggle (đủ cả hướng A và B)

1. Tải scan về máy. Trang chia sẻ model 3D thường bắt đăng nhập, nên Kaggle không tự tải được.
2. Kaggle → **Create → New Dataset** → kéo các file `.zip` của scan vào. Kaggle tự giải nén, kể cả zip lồng. Một dataset
   chứa được nhiều scan, nhưng **tên file mesh không được trùng nhau**.
3. Tạo notebook từ [`notebooks/mocban_scan3d_kaggle.ipynb`](notebooks/mocban_scan3d_kaggle.ipynb)
   (**File → Import Notebook**).
4. **Add Input → Datasets → Your Datasets** → chọn dataset vừa tạo.
5. Settings: **Accelerator = GPU (T4)**, **Internet = On** (để cài gói còn thiếu).
6. **Run All**. Kết quả nằm ở `/kaggle/working/outputs.zip`:
   `soat/` (ảnh soát + `audit.csv`), `A_render/` (ảnh + meta), `B_enhance/` (ảnh tăng cường).

Mọi tham số (số ảnh, độ phân giải, kiểu chụp, bán kính AO…) nằm ở **cell cấu hình**, mục 3 của notebook. Không bật GPU thì
notebook tự bỏ qua hướng A, hướng B vẫn chạy đủ.

**Scan bị chọn sai mặt khắc:** tải `soat/audit.csv`, sửa cột `face` (ví dụ `C+`) và/hoặc `rot90` (0–3) ở dòng sai, lưu
thành `manifest.csv`, thêm vào dataset rồi chạy lại.

## Sửa code

Code gốc nằm trong `src/core/`. Notebook **được sinh ra** từ đó, nên không sửa tay file `.ipynb`. Sửa xong module thì
build lại:

```bash
python notebooks/build_notebook.py
```

Builder tự kiểm tra ba điều và dừng nếu sai:
1. cell trong notebook phải khớp code gốc (so sánh AST);
2. hàm nào cũng phải có giải thích trong `notebooks/notebook_docs.py`;
3. các cell chạy không được ghi đè tên hàm của thư viện.

Thêm hàm mới thì làm đủ ba việc: viết hàm vào đúng file `sNN_`, thêm tên hàm vào danh sách xuất trong `__init__.py`
của package, và viết giải thích trong `notebook_docs.py`.

## Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Ảnh hướng A toàn màu xám | Không tìm thấy texture | Xem dòng soát in trạng thái texture; ghi tên ảnh vào cột `texture` của `manifest.csv` |
| `AttributeError: module 'numpy' has no attribute 'infty'` | pyrender cũ dùng tên đã bị bỏ ở NumPy 2 | Import pyrender qua package này (đã có shim), đừng import pyrender trước |
| `glGenTextures` lỗi khi render | Cài pyrender kèm `PyOpenGL==3.1.0` | `pip install PyOpenGL==3.1.7`, rồi cài lại pyrender với `--no-deps` |
| Render rất chậm (vài giây/ảnh) | OpenGL đang render bằng CPU (`llvmpipe`) | Notebook in tên GPU ở cell cấu hình; kiểm tra driver / EGL |
| Notebook bỏ qua scan, không render | Máy không chắc mặt khắc (`confident=False`) | Xem ảnh soát, chỉ định tay bằng `manifest.csv`, hoặc đặt `RENDER_UNCONFIDENT = True` |

## Tài liệu

- [REPORT.md](REPORT.md): phương pháp, tham số và lý do chọn, kết quả đo, các lỗi đã sửa, hạn chế, việc tiếp theo.
- [CONTEXT.md](CONTEXT.md): vì sao có dự án, lịch sử, các quyết định thiết kế, câu hỏi còn mở.
- [CLAUDE.md](CLAUDE.md): quy ước và các bất biến không được phá khi sửa code.
