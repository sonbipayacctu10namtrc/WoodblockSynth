"""
Giải thích chi tiết từng hàm cho notebook Kaggle (build_notebook.py ghép mỗi hàm của các file trong src/core/mocban_render/ và
src/core/mocban_enhance/ với đoạn giải thích tương ứng ở đây, đặt ngay TRƯỚC cell code của hàm).

Khoá = tên hàm / lớp ở cấp module. Thêm hàm mới vào module mà quên thêm giải thích ở đây -> build_notebook.py dừng
với lỗi, để notebook không bao giờ có hàm thiếu giải thích.
Chunk không phải hàm (import, hằng số) dùng khoá "import:<tên module>" / "const:<tên biến đầu tiên>".
"""

# =====================================================================================================
# Lời dẫn cho từng mục (theo tiêu đề banner trong module)
# =====================================================================================================
SECTION = {
    "1. Nạp mesh scan + chuẩn hoá khung (OBB)":
        "Nhận file scan bất kỳ, tìm texture rời, đưa scan về một khung toạ độ chuẩn (OBB, quay thật, scale về "
        "`TARGET_MM`). File `s01_scan.py`.",
    "2. Tự xác định mặt khắc + manifest":
        "Tìm mặt khắc → xoay mặt khắc hướng lên camera; đọc `manifest.csv` để chỉ định tay. File `s02_face.py`.\n\n"
        "Giả định: mỗi khối chỉ có **một** mặt mang thông tin. Để dùng được cho mọi bộ scan:\n"
        "- hướng tính theo **hộp bao có hướng (OBB)**, nên scan đặt nghiêng thế nào trong file cũng được;\n"
        "- mọi ngưỡng tính theo **tỉ lệ cạnh dài** của khối, nên không phụ thuộc đơn vị file (mm, m…);\n"
        "- quyết định bằng **so sánh tương đối** giữa các mặt, kèm độ tin cậy; không chắc thì đánh dấu để người soát;\n"
        "- chỉ dùng **phép quay thật** (det = +1), nên mesh không bao giờ bị lật gương (chữ khắc sẽ bị đảo);\n"
        "- luôn có đường chỉ định tay (tham số `face` hoặc file `manifest.csv`).\n\n"
        "**Tên mặt:** 3 trục OBB xếp theo độ dài giảm dần `A ≥ B ≥ C` (C = chiều dày). `C+` / `C-` là hai mặt vuông "
        "góc trục C. Tên cố định với cùng một file scan, nên dùng được trong manifest.",
    "3. Lấy mẫu camera / ánh sáng":
        "Mô tả một lần chụp bằng ba dataclass (camera, đèn, cả lần chụp) và hàm lấy mẫu ngẫu nhiên các tham số đó "
        "theo 4 kiểu chụp thực tế.",
    "4. Toán học camera":
        "Chuyển góc (elev, azim, roll) thành vector hướng và ma trận pose camera kiểu OpenGL.",
    "6. Hậu kỳ giả camera":
        "Biến ảnh render \"sạch\" thành ảnh giống máy ảnh thật: phơi sáng, cân bằng trắng, gamma, tối góc, mờ, nhiễu "
        "cảm biến, nén JPEG.",
    "6b. Che khuất bởi vật ngoài (tầng 2D, sau render)":
        "Vẽ thêm vật che thường gặp khi chụp mộc bản lên ảnh đã render.",
    "5. Renderer pyrender":
        "Bọc pyrender: dựng cảnh (khối + mặt bàn + camera + đèn) và render ra ảnh màu + độ sâu, có bóng đổ.",
    "7. Vòng lặp sinh dữ liệu":
        "Ghép các bước trên thành vòng sinh ảnh hàng loạt, cùng các hàm soát mặt khắc và ghép ảnh xem nhanh.",
    "2. Chuẩn bị trường độ cao":
        "Ảnh độ sâu của scan cần làm trơn nhẹ trước khi lấy đạo hàm; kèm hàm in các con số chi phối tham số.",
    "1. Chiếu 3D sang 2D (trực giao, góc bất kỳ)":
        "**Bước 3D → 2D của phần B.** Chiếu trực giao mesh vào mặt phẳng ảnh ở góc nhìn bất kỳ, mỗi pixel giữ bề mặt "
        "gần camera nhất (Z-buffer) → ảnh độ sâu.",
    "3. Đạo hàm & pháp tuyến":
        "Đạo hàm Gauss của ảnh độ sâu (bậc 1 và 2) — nền tảng của pháp tuyến, độ cong, shading.\n\n"
        "**Quy ước bắt buộc:** đơn vị mm, lưới bước `px_mm`; `x = cột`, `y = hàng` (frame ảnh); pháp tuyến "
        "`n = normalize(-hx, -hy, 1)`. Copy nhầm công thức frame thế giới (Y hướng lên) sẽ lật hướng đèn theo chiều "
        "dọc, và nét khắc chìm sẽ trông thành nổi.",
    "4. Bản đồ độ sâu": "**Phương pháp 1** của phần B.",
    "5. Độ cong (Monge patch chính xác)": "**Phương pháp 3** của phần B.",
    "6. Bất biến tích phân đa tỉ lệ (MSII)": "**Phương pháp 4** của phần B.",
    "7. Ambient Occlusion (horizon mapping)": "**Phương pháp 5** của phần B.",
    "8. Exaggerated Shading & Radiance Scaling": "**Phương pháp 6** của phần B, kèm Lambert làm đối chứng.",
    "9. Chuẩn hoá hiển thị":
        "Đưa kết quả về ảnh hiển thị theo **một chính sách duy nhất**. File `s09_display.py`.",
    "10. Lưới so sánh các phương pháp":
        "Bảng `METHODS` (mỗi phương pháp một hàm `m_*` trả ảnh hiển thị) và lưới so sánh. File `s10_grid.py`.",
}

# =====================================================================================================
# src/core/mocban_render/
# =====================================================================================================
RENDER = {
    "import:mocban_render": """
Thư viện dùng cho phần nạp scan và render. `trimesh` đọc mesh, `cv2` / `numpy` xử lý ảnh, `PIL` đọc texture.

**Shim `np.infty`:** pyrender 0.1.45 (bản cuối trên PyPI, 2020) còn gọi `np.infty`, tên này bị bỏ ở NumPy 2. Gán lại
`np.infty = np.inf` **trước** khi import pyrender, nếu không pyrender lỗi ngay khi import.
""",
    "const:MESH_EXT": """
Hằng số của bước nạp scan:

| Hằng số | Giá trị | Ý nghĩa |
|---|---|---|
| `MESH_EXT` | glb, gltf, obj, stl, ply | Đuôi file được coi là mesh |
| `_IMG_EXT` | jpg, jpeg, png | Đuôi file được coi là ảnh texture |
| `_NOT_ALBEDO` | `_nm`, `normal`, `_rough`… | Tên ảnh chứa các chữ này là **không phải** ảnh màu (normal map, roughness…) nên bị bỏ |
""",
    "const:FACE_THIN_RATIO": """
Ngưỡng của bước tìm mặt khắc:

| Hằng số | Giá trị | Ý nghĩa |
|---|---|---|
| `FACE_THIN_RATIO` | 0,6 | `C/B ≤ 0,6` → khối dẹt, chỉ so 2 mặt lớn `C+`, `C-`; ngược lại so cả 6 mặt |
| `FACE_MIN_RATIO` | 2,0 | Mặt tốt nhất phải chi tiết gấp ≥ 2 lần mặt thứ nhì mới **tin cậy** |
| `FACE_MIN_COVERAGE` | 0,5 | Mặt phải có dữ liệu scan trên ≥ 50 % diện tích mới được xếp hạng |
| `FACE_MISSING` | 0,15 | Mặt đối diện có dữ liệu < 15 % → scan chỉ quét một mặt → tin cậy luôn |

Các ngưỡng đặt thử trên 1 scan thật, cần hiệu chỉnh khi có thêm scan.
""",
    "_has_texture_image": """
**Làm gì:** kiểm tra mesh có ảnh texture màu **thật** hay không.

**Cách làm:** lấy `mesh.visual.material.image` (OBJ) hoặc `baseColorTexture` (GLB); có ảnh và cạnh nhỏ nhất ≥ 16 px
thì trả `True`.

**Vì sao cần ngưỡng 16 px:** với OBJ không kèm file `.mtl`, trimesh tự gắn một ảnh giữ chỗ 2 × 2 px. Không loại ảnh này
thì hàm tưởng mesh đã có texture và bỏ qua bước tự tìm ảnh rời → render ra màu xám.
""",
    "_texture_search_dirs": """
**Làm gì:** liệt kê các thư mục cần tìm ảnh texture cho một file mesh.

**Đầu vào:** `mesh_path`; `up` = số cấp thư mục cha được xét (mặc định 1).

**Đầu ra:** danh sách thư mục, không trùng, theo thứ tự gần → xa:
1. thư mục chứa mesh;
2. với mỗi cấp cha (tới `up` cấp): chính thư mục cha đó và **các thư mục con trực tiếp** của nó.

**Ví dụ:** `up=1` tìm được `<scan>/textures/` khi mesh ở `<scan>/source/x.obj`. `up=2` cần khi Kaggle giải nén zip lồng
thành `<scan>/source/<zip_long>/x.obj` — mesh sâu thêm một cấp so với thư mục `textures/`.
""",
    "find_texture": """
**Làm gì:** tự tìm ảnh texture màu cho mesh có UV nhưng không kèm ảnh.

**Đầu vào:** đường dẫn mesh. **Đầu ra:** đường dẫn ảnh, hoặc `None` nếu không chắc chắn.

**Cách làm:**
1. Ứng viên = mọi file `.jpg/.jpeg/.png` trong các thư mục của `_texture_search_dirs`, bỏ ảnh có tên kiểu normal map /
   roughness / AO… (`_NOT_ALBEDO`).
2. Điểm mỗi ứng viên = **độ dài tiền tố chung** giữa tên ảnh và tên mesh (không phân biệt hoa thường); hoà thì chọn
   file lớn hơn (thường là bản độ phân giải cao).
3. Hai vòng tìm:
   - **gần** (thư mục mesh + anh em): nhận nếu tiền tố chung ≥ 4 ký tự, **hoặc** chỉ có đúng 1 ứng viên;
   - **xa** (lên tới 3 cấp cha): **bắt buộc** tiền tố chung ≥ 4 ký tự, vì càng lên cao càng dễ gặp texture của scan
     khác trong cùng dataset.

**Không đoán bừa:** nhiều ảnh mà không ảnh nào trùng tên → trả `None`; khi đó ghi tên ảnh vào cột `texture` của
manifest.

Ví dụ: mesh `block.obj` và ảnh `block_albedo.jpg` có tiền tố chung dài → chọn đúng, bỏ qua `block_normal.png`.
""",
    "resolve_texture": """
**Làm gì:** quyết định texture cuối cùng cho một mesh từ giá trị người dùng đưa vào.

| `texture` | Kết quả |
|---|---|
| `None` | không gắn texture (phần B dùng, vì chỉ cần hình học) |
| `""` hoặc `"auto"` | gọi `find_texture` |
| đường dẫn tồn tại | dùng luôn |
| tên file / đường dẫn không tồn tại | tìm file **cùng tên** trong các thư mục quanh mesh (tới 3 cấp cha) |

Tìm theo tên để manifest viết ở máy này (đường dẫn Windows) vẫn dùng được trên Kaggle. Không thấy → báo lỗi
`FileNotFoundError` thay vì lặng lẽ render xám.
""",
    "_proper_rotation": """
**Làm gì:** biến ma trận 3 × 3 thành phép **quay thật** (định thức = +1).

**Cách làm:** nếu `det(R) < 0` (ma trận có chứa phép lật gương) thì đổi dấu hàng thứ 3.

**Vì sao:** trục OBB mà trimesh trả về có thể là hệ tay trái. Áp nguyên ma trận đó sẽ **lật gương** mesh, và chữ khắc
trên mộc bản sẽ bị đảo chiều — lỗi rất khó nhận ra bằng mắt.
""",
    "prepare_scan": """
**Làm gì:** nạp scan và đưa về **khung OBB chuẩn** — khung chung để chấm điểm các mặt, không phụ thuộc scan được đặt thế
nào trong file.

**Đầu vào:** `src` (đường dẫn hoặc `trimesh.Trimesh`), `target_extent_mm` (cạnh dài nhất sau khi scale, mặc định 200),
`texture_path` (như `resolve_texture`).

**Đầu ra:** mesh mới: tâm OBB ở gốc toạ độ, 3 cạnh OBB song song 3 trục, cạnh dài nhất = `target_extent_mm`.

**Các bước:**
1. `trimesh.load(..., process=False)`: **không** dùng `force="mesh"` vì tuỳ chọn đó làm mất texture/UV. Scene nhiều node
   (GLB) được gộp bằng `to_mesh()`, có áp phép biến đổi của từng node.
2. Mesh có UV mà chưa có ảnh → tìm và gắn ảnh rời bằng `TextureVisuals(uv, image)`.
3. **Sửa mesh lộn trong ra ngoài:** tính `s = Σ (area · (n·r)/|r|) / tổng diện tích`, với `r` là vector từ tâm tới tâm
   tam giác, `n` là pháp tuyến. Mesh đúng thì pháp tuyến hướng ra ngoài (`s > 0`); `s < −0,2` → lật mọi mặt. Cần vì
   bước chấm điểm dựa vào hướng pháp tuyến.
4. Lấy OBB, ép quay thật (`_proper_rotation`), dời tâm OBB về gốc: `M = [R | −R·t]`.
5. Scale đều để cạnh dài nhất = `target_extent_mm`.

**Metadata ghi lại:** `to_obb_frame` (ma trận 4 × 4 từ toạ độ file sang khung OBB, để truy ngược), `source_texture`
(ảnh rời đã gắn hoặc `None`), `has_texture` (có ảnh màu thật không; `False` → render phần A sẽ xám).
""",
    "_axis_names": """
**Làm gì:** đặt tên `A`, `B`, `C` cho 3 trục theo độ dài cạnh giảm dần.

**Đầu vào:** `ext` = 3 kích thước hộp bao. **Đầu ra:** dict `{chỉ số trục: tên}`, ví dụ `{0: 'B', 1: 'A', 2: 'C'}`.

Sắp xếp ổn định (`kind="stable"`): hai cạnh dài bằng nhau thì trục có chỉ số nhỏ hơn đứng trước, nên tên không nhảy
giữa các lần chạy.
""",
    "_face_axis": """
**Làm gì:** đổi tên mặt dạng `"C+"` thành `(chỉ số trục, dấu ±1)`.

Kiểm tra định dạng: 2 ký tự, ký tự đầu thuộc `ABC`, ký tự sau là `+` hoặc `-`; sai → `ValueError` với thông báo rõ
(bắt lỗi gõ nhầm trong manifest).
""",
    "_face_height": """
**Làm gì:** ảnh độ cao nhìn thẳng vào **một mặt** của khối (trong khung OBB), dùng để chấm điểm mặt đó.

**Đầu vào:** mesh khung OBB, `ax` (trục), `sgn` (phía + hay −), `px` (bước lưới mm).
**Đầu ra:** mảng 2D độ cao (mm) theo hướng nhìn; ô không có dữ liệu = `−inf`.

**Cách làm:**
1. Chỉ giữ tam giác **quay mặt về phía người nhìn**: `n · d > 0,05` với `d` là hướng ra ngoài của mặt đang xét. Scan
   chỉ quét một mặt thì nhìn từ phía sau sẽ gần như trống — chính dấu hiệu `FACE_MISSING` dùng.
2. Rải (splat) các đỉnh và tâm tam giác đó lên lưới, mỗi ô giữ điểm cao nhất (`np.maximum.at`).

**Lưu ý:** splat điểm có thể để lọt lớp bề mặt bên dưới ở vài ô (xem `project_depth`). Chấp nhận được **ở đây** vì kết
quả chỉ dùng để lấy **trung vị** làm điểm số; không dùng hàm này để tạo ảnh.
""",
    "_face_scores": """
**Làm gì:** chấm điểm một mặt: mức độ chi tiết (có hoa văn / chữ khắc không) và độ phủ dữ liệu.

**Đầu vào:** ảnh độ cao `h` (từ `_face_height`), `px`, `long_mm` (cạnh dài nhất). **Đầu ra:** `{detail_mm, coverage}`.

**Độ phủ (`coverage`):** tỉ lệ ô có dữ liệu sau **phép đóng** hình thái (giãn 3 lần rồi co 3 lần, nhân 3 × 3). Phép
đóng lấp lỗ nhỏ giữa các điểm splat nhưng **không** làm mảnh vụn rời rạc phình to — dùng phép giãn đơn thuần thì
một mặt chỉ có vài mảnh vụn cũng được tính là phủ kín.

**Độ chi tiết (`detail_mm`):**
1. Lấp độ cao cho các ô vừa được phép đóng lấp (lan giá trị lân cận 3 lần).
2. **Lớp mặt ngoài:** giữ ô có `h > p90(h) − 0,075·L` (bỏ hốc sâu, vát cạnh), rồi co vào `2,5 %·L` (bỏ mép).
3. Phần dư cao tần: `res = h − G_σ * h` với `σ = 1 %·L`.
4. `detail = trung vị |res|` trên lớp mặt ngoài.

**Vì sao trung vị, không phải độ lệch chuẩn:** đã thử trên scan thật — độ lệch chuẩn chọn nhầm **mặt lưng** vì mép hai
hốc tay cầm rất sâu kéo nó lên; độ nhám bề mặt chọn nhầm một mặt cạnh. Trung vị chỉ cao khi chi tiết **phủ kín** mặt,
đúng đặc điểm của mặt khắc: 0,220 mm (mặt khắc) so với 0,025 mm (mặt lưng) trên scan mẫu.
""",
    "detect_main_face": """
**Làm gì:** tự xác định mặt khắc của khối.

**Đầu vào:** mesh khung OBB (kết quả `prepare_scan`).
**Đầu ra:** dict gồm `face` (vd. `"C-"`), `confident`, `reason` (giải thích bằng lời), `ratio`, `flat`, `px_mm`,
`scores` (điểm từng mặt).

**Cách làm:**
1. `flat = C/B ≤ 0,6` → chỉ xét `C+`, `C-`; ngược lại xét cả 6 mặt.
2. Bước lưới `px = max(L/400, 0,7·√(2·S/N))` (S = diện tích bề mặt, N = số tam giác): đủ mịn nhưng không mịn hơn mật độ
   đỉnh của mesh, để tránh lỗ khi splat.
3. Chấm điểm từng mặt ứng viên (`_face_height` + `_face_scores`).
4. Chỉ **xếp hạng** các mặt có độ phủ ≥ 50 %; mặt được chọn = mặt có `detail` lớn nhất trong số đó.
5. Độ tin cậy:
   - mặt đối diện có độ phủ < 15 % → **tin cậy** (scan chỉ quét một mặt);
   - không còn mặt nào khác để so → **cần soát**;
   - `ratio = detail(chọn) / detail(thứ nhì) ≥ 2` → **tin cậy**, ngược lại **cần soát**.

`confident = False` không có nghĩa là sai — hàm vẫn trả mặt đoán tốt nhất, chỉ báo cần người xem ảnh soát.
""",
    "orient_to_face": """
**Làm gì:** xoay mesh (khung OBB) sao cho mặt `face` hướng lên +Z — tư thế chuẩn để render và chiếu.

**Đầu vào:** mesh khung OBB, `face` (vd. `"C-"`), `rot90` (xoay thêm 0–3 × 90° quanh Z).
**Đầu ra:** bản sao mesh: mặt khắc hướng +Z, cạnh dài còn lại dọc trục X, đáy chạm `z = 0`, tâm XY ở gốc.

**Cách làm:** `z` = hướng ngoài của mặt được chọn; `x` = trục dài nhất trong hai trục còn lại; ma trận quay có các hàng
`[x, z × x, z]` — cách dựng này luôn cho det = +1 (không lật gương). Sau đó nhân phép xoay `rot90` quanh Z, rồi dời
để đáy ở `z = 0`.

**Metadata ghi lại:** `block_size_mm` (kích thước khối sau khi xoay), `to_render_frame` (ma trận từ toạ độ file gốc
sang khung render).

**Giới hạn:** hướng 0° hay 180° trong mặt phẳng (khối lộn đầu) **không suy ra được từ hình học** → chỉnh bằng `rot90`.
""",
    "load_external_mesh": """
**Làm gì:** gói trọn 3 bước `prepare_scan` → `detect_main_face` (hoặc mặt chỉ định tay) → `orient_to_face` trong một
lời gọi; ghi kết quả vào `metadata["face_info"]`.

Các công cụ chạy ở máy local (`run_smoke.py`, `check_face_detection.py`) dùng hàm này. Notebook **không** gọi nó ở
luồng chính mà tự làm 3 bước để **nạp mỗi scan đúng một lần** rồi dùng lại cho soát, phần A và phần B.
""",
    "read_manifest": """
**Làm gì:** đọc `manifest.csv` — file chỉ định tay mặt khắc / texture cho từng scan.

**Cột dùng:** `mesh, texture, face, rot90` (cột khác bỏ qua, nên dùng thẳng được `audit.csv` do notebook sinh ra).
**Đầu ra:** dict `{tên file mesh: {"texture", "face", "rot90"}}`.

**Khoá là TÊN file mesh** (không kèm thư mục) để manifest viết ở máy này vẫn khớp trên Kaggle. Ô trống → `"auto"`
(tự xác định). Đọc bằng `utf-8-sig` để chịu được BOM do Excel thêm vào.
""",
    "CameraSpec": """
**Mô tả một camera.**

| Trường | Ý nghĩa |
|---|---|
| `elev_deg` | Góc cao của camera: 90° = nhìn thẳng từ trên xuống |
| `azim_deg` | Vị trí camera quanh khối (hướng phối cảnh) |
| `roll_deg` | Xoay trong mặt phẳng ảnh: 0° = trục +Y của khối hướng lên trên ảnh |
| `dist_mm` | Khoảng cách camera tới điểm nhìn |
| `yfov_deg` | Góc nhìn dọc (FOV) |
| `look_at` | Điểm camera nhìn vào (mm) |
| `width`, `height` | Kích thước ảnh (px) |
""",
    "LightSpec": """
**Mô tả một nguồn sáng.** `kind`: `directional` (đèn xa, song song), `point` (đèn điểm, vd. flash), `spot`;
`elev_deg`, `azim_deg`: hướng tới đèn; `intensity`; `color` (RGB 0–1, từ nhiệt độ màu); `dist_mm` (khoảng cách, chỉ có
ý nghĩa với point/spot).
""",
    "ShotSpec": """
**Mô tả trọn một lần chụp:** camera + danh sách đèn + ánh sáng môi trường (`ambient`, RGB) + dict tham số hậu kỳ
(`post`). Được ghi nguyên vào `meta.json` của mỗi ảnh (qua `asdict`) để tái tạo và truy vết.
""",
    "_kelvin_rgb": """
**Làm gì:** đổi nhiệt độ màu (Kelvin) sang màu RGB của nguồn sáng.

**Cách làm:** xấp xỉ đường cong của Tanner Helland, với `t = K/100`:
- đỏ = 255 khi `t ≤ 66`, ngược lại `329,7·(t−60)^−0,1332`;
- lục = `99,47·ln t − 161,1` khi `t ≤ 66`, ngược lại `288,1·(t−60)^−0,0755`;
- lam = 255 khi `t ≥ 66`, 0 khi `t ≤ 19`, ngược lại `138,5·ln(t−10) − 305`.

Kẹp về [0, 1]. Ví dụ 3000 K ra vàng cam (đèn sợi đốt), 6500 K gần trắng (ánh sáng ban ngày).
""",
    "sample_shot": """
**Làm gì:** lấy mẫu ngẫu nhiên **một lần chụp** (camera + đèn + hậu kỳ) theo một kiểu chụp thực tế.

**Đầu vào:** `rng` (bộ sinh số ngẫu nhiên có seed → tái tạo được), `block_wh_mm` (kích thước khối), kích thước ảnh,
`preset`. **Đầu ra:** `ShotSpec`.

**Chọn preset:** `mixed` bốc ngẫu nhiên topdown 25 %, handheld 25 %, raking 20 %, closeup 30 %.

| Preset | `elev` | FOV | Khối chiếm khung | Đèn |
|---|---|---|---|---|
| topdown (chụp lưu trữ) | 78–90° | 30–45° | 0,75–0,95 | directional cao 50–80°, ambient 0,25–0,5 |
| handheld (chụp tay) | 45–78° | 40–65° | 0,7–1,0 | 50 % flash gần camera, 50 % đèn phòng; 40 % thêm nguồn phụ 6500 K (cửa sổ) |
| raking (đèn xiên) | 55–88° | 32–50° | 0,75–1,0 | directional **thấp 10–30°** → bóng dài, nét nổi rõ |
| closeup (cận cảnh) | 60–90° | 30–45° | **2–4×** (chỉ thấy một phần khối) | directional 25–80°, 50 % có nguồn phụ |

**Chung cho mọi preset:**
- azimuth đều trong 0–360°;
- roll: 80 % gần thẳng `N(0°, 4°)`, 20 % xoay 90/180/270° (ảnh chụp vội);
- khoảng cách `dist = (đường chéo khối / fill) / (2·tan(FOV/2))` để khối chiếm đúng tỉ lệ `fill` của chiều cao khung;
- điểm nhìn lệch nhẹ khỏi tâm (closeup lệch tới ±35 % kích thước khối);
- nhiệt độ màu 3000 / 4000 / 5000 / 5600 / 6500 K;
- hậu kỳ: EV −0,4…0,5; gamma 0,9–1,15; nhiễu 0–0,03; mờ `max(0, N(0,3; 0,4))` px; vignette 0–0,35; JPEG 55–95;
  lệch cân bằng trắng R/B ±5 %; màu mặt bàn chọn trong 5 màu (giấy trắng, gỗ, vải xám, nền tối, be).
""",
    "frontal_shot": """
**Làm gì:** một lần chụp **cố định** nhìn thẳng mặt +Z: camera `elev = 90°`, đèn directional 40° / 135°, ambient 0,3,
không hậu kỳ. Dùng cho **ảnh soát mặt khắc**, để mọi mặt được chụp y như nhau và so sánh được bằng mắt.

Khoảng cách tính để khối chiếm `fill = 0,9` khung, cộng thêm độ dày khối để camera không cắt vào khối.
""",
    "_spherical_dir": """
**Làm gì:** vector đơn vị ứng với góc cao `elev` và phương vị `azim`:
`(cos e·cos a, cos e·sin a, sin e)`. `elev = 90°` → `(0, 0, 1)` (thẳng lên).

Dùng cho vị trí camera và hướng đèn. Phần B có một bản giống hệt để hai phần dùng chung quy ước góc.
""",
    "look_at_pose": """
**Làm gì:** ma trận pose 4 × 4 của camera kiểu OpenGL (camera nhìn theo trục −Z của chính nó), đặt ở `eye`, nhìn vào
`target`, xoay `roll_deg` trong mặt phẳng ảnh.

**Cách làm:**
- `f = normalize(target − eye)` (hướng nhìn);
- hướng "lên" gợi ý = trục +Y của khối xoay `roll` quanh Z: `(−sin r, cos r, 0)`;
- `s = normalize(f × up)` (phải), `u = s × f` (lên thật);
- các cột của ma trận: `s`, `u`, `−f`, `eye`.

**Vì sao gắn "lên" với trục +Y của khối:** dùng trục Z thế giới làm "lên" như thường lệ thì khi camera nhìn gần thẳng
xuống (`elev ≈ 80–90°`) hướng ảnh bị nhảy đột ngột. Gắn với trục dọc của khối thì khối luôn đứng khi `roll = 0`, ở mọi
góc. Nhìn dọc đúng trục +Y (suy biến) → dùng +Z làm dự phòng.
""",
    "camera_pose_from_spec": """
**Làm gì:** từ `CameraSpec` tính vị trí camera `eye = look_at + _spherical_dir(elev, azim) · dist` rồi dựng pose bằng
`look_at_pose`.
""",
    "apply_post": """
**Làm gì:** biến ảnh render sạch thành ảnh giống máy ảnh thật. Các bước theo đúng thứ tự:

| Bước | Công thức |
|---|---|
| Phơi sáng | `img · 2^EV`. **Auto-exposure tối thiểu:** nếu độ sáng trung bình sau phơi sáng < `min_mean` (0,14) thì tăng EV cho đủ, như máy ảnh tự nâng ISO — tránh ảnh gần đen hoàn toàn (hay gặp ở đèn xiên thấp) |
| Cân bằng trắng | nhân từng kênh với `wb_shift` |
| Gamma | `img^γ` |
| Vignette (tối góc) | `img · (1 − v·r²)`, `r` = khoảng cách chuẩn hoá tới tâm ảnh |
| Mờ | Gauss σ = `blur_sigma` (bỏ qua nếu < 0,05) |
| Nhiễu cảm biến | Poisson-Gauss: `σ = s·(0,6 + 1,2·√(1 − L))` — **vùng tối nhiễu mạnh hơn** như ảnh ISO cao; thêm nhiễu màu theo kênh ở vùng `L < 0,3` |
| Nén JPEG | mã hoá rồi giải mã lại với chất lượng `jpeg_q` → có đúng vết khối 8 × 8 của JPEG thật |

EV thực dùng được ghi lại vào `post["exposure_applied"]` khi auto-exposure can thiệp.
""",
    "add_occluders": """
**Làm gì:** vẽ thêm vật che lên ảnh đã render (tầng 2D).

**Đầu vào:** ảnh RGB, `rng`, `prob` (xác suất ảnh có vật che). **Đầu ra:** `(ảnh, mask che 0/1, danh sách loại)`.

Với xác suất `prob`, thêm 1–2 vật (trọng số ngón tay 3 : thước 2 : băng dính 2 : lóa 2 : bóng 2):

| Loại | Cách vẽ | Tính vào mask che? |
|---|---|---|
| Ngón tay | 1–2 ellipse màu da (3 tông) từ mép trái / phải / dưới, kèm bóng mềm | có |
| Thước | dải vàng hoặc trắng dọc / ngang gần mép | có |
| Băng dính | chữ nhật xanh hoặc trắng | có |
| Lóa flash | đốm trắng mềm, cộng sáng | **không** — bề mặt vẫn thấy một phần |
| Bóng người chụp | dải tối mềm hình thang | **không** |

Mask che cho phép tính `occluded_frac` (phần trăm diện tích bị che) ghi vào meta, dùng để lọc nhãn sau này.
""",
    "PyrenderBackend": """
**Làm gì:** bọc pyrender để render một khối nhiều lần với các lần chụp khác nhau.

**`__init__(mesh, width, height)`:**
- chuyển mesh sang pyrender (`smooth=False`: pháp tuyến theo từng tam giác, giữ nét sắc của mép khắc);
- vật liệu gỗ thấm mực: roughness 0,6 (hơi bóng), metallic 0;
- tạo `OffscreenRenderer` (EGL trên Kaggle);
- tạo **mặt bàn** rộng gấp 4 lần khối, dày 1 mm, ngay dưới đáy khối để nhận bóng đổ.

**`render(shot)`:** dựng cảnh mới mỗi lần (ambient; nền và mặt bàn cùng màu `table_rgb`), camera phối cảnh
(`znear` 5 mm, `zfar` 5000 mm), thêm từng đèn theo hướng trong `LightSpec`, render có **bóng đổ** cho đèn directional
và spot. Trả `(ảnh màu, ảnh độ sâu, pose camera)`. Mesh chỉ nạp lên GPU một lần và được dùng lại giữa các cảnh.

**`gl_info()`:** tên GPU / driver OpenGL thật sự đang render. `llvmpipe`, `softpipe`, `SwiftShader` nghĩa là EGL
**không** tới được driver GPU và đang render bằng CPU — chậm hàng chục lần dù máy có GPU. Cell cấu hình in thông tin này.

**`close()`:** giải phóng renderer (bộ nhớ GPU).
""",
    "render_dataset": """
**Làm gì:** vòng sinh ảnh hàng loạt cho một khối.

**Đầu vào:** backend đã tạo, thư mục ra, số ảnh `n`, kích thước khối, kích thước ảnh, `preset`, `seed`, tên file,
`occluder_prob`, `extra_meta` (thêm vào mọi bản ghi, vd. mặt khắc đã chọn).

**Mỗi ảnh:** `sample_shot` → `render` → `apply_post` → `add_occluders` → (nếu có vật che: nén JPEG lại chất lượng 90 để
vật che cũng có vết nén như phần còn lại) → ghi `<tên>_XXXX.jpg` (chất lượng 95).

**Ghi `<tên>_meta.json`:** mỗi ảnh một bản ghi gồm `file`, `shot` (toàn bộ tham số), `cam_pose` (ma trận 4 × 4),
`occluders`, `occluded_frac`, cộng `extra_meta`. Cùng `seed` → cùng bộ ảnh.
""",
    "face_audit_image": """
**Làm gì:** ảnh soát mặt khắc: render nhìn thẳng (`frontal_shot`) **từng mặt ứng viên** rồi ghép thành lưới tối đa
3 cột, mỗi ô ghi `detail` và `coverage` của mặt đó.

**Khung màu ở mặt được chọn:** **xanh** = tự chọn và tin cậy; **vàng** = tự chọn nhưng cần người xem; **tím** = chỉ
định tay.

Cần pyrender (GPU). Mỗi mặt tạo một renderer riêng nên tốn vài giây mỗi mặt.
""",
    "audit_scan": """
**Làm gì:** soát một scan từ đường dẫn: `prepare_scan` + `audit_prepared`. Trả `(info, ảnh soát, texture rời)`.
Dùng bởi `audit_scans.py` ở máy local. Notebook gọi thẳng `audit_prepared` với mesh đã nạp sẵn.
""",
    "audit_prepared": """
**Làm gì:** soát mặt khắc cho mesh **đã nạp** (kết quả `prepare_scan`).

**Cách làm:**
1. `detect_main_face` → lưu kết quả tự động vào `info["auto_face"]`.
2. Có chỉ định tay (dòng manifest có `face`) → ghi đè `face`, đặt `confident = True`, `reason` ghi rõ chỉ định tay
   **trùng** hay **KHÁC** kết quả tự động (khác → đáng xem lại).
3. Ghi `rot90`, `has_texture` vào `info`.
4. `image=True` → vẽ ảnh soát bằng `face_audit_image` (cần GPU); `image=False` → trả `None`.

Tách riêng khỏi `audit_scan` để notebook **nạp mỗi scan một lần** (mất 10–20 s với scan 1,4 triệu tam giác) rồi dùng lại.
""",
    "contact_sheet": """
**Làm gì:** ghép nhiều ảnh thành một ảnh lưới để xem nhanh: thu mỗi ảnh về rộng `thumb_w`, đệm đen cho cùng chiều cao,
xếp `cols` cột. Dùng cho ảnh `<tên>_sheet.jpg` của phần A.
""",
}

# =====================================================================================================
# src/core/mocban_enhance/
# =====================================================================================================
ENHANCE = {
    "import:mocban_enhance": """
Thư viện của phần B. `scipy.ndimage` cung cấp lọc 1 chiều (`correlate1d`), biến đổi khoảng cách (lấp lỗ) và lọc Gauss.
Lặp lại shim `np.infty` cho trường hợp chạy riêng phần B.
""",
    "prepare_height": """
**Làm gì:** chính quy hoá ảnh độ sâu **trước mọi phép đạo hàm bậc hai**.

**Đầu vào:** `h` (mm), `px_mm`, `sigma0_px` (mặc định 1), `detrend_sigma_px` (tuỳ chọn).

**Vì sao cần:** ảnh độ sâu của scan là mặt ghép từ các **tam giác phẳng** (liên tục nhưng pháp tuyến gãy ở cạnh tam
giác) cộng nhiễu đo của máy scan. Đạo hàm bậc hai của nó là dãy xung trên cạnh tam giác → bản đồ độ cong / MSII hiện lưới
tam giác và hạt nhiễu. Đây là tính chất dữ liệu, không phải lỗi. Làm trơn Gauss σ₀ ≈ 1 px xoá phần này mà gần như không
đụng tới nét khắc (rộng nhiều px).

`detrend_sigma_px`: trừ nền thấp tần `h − G * h` khi khối cong / vênh.
""",
    "relief_stats": """
**Làm gì:** vài con số chi phối việc chọn tham số: kích thước ảnh, `px_mm`, biên độ độ cao `p99 − p1` (mm và px),
độ cao nhỏ nhất / lớn nhất. In ở đầu phần phân tích.
""",
    "_spherical_dir": """
**Giống hệt** hàm cùng tên ở phần nạp / render: `(cos e·cos a, cos e·sin a, sin e)`. Định nghĩa lại để phần B cũng
dùng được khi tách riêng; hai bản trùng công thức nên định nghĩa sau ghi đè bản trước không đổi gì.
""",
    "view_basis": """
**Làm gì:** hệ trục camera trực giao `(phải, lên, hướng nhìn)` cho góc `elev`, `azim`, `roll`.

Dùng **đúng quy ước** của `look_at_pose` ở phần A: "lên" = trục +Y của khối xoay `roll` quanh Z, trực giao hoá theo
hướng nhìn. Nhờ vậy ảnh phần B cùng hướng với ảnh phần A, và khối luôn đứng ở mọi góc.

`elev = 90°` cho `phải = +X`, `lên = +Y`, `hướng nhìn = −Z` (nhìn thẳng từ trên xuống).
""",
    "_raster_zbuffer": """
**Làm gì:** Z-buffer bằng cách **rasterize tam giác** — lõi của phép chiếu.

**Đầu vào:** toạ độ pixel liên tục `(u, v)` và độ sâu `dep` của mọi đỉnh, danh sách tam giác `F`, kích thước ảnh.
**Đầu ra:** ảnh độ sâu; ô không có tam giác nào phủ = `−inf`.

**Cách làm:**
- Pixel `(r, c)` có tâm `(c + 0,5; r + 0,5)`. Với mỗi tam giác, thử các tâm pixel nằm trong hộp bao của nó, tính **toạ
  độ barycentric** `(l1, l2, l3)`; tâm nằm trong tam giác khi cả ba ≥ 0.
- Độ sâu tại tâm = nội suy tuyến tính `l1·z1 + l2·z2 + l3·z3`; mỗi pixel giữ giá trị **lớn nhất** = gần camera nhất.
- **Vector hoá:** nhóm tam giác theo cỡ hộp bao (k = 1, 2, 4, 8… tâm pixel mỗi chiều); mỗi nhóm thử k × k tâm cho mọi
  tam giác cùng lúc, chia khối để giới hạn bộ nhớ (`max_cand`).
- Tam giác chiếu thành đoạn thẳng (vách đứng nhìn ngang) không chứa tâm nào → bỏ qua, đúng hình học.

**Vì sao không splat điểm:** splat để lại khe ngẫu nhiên giữa các điểm; ở ô không nhận điểm nào của mặt ngoài, một lớp
bề mặt nằm **dưới** thắng Z-buffer → hố giả sâu 14–35 mm (đo được trên scan mẫu, vì scan có nhiều lớp chồng nhau).
Rasterize tại tâm pixel thì nơi mặt ngoài phủ tâm, lớp dưới không bao giờ thắng. Đo lại: hố giả 871 → 32 pixel, sai số
mặt sin 0,031 → 0,006 mm.
""",
    "project_depth": """
**Làm gì:** **phép chiếu 3D → 2D** của phần B: chiếu trực giao mesh vào mặt phẳng ảnh ở góc nhìn bất kỳ, mỗi pixel giữ
bề mặt gần camera nhất.

**Đầu vào:** mesh (mặt khắc hướng +Z), `elev`, `azim`, `roll`, `px_mm` hoặc `out_px`, `fill_holes`.
**Đầu ra:** dict `depth` (mm, 0 = điểm xa nhất, lớn = gần camera), `valid` (pixel có bề mặt), `px_mm`.

**Cách làm:**
1. Hệ trục `view_basis`; mỗi đỉnh: `X = V·phải`, `Y = V·lên`, `D = V·(−hướng nhìn)`.
2. Bước lưới:
   - `px_mm` cho trước → dùng luôn;
   - `out_px` cho trước → cạnh dài ảnh = `out_px` pixel;
   - không có gì → **độ phân giải gốc** của scan: `px = √(diện tích chiếu của phần quay về camera / số đỉnh của phần
     đó)`. Mịn hơn mức này chỉ là nội suy trên tam giác phẳng, không thêm thông tin.
3. Kích thước ảnh `round(phạm vi / px) + 1` (dùng `round`, không `ceil`, để sai số dấu phẩy động không cộng dư một hàng).
4. `_raster_zbuffer` → ảnh độ sâu.
5. `valid` = ô có tam giác phủ, sau một phép đóng 3 × 3 để lấp lỗ kim. `valid = False` còn lại là nền ngoài khối, lỗ
   thủng lớn của scan, hoặc vùng **bị che khuất thật** khi nhìn xiên.
6. `fill_holes`: lấp ô trống bằng giá trị ô có dữ liệu gần nhất (biến đổi khoảng cách), để các bộ lọc phía sau không
   gặp `−inf`.

**Kiểm chứng (cell kiểm chứng giải tích):** mặt sin sai số trung vị 0,006 mm; mặt phẳng nhìn xiên có độ dốc đúng
`px / tan(elev)` (lệch < 0,06 %).
""",
    "_dkernel": """
**Làm gì:** nhân Gauss đạo hàm 1 chiều bậc 0, 1 hoặc 2, **ép chính xác trên đa thức**.

**Cách làm:** lấy mẫu `g(x) = exp(−x²/2σ²)` chuẩn hoá tổng = 1; bậc 1: `(−x/σ²)·g`; bậc 2: `((x² − σ²)/σ⁴)·g`. Rồi ép
hai điều kiện:
- `Σ k = 0` → triệt tiêu thành phần hằng số;
- `Σ k·iⁿ / n! = 1` → đạo hàm đúng tuyệt đối trên đa thức bậc ≤ n.

**Vì sao không dùng `scipy.gaussian_filter(order=2)`:** scipy chỉ chuẩn hoá nhân bậc 0, nên nhân bậc 2 có tổng khác 0
(~1e-4). Phần dư đó nhân với độ cao tuyệt đối rồi rò vào kết quả — toán tử **không bất biến tịnh tiến**. Đo trên bán
cầu R = 46 mm: sai số H 138 %, K 467 % (float64 cũng sai y hệt). Sau khi ép: sai số 0,03 %.
""",
    "_sepfilt": """
**Làm gì:** lọc tách được: nhân bậc `order_row` theo trục hàng rồi nhân bậc `order_col` theo trục cột. Ví dụ `(0, 1)` =
`∂/∂x` (làm trơn theo hàng, đạo hàm theo cột). Biên xử lý kiểu `nearest`.
""",
    "derivatives": """
**Làm gì:** năm đạo hàm Gauss của ảnh độ sâu: `hx, hy` (vô thứ nguyên) và `hxx, hxy, hyy` (đơn vị 1/mm).

**Quy ước:** `x` = cột, `y` = hàng. Nhân lọc **không** chia bước lưới nên phải chia `px_mm` (bậc 1) hoặc `px_mm²`
(bậc 2) bằng tay. Không trộn với `np.gradient` (hàm đó **có** chia bước lưới) — trộn hai kiểu là nguồn lỗi đơn vị hay
gặp nhất.
""",
    "normals": """
**Làm gì:** pháp tuyến đơn vị trong frame ảnh: `n = normalize(−g·hx, −g·hy, 1)`.

**`gain` (g) < 1 nén độ dốc:** mép nét khắc dốc tới ~2,4 nên pháp tuyến thô cho nền một màu cộng viền 2 px cháy sáng.
`gain = 0,35` giữ được cả nền lẫn mép.
""",
    "normal_map_rgb": """
**Làm gì:** pháp tuyến → ảnh màu theo quy ước chuẩn `RGB = (n + 1)/2 · 255`. Không kéo giãn percentile, để màu có
nghĩa tuyệt đối (xanh tím = phẳng hướng lên).
""",
    "slope_aspect_hsv": """
**Làm gì:** mã hoá độ dốc bằng màu HSV: **màu (hue) = hướng dốc**, **độ sáng = độ lớn dốc** (`tanh(|∇h| / p95)`).

Trên nét khắc thường đọc rõ hơn normal map RGB, vì mắt phân tách nét theo hướng dốc: hai vách đối diện của một nét có
màu đối nhau.
""",
    "depth_map": """
**Làm gì:** ba dạng của cùng ảnh độ sâu:
- `mm`: độ cao thô, đơn vị vật lý;
- `stretch`: kéo giãn theo percentile 1–99 (bỏ viền 8 px), về [0, 1];
- `local = h − G_σ * h` (σ mặc định 12 px): **khử nền thấp tần**.

Bản `local` mới lộ nét khắc khi khối cong / vênh / nứt, nên là bản đưa vào lưới so sánh (**phương pháp 1**).
""",
    "curvature": """
**Làm gì:** độ cong **chính xác** của mặt Monge `z = h(x, y)` (**phương pháp 3**).

Với `p = hx, q = hy, r = hxx, s = hxy, t = hyy`, `W = √(1 + p² + q²)`:
- độ cong Gauss `K = (rt − s²) / W⁴` (1/mm²);
- độ cong trung bình `H = ((1+q²)r − 2pqs + (1+p²)t) / (2W³)` (1/mm);
- độ cong chính `k1, k2 = H ± √(H² − K)`;
- shape index `S = (2/π)·atan(H / √(H² − K))` ∈ [−1, 1] (hình dạng: lõm ↔ yên ngựa ↔ lồi), curvedness
  `C = √(2H² − K)` (độ mạnh).

**Dấu:** pháp tuyến hướng lên → phần **nổi** có `H < 0`.

**Vì sao không dùng `H ≈ ½∇²h`:** xấp xỉ chỉ đúng khi độ dốc ≪ 1; mép nét khắc dốc hơn nhiều. Công thức chính xác chỉ
tốn thêm 3 phép lọc. `max(H² − K, 0)` tránh NaN do float32 làm hiệu này âm nhẹ ở vùng phẳng.
""",
    "normal_curvature_dir": """
**Làm gì:** độ cong pháp tuyến **dọc phương vị của đèn**:
`k_l = (ux²·hxx + 2·ux·uy·hxy + uy²·hyy) / W³`, với `u` = hướng đèn chiếu xuống mặt phẳng.

Là thành phần "versatile" của Radiance Scaling: gờ **hướng về phía đèn** được làm sáng lên.
""",
    "_disk_kernel": """
**Làm gì:** nhân lọc hình đĩa bán kính `r` px, tổng = 1 → phép lọc trả **trung bình trong đĩa**. Dùng cho MSII.
""",
    "msii_volume": """
**Làm gì:** bất biến tích phân đa tỉ lệ MSII (Mara & Krömker) — **phương pháp 4**.

**Định nghĩa:** `v_r(p)` = tỉ lệ thể tích của khối vật liệu nằm trong quả cầu bán kính `r` tâm tại điểm bề mặt `p`.
Mặt phẳng → đúng ½; điểm lồi (nổi) → < ½; điểm lõm → > ½.

**Liên hệ độ cong** (khai triển Pottmann): `v_r − ½ ≈ (3/16)·H·r` — tức MSII là độ cong trung bình **có tham số tỉ
lệ** `r`, nhìn được chi tiết ở nhiều cỡ.

**Tính trên ảnh độ sâu:**
- mặc định (xấp xỉ): `v_r ≈ ½ + 3/(4r)·(trung bình đĩa r của h − h)` → **một phép lọc đĩa** cho mỗi bán kính;
- `exact=True`: cầu phương cực `n_rho × n_phi` mẫu, cắt độ cao theo chiều cao chỏm cầu `√(r² − ρ²)` — chậm hơn nhưng
  đúng ở mọi `r`.

**Giới hạn của xấp xỉ:** chỉ đúng khi `r ≳ 2 × độ sâu nét`; bán kính nhỏ hơn thì **bão hoà** (cell MSII in cảnh báo).
Kết quả luôn kẹp [0, 1] như tích phân thật.

**So với DoG:** cùng họ, nhưng MSII có mốc ½, chuẩn hoá 1/r và bị chặn — không "nổ" ở vách đứng.
""",
    "msii_combine": """
**Làm gì:** gộp MSII nhiều bán kính:
- `combined = Σ (v_j − ½) / r_j` → bản đồ đa tỉ lệ đưa vào lưới so sánh;
- `bands = v_j − v_{j+1}` → tách đặc trưng theo dải tỉ lệ;
- `scale_argmax` = bán kính có `|v − ½|` lớn nhất → "tỉ lệ đặc trưng" của từng điểm.
""",
    "_base_maps": """
**Làm gì:** lưới toạ độ pixel `(hàng, cột)` dạng float32 — đầu vào cho `cv2.remap` khi dịch ảnh.
""",
    "_shift": """
**Làm gì:** lấy mẫu ảnh `h` tại vị trí lệch `(dx, dy)` pixel (có thể lẻ) bằng `cv2.remap` nội suy song tuyến; ngoài
biên lặp giá trị biên. Dùng chung cho AO và MSII chính xác.
""",
    "horizon_tan": """
**Làm gì:** tan của **góc chân trời** theo một phương vị:
`tan θ_h = max_{t ∈ (0, R]} (h(p + t·u) − h(p)) / t`, kẹp ≥ 0 (t tính bằng mm).

Tức là: đứng ở điểm `p`, nhìn theo hướng `u`, bề mặt xung quanh che khuất tới góc nào.

Bước `t` chia theo thang **log** (12 bước từ 1 tới `R` px): tìm chân trời không phụ thuộc tỉ lệ, 12 bước log thay được
24 bước đều.
""",
    "horizon_ao": """
**Làm gì:** Ambient Occlusion — mức độ một điểm "hở" ra bầu trời (**phương pháp 5**). 1 = hở hoàn toàn, 0 = bị che kín.

**Công thức:** tích phân bán cầu có trọng số cosine; với mỗi phương vị phần nhìn thấy là `[0, π/2 − θ_h]` và
`∫cos ψ sin ψ dψ = ½cos²θ_h`, nên
`AO = trung bình theo phương vị [cos²θ_h] = trung bình [1 / (1 + tan²θ_h)]` — **không cần arctan**.

**Kiểm chứng:** mặt phẳng → AO = 1; chân tường đứng → AO ≈ 0,5.

Tham số: `n_az` phương vị (16), `radius_px` bán kính tìm chân trời (24) — bán kính quyết định cỡ chi tiết được làm nổi.
""",
    "light_dir": """
**Làm gì:** vector hướng đèn trong **frame ảnh** (x = cột, y = hàng, z hướng ra khỏi mặt) từ góc cao và phương vị.
Mặc định 35° / 135° (đèn từ góc trên trái — quy ước quen mắt cho ảnh nổi).
""",
    "lambert": """
**Làm gì:** tô bóng Lambert thuần `max(0, n · l)` — **đối chứng**. Không có nó thì không chứng minh được exaggerated
shading / radiance scaling tốt hơn đèn thường.
""",
    "exaggerated_shading": """
**Làm gì:** Exaggerated Shading (theo tinh thần Rusinkiewicz et al. 2006) — **phương pháp 6a**.

**Cách làm:**
1. Làm trơn ảnh độ sâu ở nhiều tỉ lệ `σ_i = σ₀·2^i` (i = 0…4), tính pháp tuyến `n_i` và bóng `s_i = l · n_i`.
2. Chi tiết của tỉ lệ `i` = phần bóng mà tỉ lệ đó thêm vào so với tỉ lệ thô hơn:
   `S_i = clip(½ + c·(s_i − s_{i+1}), 0, 1)` (c = 6).
3. Gộp nhân qua các tỉ lệ: `S = (Π S_i)^(1/L)`.

**Vì sao hiệu `s_i − s_{i+1}`:** bỏ thành phần bóng thô — vốn phụ thuộc hướng đèn và làm mất chi tiết khi đèn thấp — chỉ
giữ phần nổi / chìm ở từng tỉ lệ. Kết quả gần như **không đổi theo góc đèn** (xem so sánh với Lambert ở cell phương
pháp 6).

Bản trước dùng tỉ số thay cho hiệu và dồn mọi giá trị về ~0,5 (ảnh xám phẳng); dạng hiệu cho dải gần kín [0, 1].
""",
    "radiance_scaling": """
**Làm gì:** Radiance Scaling (Vergne et al. 2010) — **phương pháp 6b**: `L' = σ(κ̄) · L_lambert`.

**Hàm scaling** (hàm Möbius) xác định bởi 3 tính chất: `σ(0) = 1`, `σ(+1) = α` (lồi sáng lên α lần),
`σ(−1) = 1/α` (lõm tối đi α lần):
`σ(κ̄) = ((α+1) + (α−1)κ̄) / ((α+1) − (α−1)κ̄)`.

**Chuẩn hoá độ cong bắt buộc:** `κ̄ = (2/π)·atan(κ / κ_ref)` với `κ_ref` = percentile 95 của `|κ|` — độ cong thô không bị
chặn; tự chuẩn theo percentile làm `α` thành núm vặn ổn định giữa các khối khác nhau.

`kappa="directional"` dùng `normal_curvature_dir` (gờ hướng về đèn sáng lên); `"mean"` dùng độ cong trung bình H.
""",
    "_interior": """
**Làm gì:** cắt bỏ viền `margin_px` pixel mỗi cạnh. Viền ảnh độ sâu thường là mép khối dốc đứng, nếu để lại sẽ chi phối
các percentile dùng để chuẩn hoá.
""",
    "_interior_percentile": """
**Làm gì:** hai percentile `(thấp, cao)` tính trên phần **bên trong** (đã bỏ viền) của ảnh.
""",
    "norm01": """
**Làm gì:** đưa một đại lượng về [0, 1] để hiển thị, theo **một chính sách duy nhất** cho cả notebook.

| Tuỳ chọn | Thang |
|---|---|
| mặc định | percentile 2–98 phần bên trong |
| `sym=True` | đối xứng quanh 0: `[−a, a]` với `a` = percentile 98 của `|x|` → 0 luôn nằm giữa (dùng cho đại lượng có dấu) |
| `vrange=(lo, hi)` | khoá thang tay — **bắt buộc khi quét tham số**, nếu không mỗi ảnh tự chuẩn hoá và dãy so sánh vô nghĩa |

**Không** dùng cho đại lượng đã bị chặn sẵn (AO, MSII `v_r`, mọi shading): kéo giãn sẽ phá mốc "½ = phẳng",
"1 = hở", vốn là toàn bộ ý nghĩa của chúng.
""",
    "colorize": """
**Làm gì:** giá trị [0, 1] → ảnh RGB uint8 qua bảng màu matplotlib. `gray` làm trực tiếp (nhanh); đại lượng có dấu dùng
`coolwarm` (xanh = âm, đỏ = dương, trắng = 0).
""",
    "_lab": """
**Làm gì:** vẽ nhãn chữ trắng trên dải đen ở đầu mỗi ô lưới. Nhãn phải là **ASCII** vì font Hershey của `cv2.putText`
không vẽ được dấu tiếng Việt.
""",
    "m_depth": """
**Bọc phương pháp 1** cho lưới so sánh: độ sâu cục bộ `local`, thang đối xứng, màu `coolwarm`. Mọi hàm `m_*` cùng trả dict
`{raw: số liệu, img: ảnh RGB, label: nhãn}` và nhận `**kw` để bỏ qua tham số không dùng.
""",
    "m_normal": """
**Bọc phương pháp 2:** normal map RGB với `gain` mặc định 0,35.
""",
    "m_curvature": """
**Bọc phương pháp 3:** hiển thị **shape index × curvedness** (điều biên), thang cố định [−1, 1].

Vì sao không hiện shape index trần: S là **tỉ số** nên bão hoà về ±1 ngay cả khi độ cong nhỏ xíu — nền khoét chỉ có
nhiễu vết đục cũng loang lổ đỏ / xanh và nuốt mất nét. Nhân với curvedness (mức độ cong) đưa vùng phẳng về trung tính mà
giữ phân loại lồi / lõm ở nơi thật sự cong (Koenderink vốn định nghĩa S đi kèm C).
""",
    "m_msii": """
**Bọc phương pháp 4:** MSII gộp (`combined`) với các bán kính `radii_px`, thang đối xứng, màu `coolwarm`.
""",
    "m_ao": """
**Bọc phương pháp 5:** AO hiển thị thang xám [0, 1] **không kéo giãn**. Nhận `n_az`, `radius_px`, `n_steps`.
""",
    "m_exaggerated": """
**Bọc phương pháp 6a:** exaggerated shading với hướng đèn `ldir` (mặc định `light_dir()`), thang xám.
""",
    "m_radiance": """
**Bọc phương pháp 6b:** radiance scaling với `alpha` (mặc định 3), thang xám.
""",
    "m_lambert": """
**Bọc Lambert** (đối chứng) để đưa vào lưới khi cần so sánh.
""",
    "const:METHODS": """
**Bảng tra phương pháp:** `METHODS` ánh xạ tên ngắn (`depth`, `normal`, `curvature`, `msii`, `ao`, `exaggerated`,
`radiance`, `lambert`) sang hàm `m_*` tương ứng — cell xuất hàng loạt chọn phương pháp theo tên trong `BATCH_METHODS`.
`GRID6` = 6 phương pháp mặc định của lưới so sánh.
""",
    "enhance_grid": """
**Làm gì:** chạy nhiều phương pháp rồi ghép thành **một ảnh lưới** RGB (mặc định 6 phương pháp, 3 cột).

**Tham số:** `names` (danh sách phương pháp), `crop = (y0, y1, x0, x1)` — **cắt trước khi tính** (nén cả mặt khắc vào
một ô lưới làm mất chi tiết nhỏ, bản crop mới là kết quả thật), `thumb_w` (bề rộng mỗi ô), `flip_mirror` (lật ngang:
mộc bản khắc ngược nên lật để chữ đọc xuôi), `**kw` chuyển tiếp cho các hàm `m_*`.

Ghép bằng numpy thay vì matplotlib: nhanh hơn và giữ nguyên độ phân giải để phóng to đọc lại.
""",
}
