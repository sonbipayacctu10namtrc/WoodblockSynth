# CONTEXT.md

Bối cảnh của dự án: vì sao có nó, nó đã đi qua những gì, các quyết định quan trọng dựa trên lý do gì, và còn gì bỏ ngỏ.
Phương pháp chi tiết và số liệu nằm ở [REPORT.md](REPORT.md); quy ước code nằm ở [CLAUDE.md](CLAUDE.md).

## 1. Mục tiêu lớn và vị trí của repo

**Mục tiêu lớn:** OCR văn bản Hán-Nôm trên **mộc bản** (khối gỗ khắc chữ ngược để in) và ảnh chụp mộc bản.

**Vấn đề:** dữ liệu huấn luyện ít, trong khi ảnh chụp thật rất đa dạng về góc chụp, ánh sáng, bóng đổ, độ mòn và vật che
(ngón tay, thước, nhãn…). Chụp thêm thì tốn công và vẫn không bao quát hết các trường hợp.

**Repo này** chỉ làm một mắt xích: từ scan 3D của khối, **sinh ảnh 2D có kiểm soát** để làm giàu dữ liệu huấn luyện.
Phần OCR (huấn luyện, đánh giá) **không** nằm ở đây.

## 2. Dữ liệu

**Scan 3D**
- **Không có scan 3D mộc bản Hán-Nôm nào công khai** trên các kho model 3D và kho bảo tàng đã tìm.
- Scan đang dùng: một scan công khai (giấy phép cho phép dùng lại) của khối in **hoa văn** trên vải,
  **không có chữ**, nên chưa sinh được ảnh có nhãn chữ.
- Đã có vài scan khối in **có chữ** được chia sẻ công khai làm ứng viên, chưa tải về.
- Trang chia sẻ model 3D thường bắt đăng nhập mới tải được, nên scan phải tải tay rồi đưa lên Kaggle Dataset.
- Có một model mộc bản Việt Nam được đăng công khai để xem, nhưng tác giả **không cho tải** và không ghi license. Muốn dùng thì phải xin trực tiếp tác giả; không trích xuất từ trình xem.

**Ảnh chụp thật**
- Các thí nghiệm OCR trước đây dùng **hai bộ ảnh chụp mộc bản thật**. Hai bộ này **không** nằm trong repo. Chúng là mốc để so "độ giống thật" của ảnh render (việc chưa làm).
- Nhãn CER của một trong hai bộ được **căn tự động** theo một văn bản kinh công khai, **chưa có người kiểm**. Dùng được để so
  A/B giữa hai cấu hình, nhưng không nên coi là con số độ chính xác tuyệt đối.

## 3. Lịch sử

| Thời điểm | Sự kiện |
|---|---|
| trước 09/2026 | Thí nghiệm OCR bằng PaddleOCR PP-OCRv6 trên ảnh thật. Kết luận về tiền xử lý: dùng **ảnh thô** tốt hơn CLAHE cả về CER lẫn phát hiện chữ; CLAHE là nguyên nhân làm kết quả kém. |
| 16/09/2026 | Bắt đầu sinh ảnh từ 3D, gồm ba nhánh: **A1** dựng mesh chữ Hán tổng hợp từ font (có nhãn từng chữ), **A2** dựng relief từ ảnh thật, **A3** dùng scan 3D thật. |
| 16–22/09 | Trên ảnh render của A1, PP-OCRv6 đọc được khi góc chụp dễ (recall ~0,7), gần như **không đọc được khi góc camera ≤ 50°**, và rất nhạy với **mòn một phần nét**. Đây là lý do hướng A giữ góc camera ≥ 45°. |
| 23/09 | Quyết định **bỏ A1 và A2**, chỉ giữ scan thật, và không cần đánh giá OCR trong nhánh này. Khi chạy trên scan thật, phát hiện: scan đặt nghiêng (AABB chọn sai trục), ma trận hoán trục lật gương, không phân biệt mặt khắc với mặt lưng. Từ đó sinh ra `prepare_scan` / `detect_main_face` / `orient_to_face`. Hướng B chuyển sang scan thật và lộ ra lỗi hố giả của phép rải điểm; thay bằng rasterize tam giác. |
| 24/09 | Gộp hai notebook (render và tăng cường) thành **một** notebook Kaggle. Chạy thật lần đầu trên Kaggle (2 × T4): hướng B đúng, nhưng **ảnh hướng A toàn màu xám** do zip lồng làm lạc đường dẫn texture; đã sửa. Notebook viết lại thành mỗi hàm một cell kèm giải thích. |
| 25/09 | Tách khỏi project OCR cũ (`OCRMocBan`): bỏ toàn bộ code OCR và tiền xử lý; tái cấu trúc thành `src/` · `tests/` · `notebooks/`; tách module thành các file `sNN_` theo thứ tự pipeline; gộp ba báo cáo thành `REPORT.md`; đặt tên **WoodblockSynth**. Chạy lại notebook trên Kaggle: hết ảnh xám, số liệu hướng B trùng máy local. |

## 4. Các quyết định thiết kế và lý do

| Quyết định | Lý do |
|---|---|
| Chỉ dùng scan 3D thật | Quyết định của người dùng ngày 23/09. Texture thật (vân gỗ, vết mực) cho ảnh giống ảnh chụp hơn hẳn texture thủ tục. |
| Giả định mỗi khối có **một** mặt mang thông tin | Đơn giản hoá bước tìm mặt khắc; khối khắc hai mặt để sau. |
| Tự tìm mặt khắc bằng **so sánh tương đối** giữa các mặt, kèm độ tin cậy, luôn có đường chỉ định tay | Phải chạy được trên scan bất kỳ, không biết trước đơn vị hay hướng; khi không chắc thì người soát thay vì render sai mặt. |
| Hai hướng A và B song song | A giống ảnh chụp nhưng phụ thuộc đèn (đèn thấp là mất nét); B không phụ thuộc đèn, đọc nét trực tiếp từ hình học. |
| Góc camera hướng A không dưới 45° | Góc thấp làm mặt khắc co ngắn, bóng che nhiều; thực nghiệm với A1 cho recall ≈ 0 ở ≤ 50°. |
| Chạy chính trên **Kaggle** | Có GPU miễn phí cho hướng A. Máy local (GTX 1660 Ti) dùng để thử nhanh. |
| Notebook **sinh từ code**, mỗi hàm một cell kèm giải thích | Đọc hiểu từng bước ngay trong notebook, và code local với notebook không bao giờ lệch nhau (builder so AST). |
| Tên file `sNN_` theo thứ tự pipeline | Mở thư mục là thấy luồng xử lý; lỗi ở mục nào của notebook thì biết mở file nào. |

## 5. Trạng thái hiện tại

- Pipeline chạy trọn vẹn ở máy local, có GPU hay không có GPU đều được. 20/20 bài kiểm giải tích đạt.
- Chạy lại trên Kaggle (25/09) với bản đã sửa: **hết lỗi ảnh xám**; soát mặt khắc và số liệu hướng B trùng máy local.
- Render trên Kaggle chậm (~5 s/ảnh ở lần chạy đầu, so với ~0,6 s ở local). Nghi EGL đang render bằng CPU (`llvmpipe`),
  **chưa xác nhận**: tên GPU và thời gian mỗi ảnh nằm trong log notebook, không có trong `outputs.zip`.

## 6. Câu hỏi còn mở

1. **Ngưỡng tìm mặt khắc** (0,6 / 2× / 50 % / 15 %) mới thử trên 1 scan. Cần thêm scan để hiệu chỉnh, nhất là khối gần
   lập phương (biên độ chỉ 2,1× so với ngưỡng 2×).
2. **Nhãn chữ cho ảnh render** khi có scan có chữ. Hướng dự kiến: gắn nhãn một lần trên ảnh độ sâu nhìn thẳng, rồi chiếu
   sang mọi khung hình bằng pose camera đã lưu trong `meta.json`.
3. **Độ giống thật**: chưa đo. Cần so thống kê (phân bố độ sáng, độ tương phản nét, phổ nhiễu) với hai bộ ảnh chụp thật trước
   khi sinh số lượng lớn.
4. **Tham số hướng B cho nét chữ Hán** (nhỏ và nông hơn hoa văn): `MSII_RADII`, `AO_R_PX` nhiều khả năng phải giảm, chưa
   kiểm chứng.
5. **Vật che** còn là hình học đơn giản; có nên thay bằng patch cắt từ ảnh thật hay không.
6. **Cách dùng đầu ra hướng B** khi huấn luyện: làm kênh phụ, làm ảnh tiền xử lý, hay làm ảnh huấn luyện riêng.

## 7. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| Mộc bản | Khối gỗ khắc chữ **ngược** để in; ảnh phải lật ngang mới đọc xuôi (`FLIP_MIRROR`). |
| Mặt khắc | Mặt mang chữ / hoa văn của khối; các mặt còn lại là lưng và cạnh. |
| `A`, `B`, `C`; `C+`, `C-` | Ba trục của hộp bao định hướng (OBB) xếp theo độ dài giảm dần; C là chiều dày. `C+` / `C-` là hai mặt vuông góc trục C. |
| Soát | Kiểm tra bằng mắt ảnh nhìn thẳng các mặt ứng viên (`*_faces.jpg`) để xác nhận máy chọn đúng mặt khắc. |
| Manifest | `manifest.csv`, chỉ định tay `face` / `rot90` / `texture` cho từng scan, khoá theo tên file mesh. |
| detail / coverage | Độ chi tiết (trung vị phần dư cao tần) và độ phủ dữ liệu của một mặt; dùng để chọn mặt khắc. |
| Hướng A | Render giả ảnh chụp (pyrender, cần GPU). |
| Hướng B | Ảnh độ sâu + 6 phương pháp tăng cường hình học (CPU, không dùng đèn). |
| Rải điểm / rasterize | Hai cách chiếu mesh thành ảnh độ sâu; rải điểm tạo hố giả trên scan thật nên đã bị thay bằng rasterize tam giác. |
