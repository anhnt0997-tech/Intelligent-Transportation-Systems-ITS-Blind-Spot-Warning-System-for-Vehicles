# Phát hiện và theo vết phương tiện trong video giao thông

## Chạy trên Windows

Cần **Python 3.11 64-bit** và **kết nối Internet ở lần chạy đầu** để cài thư viện, tải clip và trọng số. Khi cài Python, bật **Add python.exe to PATH**.

1. Trên GitHub, chọn nhánh **main** rồi chọn **Code → Download ZIP**.
2. Giải nén ZIP vào thư mục có quyền ghi và mở **`run_demo.bat`** ở thư mục gốc. Để demo chạy hết clip. Cửa sổ giữ khung hình cuối, biểu đồ và bảng đối chiếu Ground Truth cho đến khi bấm phím hoặc đóng cửa sổ.

Demo dùng YOLO11n đã huấn luyện trên COCO và ByteTrack trên CPU để đếm ô tô con, xe máy và xe tải. Video bên trái hiển thị khung bao, loại xe, ID, quỹ đạo và vạch đếm. Biểu đồ ba nhóm bên phải bắt đầu từ 0, giữ trục dọc từ 0 đến 16 và cập nhật tại khung hình xác nhận lượt qua vạch của lần chạy hiện tại.

Mỗi lượt chạy tạo `tracked.mp4` (17 giây cảnh và 5 giây giữ bảng cuối) và `events.csv` (khung hình từ 0, thời điểm xác nhận, ID, loại xe, chiều) trong một thư mục riêng bên dưới `outputs`.

Nhấn **Q**, **Esc** hoặc đóng cửa sổ khi đang chạy để lưu `partial.mp4` và `partial.csv`. Có thể chạy lệnh dưới đây từ PowerShell trong thư mục demo.

```powershell
.\run_demo.bat --headless
.\run_demo.bat --resume "outputs\THU_MUC_LUOT_CHAY"
.\.venv\Scripts\python.exe -m unittest -v test_counting test_evaluate
```

`--headless` xuất kết quả trực tiếp. `--resume` dùng lại phát hiện đã lưu trong `checkpoint.sqlite` của lượt chạy cùng phiên bản, khôi phục ID và bộ đếm rồi dựng lại kết quả. Hai tùy chọn có thể dùng cùng nhau.

## Clip và vạch đếm

Clip gốc `VNTraffic/VNTraffic_Original-video.mp4` dài 17 giây (1920 × 1080, 30 FPS), quay từ camera cố định tại nút Quang Trung – Lê Trọng Tấn. Vạch đứng ở bên phải trụ cầu, từ **(1160, 630)** đến **(1160, 1044)** trên video gốc, tính từ góc trên trái. Vạch cắt hai luồng xe chạy ngược chiều trên màn hình. `L-R` là trái sang phải, `R-L` là phải sang trái.

Điểm đại diện cho xe là trung điểm cạnh dưới của khung bao. Một lượt được xác nhận khi điểm cắt trong đoạn vạch rồi ra khỏi dải đệm ở phía đối diện. Xe dừng trên vạch rồi đi tiếp được tính một lượt. Xe chạm vạch rồi trở về phía cũ được tính 0 lượt. Che khuất hoặc đổi ID gần vạch có thể làm sai số đếm.

## Kết quả ba lớp

| Loại xe | Tham chiếu 0–500 | Demo 0–500 | Demo toàn clip |
| --- | ---: | ---: | ---: |
| Ô tô con | 0 / 7 | 2 / 6 | 2 / 7 |
| Xe máy | 9 / 2 | 11 / 1 | 11 / 1 |
| Xe tải | 1 / 2 | 0 / 2 | 0 / 2 |
| **Tổng hai chiều** | **21** | **22** | **23** |

Mỗi ô theo loại xe ghi số lượt **trái → phải / phải → trái**. **21 lượt tham chiếu** thuộc khung 0–500. **23 lượt demo toàn clip** thuộc khung 0–509, gồm một lượt ô tô ở khung 507. Bảng cuối dùng phạm vi 0–500 cho cả hai cột. Biểu đồ và `events.csv` dùng toàn clip.

Trong khung 0–500 có **18 lượt khớp đúng xe, loại và chiều**. **Precision = 81,8%** (18/22) và **Recall = 85,7%** (18/21). Xe tải ở L12 và L22 được đếm đúng lớp và chiều. Chiếc xe tải cabin xanh ở L06 bị ghi hai lượt ô tô con từ demo ID **16** (cabin, khung **58**) và ID **19** (cả xe, khung **63**). Sau vạch, ID 19 có lúc mang nhãn xe tải, và các ID **163, 196, 222** cũng xuất hiện thoáng qua trên chiếc xe. Các ID này không tạo thêm lượt.

Tệp `tham-chieu-ba-lop.csv` ghi từng lượt Ground Truth đã xác nhận, gồm ba xe tải L06, L12 và L22. L05 là lượt demo trùng xe với L06. [CSV đối chiếu của lần kiểm thử](doi-chieu-ba-lop.csv) lưu kết quả ghép từng lượt của bản ba lớp. Lệnh sau tái dựng quỹ đạo từ checkpoint của lần chạy, xác minh sự kiện khớp `events.csv` rồi tạo `doi-chieu-ba-lop.csv` trong thư mục kết quả.

```powershell
.\.venv\Scripts\python.exe evaluate.py "outputs\THU_MUC_LUOT_CHAY"
```

Ghép một-một theo cùng chiều, sai lệch tối đa 0,40 giây và 24 px dọc vạch trên ảnh 960 × 540. Lớp xe được chấm sau khi ghép. CSV giữ cả sự kiện chưa ghép và các ứng viên cạnh tranh.

### Mốc so sánh hai lớp

Bản hai lớp ghi **21 lượt toàn clip**, gồm ô tô con **2 / 7**, xe máy **11 / 1**. Trong khung 0–500, bản này ghi **20 lượt demo**, có **18 lượt tham chiếu** trong hai lớp và **16 lượt đúng xe, loại và chiều**. Precision là **80,0%**, Recall là **88,9%**. Tệp `doi-chieu-da-xac-nhan.csv` lưu đối chiếu của bản hai lớp.

## Kiểm thử

Chạy mới từ khung 0 trên Windows 11, Python 3.11.9 và CPU Ryzen 7 7735HS. YOLO11n + ByteTrack xử lý 510 khung hình trong 53,4 giây. Đã đối chiếu cả sáu cột trong `tracked.mp4` với `events.csv` theo từng khung hình. Cả 15 kiểm thử tự động đều đạt, gồm quy tắc qua vạch, dữ liệu tham chiếu của hai phiên bản và ghép các lượt trùng hoặc sai lớp.

## Nguồn và mã nguồn

*Vehicle Tracking Dataset*, Le Thi-Nhung và cộng sự, DOI 10.5281/zenodo.18195750. Bản ghi Zenodo ghi CC BY 4.0 cho bộ dữ liệu và ghi nguồn cảnh quay là kênh chính thức của HTV – Đài Phát thanh và Truyền hình Hà Nội. Video kết quả thu nhỏ khung hình, thêm lớp theo vết và đặt biểu đồ bên cạnh cảnh quay. [Bản ghi Zenodo](https://zenodo.org/records/18195750) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Trọng số YOLO11n và cấu hình ByteTrack lấy từ Ultralytics. Trình tải xác minh SHA-256 của clip và trọng số. ZIP mã nguồn dùng để đưa lên GitHub. `.gitignore` loại môi trường Python, dữ liệu tải về và các kết quả chạy. Khi chọn giấy phép cho kho mã, tham chiếu điều khoản AGPL-3.0 của Ultralytics. [YOLO11](https://docs.ultralytics.com/models/yolo11/) · [Điều khoản Ultralytics](https://www.ultralytics.com/license)
