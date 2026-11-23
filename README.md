# Phát hiện và theo vết phương tiện trong video giao thông

## Chạy trên Windows

Cần **Python 3.11 64-bit** và **kết nối Internet ở lần chạy đầu** để cài thư viện, tải clip và trọng số. Khi cài Python, bật **Add python.exe to PATH**.

1. Trên GitHub, chọn nhánh **main**, rồi chọn **Code → Download ZIP**.
2. Giải nén ZIP vào thư mục có quyền ghi và mở **`run_demo.bat`** ở thư mục gốc vừa giải nén.
3. Để demo chạy hết clip. Cửa sổ giữ khung hình cuối, số đếm và bảng đối chiếu cho đến khi bấm phím hoặc đóng cửa sổ.

Demo dùng YOLO11n đã huấn luyện trên COCO và ByteTrack trên CPU. Video bên trái hiển thị khung bao, loại xe, ID, quỹ đạo và vạch đếm. Bảng bên phải cập nhật số lượt theo loại xe và chiều tại khung hình xác nhận.

Mỗi lượt chạy tạo `tracked.mp4` (17 giây cảnh và 5 giây giữ bảng cuối) và `events.csv` (khung hình từ 0, thời điểm xác nhận, ID, loại xe, chiều) trong một thư mục riêng bên dưới `outputs`.

Nhấn **Q**, **Esc** hoặc đóng cửa sổ khi đang chạy để lưu `partial.mp4` và `partial.csv`. Có thể chạy lệnh dưới đây từ PowerShell trong thư mục demo.

```powershell
.\run_demo.bat --headless
.\run_demo.bat --resume "outputs\THU_MUC_LUOT_CHAY"
python -m unittest -v test_counting
```

`--headless` xuất kết quả trực tiếp. `--resume` dùng lại phát hiện đã lưu trong `checkpoint.sqlite` của lượt chạy cùng phiên bản, khôi phục ID và bộ đếm rồi dựng lại kết quả. Hai tùy chọn có thể dùng cùng nhau.

## Clip và vạch đếm

Clip gốc `VNTraffic/VNTraffic_Original-video.mp4` dài 17 giây (1920 × 1080, 30 FPS), quay từ camera cố định tại nút Quang Trung – Lê Trọng Tấn. Vạch đứng ở bên phải trụ cầu, từ **(1160, 630)** đến **(1160, 1044)** trên video gốc, tính từ góc trên trái. Vạch cắt hai luồng xe chạy ngược chiều trên màn hình. `L-R` là trái sang phải, `R-L` là phải sang trái.

Điểm đại diện cho xe là trung điểm cạnh dưới của khung bao. Một lượt được xác nhận khi điểm cắt trong đoạn vạch rồi ra khỏi dải đệm ở phía đối diện. Xe dừng trên vạch rồi đi tiếp được tính một lượt. Xe chạm vạch rồi trở về phía cũ được tính 0 lượt. Che khuất hoặc đổi ID gần vạch có thể làm sai số đếm.

## Đối chiếu Ground Truth

Bảng cuối lấy số liệu từ `doi-chieu-da-xac-nhan.csv` đi kèm mã nguồn. **Tham chiếu** đếm các lượt Ground Truth đã xác nhận là ô tô con hoặc xe máy. **Demo đã duyệt** đếm các lượt YOLO11n + ByteTrack theo loại dự đoán trong CSV. Bảng dùng chung phạm vi khung 0–500. Bộ đếm trực tiếp và `events.csv` dùng đủ 510 khung, gồm cả lượt ở khung 507.

Trong phạm vi đối chiếu có **18 lượt tham chiếu**, **20 lượt demo** và **16 lượt khớp đúng xe, loại và chiều**. Với sự kiện đếm đúng loại và chiều, **Precision = 80,0%** (16/20) và **Recall = 88,9%** (16/18).

## Kiểm thử

Đã giải nén ZIP vào thư mục mới và chạy `run_demo.bat` trên Windows 11 với Python 3.11.9 64-bit, dùng môi trường và tài nguyên đã cài. Demo xử lý đủ 510 khung hình trong 67,8 giây trên CPU Ryzen 7 7735HS. Đã kiểm tra bảng cuối trong cửa sổ và video xuất dài 22 giây (1368 × 540, 30 FPS). CSV xuất có 21 lượt, gồm 20 lượt nằm trong phạm vi đối chiếu.

| Loại xe | Trái → phải | Phải → trái |
| --- | ---: | ---: |
| Ô tô con | 2 | 7 |
| Xe máy | 11 | 1 |
| Tổng | 13 | 8 |

Cả 11 kiểm thử tự động đều đạt, gồm quy tắc qua vạch, tổng từ CSV đã xác nhận và ranh giới khung 500.

## Nguồn và mã nguồn

*Vehicle Tracking Dataset*, Le Thi-Nhung và cộng sự, DOI 10.5281/zenodo.18195750. Bản ghi Zenodo ghi CC BY 4.0 cho bộ dữ liệu và ghi nguồn cảnh quay là kênh chính thức của HTV – Đài Phát thanh và Truyền hình Hà Nội. Video kết quả thu nhỏ khung hình, thêm lớp theo vết và đặt bảng đếm bên cạnh cảnh quay. [Bản ghi Zenodo](https://zenodo.org/records/18195750) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Trọng số YOLO11n và cấu hình ByteTrack lấy từ Ultralytics. Trình tải xác minh SHA-256 của clip và trọng số. ZIP mã nguồn dùng để đưa lên GitHub. `.gitignore` loại môi trường Python, dữ liệu tải về và các kết quả chạy. Khi chọn giấy phép cho kho mã, tham chiếu điều khoản AGPL-3.0 của Ultralytics. [YOLO11](https://docs.ultralytics.com/models/yolo11/) · [Điều khoản Ultralytics](https://www.ultralytics.com/license)
