# Hướng dẫn train trên Google Colab hoặc Kaggle

File dùng: **`02_resnet_unet_colab_kaggle.ipynb`**. File này tự chứa mọi thứ (ô đầu tiên tự ghi `common.py`), nên chỉ cần tải đúng một file lên, không cần upload dữ liệu (dữ liệu tự tải trong notebook).

> Giao diện Colab và Kaggle thay đổi theo thời gian, tên menu dưới đây có thể lệch đôi chút.

## So sánh nhanh

| | Google Colab | Kaggle |
|---|---|---|
| Tài khoản | Tài khoản Google | Tài khoản Kaggle (đăng nhập bằng Google được) |
| GPU miễn phí | T4, không đảm bảo luôn có | T4 hoặc P100, có hạn mức giờ mỗi tuần |
| Phiên chạy | Ngắt khi không hoạt động, giới hạn thời gian | Có thể chạy nền ("Save & Run All") |
| Internet | Có sẵn | Phải bật thủ công, có thể cần xác minh số điện thoại |
| Lấy kết quả | Tự tải về ở ô cuối | Tab Output sau khi chạy xong |

Nếu muốn đơn giản nhất: dùng **Colab**. Nếu muốn chạy nền để đi làm việc khác: dùng **Kaggle**.

---

## Cách 1: Google Colab

1. Vào https://colab.research.google.com và đăng nhập tài khoản Google.
2. **File > Upload notebook**, chọn `02_resnet_unet_colab_kaggle.ipynb`.
3. Bật GPU: **Runtime > Change runtime type**, chọn **T4 GPU**, bấm Save.
4. **Runtime > Run all**. Lần đầu cần bấm cho phép khi có hộp thoại hỏi.
5. Ô đầu tiên in tên GPU. Nếu thấy `GPU: True` là đúng. Nếu `False`, quay lại bước 3.
6. Chờ chạy xong (khoảng 30 đến 40 phút ở ảnh 256px, thời gian thực tế phụ thuộc loại GPU Colab cấp). Ô cuối tự tải file `outputs_resnet_unet.zip` về máy.

**Lưu ý Colab:**
- Giữ tab trình duyệt mở. Để máy ngủ hoặc đóng tab có thể làm phiên bị ngắt và mất dữ liệu đang chạy.
- Dữ liệu và file trên Colab **bị xóa khi phiên kết thúc**. Phải tải kết quả về ngay.
- Muốn giữ lại lâu hơn, gắn Google Drive (`from google.colab import drive; drive.mount('/content/drive')`) và chép thư mục `outputs` sang đó.

## Cách 2: Kaggle

1. Vào https://www.kaggle.com, đăng ký hoặc đăng nhập.
2. Xác minh số điện thoại tại **Settings > Phone verification** (thường bắt buộc để dùng GPU và Internet).
3. Menu **Code > New Notebook**, rồi **File > Import Notebook**, chọn `02_resnet_unet_colab_kaggle.ipynb`.
4. Bên phải, mở **Settings** (hoặc **Session options**):
   - **Accelerator**: chọn **GPU T4 x2** hoặc **GPU P100** (code chỉ dùng 1 GPU, loại nào cũng được).
   - **Internet**: bật **On** (cần để tải bộ dữ liệu Oxford-IIIT Pet).
5. Bấm **Run All** để chạy tương tác, hoặc **Save Version > Save & Run All (Commit)** để chạy nền và đóng tab được.
6. Khi xong, mở tab **Output** (sau khi commit) để tải `outputs_resnet_unet.zip` và các file khác.

**Lưu ý Kaggle:**
- Mỗi tuần có giới hạn số giờ GPU miễn phí. Xem hạn mức ở trang cá nhân.
- Nếu **Internet đang tắt**, ô tải dữ liệu sẽ lỗi kết nối. Bật Internet rồi chạy lại.
- Ghi chú: đường dẫn làm việc là `/kaggle/working`, notebook tự dùng thư mục hiện tại nên không cần sửa.

---

## Sau khi train xong: đưa kết quả về máy

Giải nén `outputs_resnet_unet.zip` được:
- `resnet_unet.pt`: trọng số mô hình tốt nhất.
- `resnet_unet_results.json`: mIoU, Dice và đường cong huấn luyện.

Chép `resnet_unet_results.json` vào `ImageSegmentation/outputs/` trên máy để so sánh với kết quả U-Net bài 01.

## Gặp lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `GPU: False` | Chưa bật GPU. Làm lại bước chọn Accelerator, rồi chạy lại từ ô đầu |
| `CUDA out of memory` | Đổi `BATCH_SIZE` trong ô dữ liệu từ 16 xuống 8, hoặc `IMG_SIZE` về 128 |
| Lỗi kết nối khi tải dữ liệu (Kaggle) | Bật Internet trong Settings |
| Mất kết nối giữa chừng (Colab) | Chạy lại từ đầu; bớt `EPOCHS` (ví dụ 15) để chạy ngắn hơn |
| `No module named 'common'` | Chưa chạy ô `%%writefile common.py`. Chạy ô đó trước |

## Nên đổi gì nếu muốn chạy nhanh hơn
- `EPOCHS = 15` (mặc định 25).
- `IMG_SIZE = 128` và `BATCH_SIZE = 32`: nhanh hơn khoảng 4 lần, nhưng viền (`border`) ít được cải thiện.
