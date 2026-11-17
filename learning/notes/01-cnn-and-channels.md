# CNN blocks and channels

## English

**Question I had:** what do the "4 blocks" of my CNN do, and what is a *channel*?

**A block** is `Conv3x3 → BatchNorm → ReLU → Conv3x3 → BatchNorm → ReLU → MaxPool2x2`.

- `Conv 3x3, padding=1` finds patterns and keeps the image size. Two in a row see about as much as one 5x5 filter with fewer parameters.
- `BatchNorm` normalises each channel, which makes training steadier and faster.
- `ReLU` (`max(0, x)`) adds the non-linearity. Without it, stacked convolutions collapse into one linear operation.
- `MaxPool 2x2` halves height and width.

**A channel** is one layer of the stack at each pixel. An RGB image has 3. After a convolution with `o` filters it has `o` channels, each a *feature map* that lights up where one pattern appears.

**Shapes through my network** (128 px input): `128x128x3 → 64x64x32 → 32x32x64 → 16x16x128 → 8x8x256`. Space shrinks, channels grow: the network trades "where exactly" for "what kind of pattern". Global average pooling turns `8x8x256` into a 256-vector and a linear layer gives 12 class scores.

**Result in my project:** this 1.18 M-parameter CNN, trained from scratch for 20 epochs, reached 83.1% test accuracy (macro-F1 0.809) on 12 waste classes. It did not overfit (train 84.5%, val 83.6%) but was still improving, so it was under-trained rather than too big.

Code: [`projects/vision-toolkit/training/classification/01_cnn.ipynb`](../../projects/vision-toolkit/training/classification/01_cnn.ipynb).

## Tiếng Việt

**Câu hỏi của mình:** "4 khối" trong CNN làm gì, và *kênh* (channel) là gì?

**Một khối** là `Conv3x3 → BatchNorm → ReLU → Conv3x3 → BatchNorm → ReLU → MaxPool2x2`.

- `Conv 3x3, padding=1` tìm mẫu và giữ nguyên kích thước ảnh. Hai lớp liên tiếp nhìn được vùng bằng một bộ lọc 5x5 nhưng ít tham số hơn.
- `BatchNorm` chuẩn hóa từng kênh, giúp huấn luyện ổn định và nhanh hơn.
- `ReLU` (`max(0, x)`) thêm tính phi tuyến. Thiếu nó, các lớp tích chập chồng lên nhau chỉ là một phép tuyến tính.
- `MaxPool 2x2` giảm một nửa chiều cao và chiều rộng.

**Kênh** là một lớp trong chồng dữ liệu tại mỗi pixel. Ảnh RGB có 3 kênh. Sau một phép tích chập với `o` bộ lọc, ta có `o` kênh, mỗi kênh là một *bản đồ đặc trưng* sáng lên ở chỗ có một mẫu nhất định.

**Kích thước qua mạng của mình** (ảnh vào 128 px): `128x128x3 → 64x64x32 → 32x32x64 → 16x16x128 → 8x8x256`. Không gian nhỏ dần, kênh dày dần: mạng đổi "chính xác ở đâu" lấy "mẫu thuộc loại nào". Global average pooling biến `8x8x256` thành vector 256 chiều, rồi một lớp tuyến tính cho 12 điểm số lớp.

**Kết quả trong dự án:** CNN 1,18 triệu tham số, train từ đầu 20 epoch, đạt 83,1% accuracy trên tập test (macro-F1 0,809) với 12 lớp rác. Không overfit (train 84,5%, val 83,6%) nhưng loss vẫn đang giảm, nên mô hình còn *thiếu huấn luyện* chứ không phải quá lớn.

Mã: [`projects/vision-toolkit/training/classification/01_cnn.ipynb`](../../projects/vision-toolkit/training/classification/01_cnn.ipynb).
