# U-Net and segmentation metrics

## English

**Segmentation** labels every pixel instead of the whole image.

**U-Net** has a U shape:

- **Encoder**: four `DoubleConv → MaxPool` stages (128 → 8 px; channels 32 → 512). It learns *what* is in the image.
- **Decoder**: four `ConvTranspose (x2) → concatenate skip → DoubleConv` stages back to full size. It learns *where*.
- **Skip connections** copy encoder features of the same resolution into the decoder. The encoder knows exact positions but little meaning; the decoder knows meaning but lost detail. Together they give sharp boundaries.
- A final `1x1` convolution turns 32 channels into 3 class scores per pixel.

**Loss:** cross-entropy plus Dice loss. Dice rewards overlap, so a small class like `border` is not ignored.

**Metrics** (from the confusion matrix over all pixels): **IoU** = `TP / (TP + FP + FN)`, **Dice** = `2TP / (2TP + FP + FN)`, **mIoU** = mean IoU over classes. Pixel accuracy is misleading because the background covers most pixels.

**My results** (Oxford-IIIT Pet: pet / background / border, official test set):

| Model | Input | mIoU | Border IoU |
|---|---|---|---|
| U-Net from scratch | 128 px | 0.767 | 0.548 |
| U-Net with ResNet18 encoder | 256 px | 0.809 | 0.617 |

**Lessons**

- The thin `border` class was the real bottleneck; mIoU hid it less than pixel accuracy would.
- The second model changed several things at once (pretrained encoder, 256 px, border weight 3, fewer epochs and a different learning rate). I cannot credit the +0.042 to any one of them. A proper ablation is on the [roadmap](../ROADMAP.md).
- The score depends on *where* you measure: at original resolution on 400 random test images the two models score 0.733 and 0.772.

## Tiếng Việt

**Phân đoạn** gán nhãn cho từng pixel thay vì cả ảnh.

**U-Net** có hình chữ U:

- **Encoder**: bốn tầng `DoubleConv → MaxPool` (128 → 8 px; kênh 32 → 512). Học *có gì* trong ảnh.
- **Decoder**: bốn tầng `ConvTranspose (x2) → nối skip → DoubleConv` quay về kích thước đầy đủ. Học *ở đâu*.
- **Skip connection** sao chép đặc trưng encoder cùng độ phân giải sang decoder. Encoder biết vị trí chính xác nhưng ít ngữ nghĩa; decoder biết ngữ nghĩa nhưng mất chi tiết. Ghép lại cho ranh giới sắc nét.
- Conv `1x1` cuối đổi 32 kênh thành 3 điểm số lớp tại mỗi pixel.

**Loss:** cross-entropy cộng Dice loss. Dice thưởng cho phần chồng lấp nên lớp nhỏ như `border` không bị bỏ qua.

**Chỉ số** (từ ma trận nhầm lẫn trên mọi pixel): **IoU** = `TP / (TP + FP + FN)`, **Dice** = `2TP / (2TP + FP + FN)`, **mIoU** = trung bình IoU các lớp. Pixel accuracy dễ gây hiểu lầm vì nền chiếm phần lớn pixel.

**Kết quả của mình** (Oxford-IIIT Pet: con vật / nền / viền, tập test chính thức):

| Mô hình | Ảnh vào | mIoU | IoU viền |
|---|---|---|---|
| U-Net từ đầu | 128 px | 0,767 | 0,548 |
| U-Net encoder ResNet18 | 256 px | 0,809 | 0,617 |

**Bài học**

- Lớp `border` mỏng mới là điểm nghẽn thật; mIoU cho thấy điều này rõ hơn pixel accuracy.
- Mô hình thứ hai đổi nhiều thứ cùng lúc (encoder pretrained, 256 px, trọng số viền 3, ít epoch hơn và learning rate khác). Mình không thể quy mức +0,042 cho riêng thứ nào. Ablation đúng cách nằm trong [lộ trình](../ROADMAP.md).
- Điểm số phụ thuộc *đo ở đâu*: ở độ phân giải gốc trên 400 ảnh test ngẫu nhiên, hai mô hình đạt 0,733 và 0,772.
