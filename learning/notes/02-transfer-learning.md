# Transfer learning: ResNet18 and EfficientNet-B0

## English

**Idea:** a network trained on ImageNet already knows edges, textures and shapes. Instead of learning them again, start from its weights, replace the last layer with a new head for my 12 classes, and fine-tune.

**ResNet** adds *skip connections*: a block outputs `x + F(x)`, so gradients have a short path back and deep networks train reliably. **EfficientNet** scales depth, width and resolution together (compound scaling) and uses MBConv blocks with squeeze-and-excitation.

**My results** (same 12-class waste data, same split, 128 px):

| Model | Parameters | Epochs | Test accuracy | Macro-F1 |
|---|---|---|---|---|
| CNN from scratch | 1.18 M | 20 | 83.1% | 0.809 |
| ResNet18 (pretrained) | 11.18 M | 10 | 94.0% | 0.923 |
| EfficientNet-B0 (pretrained) | 4.02 M | 10 | **96.1%** | **0.953** |

**What I take from it**

- Starting from pretrained weights mattered more than any architectural detail: both pretrained models beat the scratch CNN by 11 to 13 points with half the epochs.
- EfficientNet-B0 beat ResNet18 with about 36% of the parameters, but each epoch was a little slower.
- All three confuse the same pairs (`clothes`/`shoes`, `cardboard`/`paper`, `plastic`/`white-glass`), which suggests the limit is partly the data and the 128 px input.
- Caveat: one run per model, one seed.

## Tiếng Việt

**Ý tưởng:** mạng đã học trên ImageNet đã biết cạnh, kết cấu và hình dạng. Thay vì học lại, ta bắt đầu từ trọng số của nó, thay lớp cuối bằng một "đầu" mới cho 12 lớp của mình, rồi fine-tune.

**ResNet** thêm *skip connection*: một khối cho ra `x + F(x)`, nên gradient có đường ngắn quay về và mạng sâu huấn luyện ổn định. **EfficientNet** mở rộng đồng thời độ sâu, độ rộng và độ phân giải (compound scaling) và dùng khối MBConv có squeeze-and-excitation.

**Kết quả của mình** (cùng dữ liệu rác 12 lớp, cùng cách chia, 128 px):

| Mô hình | Tham số | Epoch | Accuracy test | Macro-F1 |
|---|---|---|---|---|
| CNN từ đầu | 1,18 M | 20 | 83,1% | 0,809 |
| ResNet18 (pretrained) | 11,18 M | 10 | 94,0% | 0,923 |
| EfficientNet-B0 (pretrained) | 4,02 M | 10 | **96,1%** | **0,953** |

**Điều mình rút ra**

- Bắt đầu từ trọng số pretrained quan trọng hơn bất kỳ chi tiết kiến trúc nào: cả hai mô hình pretrained hơn CNN từ đầu 11 đến 13 điểm với một nửa số epoch.
- EfficientNet-B0 vượt ResNet18 với khoảng 36% số tham số, nhưng mỗi epoch chậm hơn một chút.
- Cả ba đều nhầm các cặp giống nhau (`clothes`/`shoes`, `cardboard`/`paper`, `plastic`/`white-glass`), cho thấy giới hạn một phần nằm ở dữ liệu và ảnh vào 128 px.
- Lưu ý: mỗi mô hình một lần chạy, một seed.
