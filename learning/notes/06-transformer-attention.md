# Transformer and attention

## English

**Self-attention:** every token builds a **Query** ("what am I looking for"), a **Key** ("what do I offer") and a **Value** ("my content"). The score between two tokens is `Q·K`; softmax turns scores into weights; the output is a weighted sum of Values:

`Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V`

**Why it helps:** every token sees every other token in one step, so distant words are only one hop apart, and all positions are computed in parallel, which suits GPUs. **Cost:** attention grows with the *square* of the length, which is why I truncate reviews to 256 tokens.

**Pretraining:** DistilBERT (6 layers, about 67 M parameters) was already trained on a large text corpus. I only fine-tune it (learning rate 3e-5, 2 epochs).

**Results on IMDB** (25,000 test reviews):

| Model | Accuracy | Parameters | Train time |
|---|---|---|---|
| BiLSTM from scratch | 87.7% | 2.8 M | 1.4 min |
| DistilBERT fine-tuned | **91.6%** | 67.0 M | 9.5 min |

**Reading it honestly**

- About 4 points more accuracy for roughly 24x the parameters and 6.6x the training time.
- The gain mixes two things: the attention architecture and the pretraining. A Transformer trained from scratch on 22,500 reviews might not beat the LSTM. I have not tested that.
- On four hand-written sentences (two with a change of mind, such as "I expected to hate it, but it turned out to be surprisingly good"), DistilBERT got all four right and the BiLSTM missed the two twists. Four sentences is an illustration, not evidence.
- Missing baseline: TF-IDF + linear SVM. It is on the [roadmap](../ROADMAP.md).

## Tiếng Việt

**Self-attention:** mỗi token tạo một **Query** ("tôi đang tìm gì"), một **Key** ("tôi có gì") và một **Value** ("nội dung của tôi"). Điểm giữa hai token là `Q·K`; softmax đổi điểm thành trọng số; đầu ra là tổng có trọng số của các Value:

`Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V`

**Vì sao hiệu quả:** mỗi token nhìn thấy mọi token khác chỉ trong một bước, nên từ ở xa chỉ cách một bước, và mọi vị trí được tính song song, hợp với GPU. **Cái giá:** attention tăng theo *bình phương* độ dài, nên mình cắt review ở 256 token.

**Pretrained:** DistilBERT (6 lớp, khoảng 67 triệu tham số) đã được huấn luyện trước trên kho văn bản lớn. Mình chỉ fine-tune (learning rate 3e-5, 2 epoch).

**Kết quả trên IMDB** (25.000 review test):

| Mô hình | Accuracy | Tham số | Thời gian train |
|---|---|---|---|
| BiLSTM từ đầu | 87,7% | 2,8 M | 1,4 phút |
| DistilBERT fine-tune | **91,6%** | 67,0 M | 9,5 phút |

**Đọc kết quả cho trung thực**

- Hơn khoảng 4 điểm accuracy với số tham số gấp khoảng 24 lần và thời gian train gấp 6,6 lần.
- Mức tăng trộn hai yếu tố: kiến trúc attention và việc pretrained. Một Transformer train từ đầu trên 22.500 review có thể không vượt được LSTM. Mình chưa kiểm tra điều đó.
- Trên bốn câu tự viết (hai câu có chuyển ý, như "I expected to hate it, but it turned out to be surprisingly good"), DistilBERT đúng cả bốn còn BiLSTM sai hai câu chuyển ý. Bốn câu chỉ là minh họa, không phải bằng chứng.
- Còn thiếu baseline: TF-IDF + SVM tuyến tính. Có trong [lộ trình](../ROADMAP.md).
