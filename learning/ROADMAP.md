# Roadmap

A 24-week plan for learning machine learning through Jupyter notebooks, from NumPy to a capstone project. The plan is a guide, not a promise.

**How to read the status column**

- **Applied here**: the idea is used in a project in this repo, with an experiment I can point to. It does **not** mean I have mastered it.
- **Partly applied**: some of the planned work is done; the gap is stated.
- **Not tracked yet**: I may have studied it elsewhere, but there is no evidence in this repo yet, so I am not claiming it.

| Week | Topic | Status | Where |
|---|---|---|---|
| 1 | NumPy, pandas, matplotlib, linear algebra basics | Not tracked yet | |
| 2 | Probability, gradients, the ML workflow, data leakage | Partly applied | [chronological split](notes/07-honest-experiments.md) |
| 3 | K-Means | Not tracked yet | |
| 4 | Hierarchical clustering, DBSCAN | Not tracked yet | |
| 5 | PCA and SVD | Not tracked yet | |
| 6 | Gaussian mixtures, ICA | Not tracked yet | |
| 7 | Linear regression, regularisation | Not tracked yet | |
| 8 | Logistic regression, Naive Bayes, precision / recall / F1 | Partly applied | [metrics note](notes/03-classification-metrics.md) |
| 9 | LDA, SVM | Not tracked yet | |
| 10 | Trees, ensembles, cross-validation | Not tracked yet | |
| 11 | Gradient descent and optimisation | Not tracked yet | |
| 12 | Momentum, Adam, schedulers | Partly applied | AdamW + OneCycleLR used in every project |
| 13 | Perceptron, MLP, backpropagation | Not tracked yet | |
| 14 | PyTorch basics: tensors, autograd, training loops | Applied here | training loops in [`common.py`](../projects/vision-toolkit/training/classification/common.py) |
| 15 | Training diagnostics: overfitting, normalisation, dropout | Partly applied | BatchNorm and dropout in the [CNN](notes/01-cnn-and-channels.md) |
| 16 | CNNs: convolution, pooling, receptive field | Applied here | [CNN note](notes/01-cnn-and-channels.md) |
| 17 | Transfer learning | Applied here | [ResNet / EfficientNet note](notes/02-transfer-learning.md) |
| 18 | A computer-vision project with error analysis; segmentation concepts | Applied here | [Vision Toolkit](../projects/vision-toolkit), [U-Net note](notes/04-unet-segmentation.md) |
| 19 | Classic text classification (TF-IDF, Naive Bayes, linear SVM) | Partly applied | [TF-IDF + Logistic Regression / SVM](../projects/model-study/results/REPORT.md); Naive Bayes pending |
| 20 | Topic modelling (LDA, LSA) | Not tracked yet | |
| 21 | Embeddings, RNN / LSTM intuition, attention | Partly applied | [RNN and LSTM note](notes/05-rnn-lstm.md), [Transformer note](notes/06-transformer-attention.md) |
| 22 | Transformers and fine-tuning | Partly applied | DistilBERT fine-tune in [Sequence Modeling](../projects/sequence-modeling); the [TF-IDF + SVM comparison is complete](../projects/model-study/results/REPORT.md) |
| 23 | Choose a capstone | Not tracked yet | |
| 24 | Ablations, error analysis, a 10-minute talk | Not tracked yet | |

## Next

Things I want to do next, in no firm order:

- [x] The TF-IDF + linear SVM baseline for IMDB, so the DistilBERT result has an honest comparison.
- [ ] Re-run the key experiments with several seeds and report the spread.
- [ ] A clean ablation for the segmentation gain (pretrained encoder, image size, border weight, one at a time).
- [ ] An experiment where long-range memory really matters, to test the LSTM claim properly (see the [RNN and LSTM note](notes/05-rnn-lstm.md)).

---

# Lộ trình

Kế hoạch 24 tuần học machine learning qua Jupyter Notebook, từ NumPy đến một dự án cuối khóa. Kế hoạch là định hướng, không phải lời hứa.

**Cách đọc cột trạng thái**

- **Đã áp dụng**: ý tưởng được dùng trong một dự án của repo này, kèm thí nghiệm có thể chỉ ra. Điều này **không** có nghĩa là mình đã thành thạo.
- **Áp dụng một phần**: một phần việc đã làm; phần còn thiếu được nêu rõ.
- **Chưa ghi nhận**: có thể mình đã học ở nơi khác, nhưng repo này chưa có bằng chứng nên mình không tuyên bố.

| Tuần | Chủ đề | Trạng thái | Ở đâu |
|---|---|---|---|
| 1 | NumPy, pandas, matplotlib, đại số tuyến tính cơ bản | Chưa ghi nhận | |
| 2 | Xác suất, gradient, quy trình ML, rò rỉ dữ liệu | Áp dụng một phần | [chia theo thời gian](notes/07-honest-experiments.md) |
| 3 | K-Means | Chưa ghi nhận | |
| 4 | Gom cụm phân cấp, DBSCAN | Chưa ghi nhận | |
| 5 | PCA và SVD | Chưa ghi nhận | |
| 6 | Gaussian mixture, ICA | Chưa ghi nhận | |
| 7 | Hồi quy tuyến tính, regularization | Chưa ghi nhận | |
| 8 | Logistic regression, Naive Bayes, precision / recall / F1 | Áp dụng một phần | [ghi chú về chỉ số](notes/03-classification-metrics.md) |
| 9 | LDA, SVM | Chưa ghi nhận | |
| 10 | Cây quyết định, ensemble, cross-validation | Chưa ghi nhận | |
| 11 | Gradient descent và tối ưu | Chưa ghi nhận | |
| 12 | Momentum, Adam, scheduler | Áp dụng một phần | AdamW + OneCycleLR dùng trong mọi dự án |
| 13 | Perceptron, MLP, lan truyền ngược | Chưa ghi nhận | |
| 14 | Cơ bản PyTorch: tensor, autograd, vòng lặp huấn luyện | Đã áp dụng | vòng lặp huấn luyện trong [`common.py`](../projects/vision-toolkit/training/classification/common.py) |
| 15 | Chẩn đoán huấn luyện: overfitting, chuẩn hóa, dropout | Áp dụng một phần | BatchNorm và dropout trong [CNN](notes/01-cnn-and-channels.md) |
| 16 | CNN: tích chập, pooling, receptive field | Đã áp dụng | [ghi chú CNN](notes/01-cnn-and-channels.md) |
| 17 | Transfer learning | Đã áp dụng | [ghi chú ResNet / EfficientNet](notes/02-transfer-learning.md) |
| 18 | Dự án thị giác máy tính có phân tích lỗi; khái niệm phân đoạn | Đã áp dụng | [Vision Toolkit](../projects/vision-toolkit), [ghi chú U-Net](notes/04-unet-segmentation.md) |
| 19 | Phân loại văn bản cổ điển (TF-IDF, Naive Bayes, SVM tuyến tính) | Áp dụng một phần | [TF-IDF + Logistic Regression / SVM](../projects/model-study/results/REPORT.md); chưa làm Naive Bayes |
| 20 | Topic modeling (LDA, LSA) | Chưa ghi nhận | |
| 21 | Embedding, trực giác RNN / LSTM, attention | Áp dụng một phần | [ghi chú RNN và LSTM](notes/05-rnn-lstm.md), [ghi chú Transformer](notes/06-transformer-attention.md) |
| 22 | Transformer và fine-tuning | Áp dụng một phần | Fine-tune DistilBERT trong [Sequence Modeling](../projects/sequence-modeling); [đã có so sánh TF-IDF + SVM](../projects/model-study/results/REPORT.md) |
| 23 | Chọn dự án cuối khóa | Chưa ghi nhận | |
| 24 | Ablation, phân tích lỗi, trình bày 10 phút | Chưa ghi nhận | |

## Việc tiếp theo

Những việc mình muốn làm tiếp, không theo thứ tự cố định:

- [x] Baseline TF-IDF + SVM tuyến tính cho IMDB, để kết quả DistilBERT có so sánh trung thực.
- [ ] Chạy lại các thí nghiệm chính với nhiều seed và báo cáo độ dao động.
- [ ] Ablation sạch cho mức cải thiện của phân đoạn (encoder pretrained, kích thước ảnh, trọng số viền, từng thứ một).
- [ ] Thí nghiệm mà trí nhớ xa thật sự quan trọng, để kiểm tra đúng nhận định về LSTM (xem [ghi chú RNN và LSTM](notes/05-rnn-lstm.md)).

2026-11-17: [Kaggle results](../projects/model-study/results/REPORT.md) include 3 seeds for waste ablation and students. The clean segmentation rerun and multi-seed IMDB comparison remain pending.
