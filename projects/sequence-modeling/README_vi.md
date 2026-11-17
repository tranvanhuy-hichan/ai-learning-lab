# Sequence Modeling: RNN, LSTM và Transformer

[English](README.md) | [Tiếng Việt](README_vi.md)

> Một phần của [AI Learning Lab](../../README_vi.md), repo học tập cá nhân. Mỗi mô hình có một bài toán thể hiện đúng thế mạnh của nó, và các notebook nói thẳng chỗ nào thí nghiệm **không** cho kết quả như mình dự đoán.

| Notebook | Mô hình | Bài toán | Kết quả test |
|---|---|---|---|
| [01_rnn_timeseries.ipynb](01_rnn_timeseries.ipynb) | RNN | Nhiệt độ thấp nhất ngày mai ở Melbourne từ 30 ngày trước | MAE 1,73 °C (LSTM 1,71, "bằng hôm nay" 2,04) |
| [02_lstm_text.ipynb](02_lstm_text.ipynb) | LSTM | Cảm xúc review IMDB, và RNN so với LSTM theo nhiều độ dài review | BiLSTM 87,7% |
| [03_transformer_text.ipynb](03_transformer_text.ipynb) | Transformer | Cùng bài IMDB với DistilBERT fine-tune | **91,6%** |
| [04_compare.ipynb](04_compare.ipynb) | Tất cả | Bảng so sánh và biểu đồ RNN so với LSTM theo độ dài | |

Các notebook được lưu kèm kết quả, nên bạn đọc được mà không cần chạy gì. Ghi chú phía sau: [RNN và LSTM](../../learning/notes/05-rnn-lstm.md), [Transformer](../../learning/notes/06-transformer-attention.md).

## Chạy thử

```bash
pip install -r requirements.txt
python scripts/download_data.py      # ~41 MB: nhiệt độ Melbourne + IMDB (cần internet)
jupyter lab                          # chạy notebook 01 đến 04 theo thứ tự
```

Fine-tune DistilBERT (notebook 03) mất khoảng 10 phút trên GPU 4 GB. Các file kết quả nhỏ nằm trong `outputs/`; trọng số đã train (`*.pt`) không được commit.

## Lưu ý trung thực

- Mọi kết quả là một lần chạy với một seed.
- RNN và LSTM trên bài nhiệt độ chỉ khác nhau 0,02 °C, quá nhỏ để kết luận.
- Khoảng cách RNN và LSTM trên IMDB **không** tăng theo độ dài review, nên thí nghiệm đó không chứng minh LSTM nhớ được từ ở xa.
- Mức tăng của DistilBERT trộn lẫn kiến trúc và pretrained; vẫn còn thiếu baseline TF-IDF + SVM.
- Các mô hình IMDB chỉ hiểu tiếng Anh.

## Dữ liệu

- Nhiệt độ thấp nhất hằng ngày ở Melbourne, từ repo GitHub `jbrownlee/Datasets`.
- Review IMDB, từ bộ dữ liệu Hugging Face `stanfordnlp/imdb`.

Hãy kiểm tra giấy phép từng bộ dữ liệu trước khi dùng lại; dữ liệu không được kèm trong repo.
