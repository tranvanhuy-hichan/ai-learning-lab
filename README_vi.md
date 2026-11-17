# AI Learning Lab

[English](README.md) | [Tiếng Việt](README_vi.md)

[![CI](https://github.com/tranvanhuy-hichan/ai-learning-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/tranvanhuy-hichan/ai-learning-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)

Repo cá nhân nơi mình **áp dụng những gì học được về AI**. Mình học từ môn học và sách, viết một ghi chú ngắn, rồi xây và đo một mô hình ở đây. Đây là dự án học tập: không có gì sẵn sàng cho sản xuất, và repo không được thiết lập để nhận đóng góp.

> **Học → Ghi chú → Áp dụng**
> [Lộ trình](learning/ROADMAP.md) · [Nhật ký học](learning/log) · [Ghi chú](learning/notes) · [Dự án](projects)

![Công cụ phân đoạn, tiếng Việt, giao diện sáng](docs/images/web-segmentation-vi-light.png)

## Chạy thử

```bash
git clone https://github.com/tranvanhuy-hichan/ai-learning-lab.git
cd ai-learning-lab/projects/lab-web
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m labweb          # mở http://127.0.0.1:8000
```

Web app song ngữ (Anh / Việt) có giao diện sáng và tối. Chạy được trên CPU. Chi tiết: [`projects/lab-web`](projects/lab-web).

## Các dự án

| Dự án | Nội dung | Kết quả nổi bật |
|---|---|---|
| [`vision-toolkit`](projects/vision-toolkit) | Phân loại rác (CNN, ResNet18, EfficientNet-B0) và phân đoạn thú cưng (U-Net, U-Net encoder ResNet18), kèm app Gradio | EfficientNet-B0 đạt 96,1% accuracy; mIoU phân đoạn 0,767 và 0,809 |
| [`sequence-modeling`](projects/sequence-modeling) | RNN, LSTM và Transformer trên dự báo nhiệt độ và cảm xúc IMDB | DistilBERT 91,6% so với BiLSTM 87,7%; RNN và LSTM tốt hơn "bằng hôm nay" khoảng 15% |
| [`lab-web`](projects/lab-web) | Một giao diện web cho mọi mô hình ở trên | 4 công cụ, tiếng Anh và tiếng Việt |

## Mình đang học gì

[Lộ trình](learning/ROADMAP.md) theo dõi kế hoạch 24 tuần và nói thật phần nào chỉ mới được *áp dụng* ở đây và phần nào mình chưa ghi nhận. Các ghi chú ngắn gồm [CNN](learning/notes/01-cnn-and-channels.md), [transfer learning](learning/notes/02-transfer-learning.md), [chỉ số phân loại](learning/notes/03-classification-metrics.md), [U-Net](learning/notes/04-unet-segmentation.md), [RNN và LSTM](learning/notes/05-rnn-lstm.md), [Transformer](learning/notes/06-transformer-attention.md) và [cách làm thí nghiệm trung thực](learning/notes/07-honest-experiments.md), trong đó có một giả thuyết của mình mà dữ liệu đã bác bỏ.

## Các con số trung thực đến đâu?

- Mọi kết quả đến từ một lần huấn luyện với một seed. Chênh lệch nhỏ chỉ là nhiễu cho đến khi lặp lại.
- Một số so sánh đổi nhiều thứ cùng lúc (ví dụ hai mô hình phân đoạn), nên mình không khẳng định thay đổi nào có tác dụng.
- Giới hạn được ghi ngay cạnh mỗi kết quả: bộ phân loại rác chưa được thử trên ảnh thực tế lộn xộn; các mô hình cảm xúc chỉ hiểu tiếng Anh.

## Cấu trúc repo

```
learning/        lộ trình, nhật ký theo ngày, ghi chú song ngữ
projects/
  vision-toolkit/      phân loại + phân đoạn, app Gradio, notebook huấn luyện, trọng số
  sequence-modeling/   notebook RNN / LSTM / Transformer và kết quả
  lab-web/             web app song ngữ FastAPI + JS thuần
docs/images/     ảnh chụp dùng trong README
```

`make web` chạy web app, `make test` chạy test của mọi dự án và `make lint` chạy ruff. CI chạy các kiểm tra tương tự.

## Minh bạch

Mình dùng trợ lý lập trình AI (Claude Code) để giúp viết code và giải thích từ các câu hỏi của mình. Mình tự huấn luyện và đọc kết quả, và các ghi chú ghi lại những chỗ dự đoán đầu tiên của mình sai.

## Giấy phép

MIT, xem [LICENSE](LICENSE). Các bộ dữ liệu giữ giấy phép riêng và không được kèm theo: Oxford-IIIT Pet (CC BY-SA 4.0), bộ dữ liệu rác 12 lớp từ Hugging Face hub (hãy kiểm tra giấy phép trước khi dùng lại), nhiệt độ hằng ngày ở Melbourne và review IMDB. Backbone pretrained lấy từ torchvision và Hugging Face.
