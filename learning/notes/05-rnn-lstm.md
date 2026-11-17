# RNN and LSTM

## English

**RNN:** at each step `h_t = tanh(W·x_t + U·h_(t-1) + b)`. The same weights are reused at every step and `h_t` is the memory.

**The problem:** training sends gradients back through every step, multiplying by the same matrix again and again, so they shrink (vanishing gradient) or blow up. Far-away inputs barely influence learning.

**LSTM** adds a *cell state* `c_t` and three gates: **forget** (drop old information), **input** (write new information), **output** (use the memory now). Gradients flow along `c_t` mostly by addition, so they survive longer.

**Experiment 1: time series.** Predict tomorrow's minimum temperature in Melbourne from the last 30 days (3,650 days, chronological 70/15/15 split).

| Model | MAE | Parameters |
|---|---|---|
| "Same as today" | 2.04 °C | none |
| 7-day average | 1.97 °C | none |
| RNN | 1.73 °C | 12.7 K |
| LSTM | 1.71 °C | 50.5 K |

Both beat the baselines by about 15%. RNN and LSTM are too close to separate with one run on 548 test days, and the RNN has a quarter of the parameters. For a short-range signal the plain RNN is enough.

**Experiment 2: IMDB sentiment at different review lengths** (same architecture, only the maximum length changes):

| Max length | RNN | LSTM | Gap |
|---|---|---|---|
| 50 | 0.669 | 0.757 | +8.8 |
| 100 | 0.686 | 0.808 | +12.3 |
| 200 | 0.762 | 0.839 | +7.7 |
| 400 | 0.786 | 0.861 | +7.5 |

**My hypothesis was wrong.** I expected the gap to *grow* with length. It did not. The LSTM was better everywhere (it is easier to train), but both improved with more text because sentiment words anywhere in a review help. So this experiment does **not** prove the "long-range memory" claim; that needs a task where a distant token truly matters (on the roadmap).

## Tiếng Việt

**RNN:** ở mỗi bước `h_t = tanh(W·x_t + U·h_(t-1) + b)`. Cùng bộ trọng số được dùng ở mọi bước và `h_t` là bộ nhớ.

**Vấn đề:** khi huấn luyện, gradient đi ngược qua mọi bước, nhân đi nhân lại cùng một ma trận nên bị co lại (vanishing gradient) hoặc bùng nổ. Đầu vào ở xa gần như không ảnh hưởng đến việc học.

**LSTM** thêm *cell state* `c_t` và ba cổng: **quên** (bỏ thông tin cũ), **vào** (ghi thông tin mới), **ra** (dùng bộ nhớ ở bước này). Gradient chạy dọc `c_t` chủ yếu bằng phép cộng nên sống lâu hơn.

**Thí nghiệm 1: chuỗi thời gian.** Dự đoán nhiệt độ thấp nhất ngày mai ở Melbourne từ 30 ngày trước (3.650 ngày, chia theo thời gian 70/15/15).

| Mô hình | MAE | Tham số |
|---|---|---|
| "Bằng hôm nay" | 2,04 °C | không có |
| Trung bình 7 ngày | 1,97 °C | không có |
| RNN | 1,73 °C | 12,7 K |
| LSTM | 1,71 °C | 50,5 K |

Cả hai tốt hơn mốc khoảng 15%. RNN và LSTM quá sát nhau để tách bạch với một lần chạy trên 548 ngày test, và RNN chỉ có một phần tư số tham số. Với tín hiệu phụ thuộc gần, RNN thuần là đủ.

**Thí nghiệm 2: cảm xúc IMDB theo độ dài review** (cùng kiến trúc, chỉ đổi độ dài tối đa):

| Độ dài tối đa | RNN | LSTM | LSTM hơn |
|---|---|---|---|
| 50 | 0,669 | 0,757 | +8,8 |
| 100 | 0,686 | 0,808 | +12,3 |
| 200 | 0,762 | 0,839 | +7,7 |
| 400 | 0,786 | 0,861 | +7,5 |

**Giả thuyết của mình đã sai.** Mình dự đoán khoảng cách sẽ *tăng* theo độ dài. Thực tế không. LSTM tốt hơn ở mọi độ dài (vì dễ huấn luyện hơn), nhưng cả hai đều khá hơn khi có nhiều chữ hơn vì từ biểu lộ cảm xúc ở đâu trong review cũng giúp ích. Vậy thí nghiệm này **không** chứng minh nhận định "nhớ xa"; cần một bài toán mà token ở xa thật sự quan trọng (có trong lộ trình).
