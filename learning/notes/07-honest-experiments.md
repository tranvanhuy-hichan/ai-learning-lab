# Running honest experiments

## English

Mistakes and habits I picked up while building these projects.

1. **State a hypothesis, then let the data disagree.** I predicted the LSTM's lead over the RNN would grow with review length. It did not, and the write-up says so ([note 05](05-rnn-lstm.md)).
2. **Change one thing at a time.** My better segmentation model changed several settings at once, so its gain cannot be credited to any one. I wrote that down instead of claiming "pretrained encoders help by 0.04".
3. **One run is one sample.** Every number here is a single seed. Small gaps (RNN 1.73 vs LSTM 1.71 °C) are noise until repeated.
4. **Beat a baseline first.** "Tomorrow equals today" scored 2.04 °C. Without it, 1.73 °C means nothing.
5. **Split by time for time series.** Random splits let the model see the future. I split 70/15/15 in order and normalised with training statistics only.
6. **Measure where you claim.** The same U-Net scored 0.767 at 128 px and 0.733 at original resolution. I report both and say which is which.
7. **Check that the app matches the notebook.** Before building the web demo I re-ran the inference code on the same test split and got the notebook's accuracy exactly (96.13%).
8. **Say what you did not test.** The waste classifier was never run on cluttered real-world photos; the limitations section says so.

## Tiếng Việt

Những lỗi và thói quen mình rút ra khi làm các dự án này.

1. **Nêu giả thuyết, rồi để dữ liệu phản bác.** Mình dự đoán LSTM sẽ hơn RNN nhiều hơn khi review dài ra. Thực tế không, và bài viết nói rõ điều đó ([ghi chú 05](05-rnn-lstm.md)).
2. **Mỗi lần chỉ đổi một thứ.** Mô hình phân đoạn tốt hơn của mình đổi nhiều thiết lập cùng lúc, nên mức cải thiện không quy được cho thứ nào. Mình ghi lại điều đó thay vì khẳng định "encoder pretrained giúp tăng 0,04".
3. **Một lần chạy chỉ là một mẫu.** Mọi con số ở đây là một seed. Chênh lệch nhỏ (RNN 1,73 so với LSTM 1,71 °C) chỉ là nhiễu cho đến khi lặp lại.
4. **Thắng một mốc đơn giản trước.** "Ngày mai bằng hôm nay" đạt 2,04 °C. Không có mốc này, 1,73 °C chẳng có nghĩa gì.
5. **Chia theo thời gian với chuỗi thời gian.** Chia ngẫu nhiên khiến mô hình nhìn thấy tương lai. Mình chia 70/15/15 theo thứ tự và chỉ chuẩn hóa bằng thống kê của tập train.
6. **Đo ở đúng chỗ mình khẳng định.** Cùng một U-Net đạt 0,767 ở 128 px và 0,733 ở độ phân giải gốc. Mình báo cả hai và nói rõ cái nào là cái nào.
7. **Kiểm tra ứng dụng khớp với notebook.** Trước khi làm bản demo web, mình chạy lại code suy luận trên cùng tập test và ra đúng accuracy của notebook (96,13%).
8. **Nói rõ điều chưa kiểm tra.** Bộ phân loại rác chưa từng chạy trên ảnh thực tế lộn xộn; phần hạn chế nói rõ điều đó.
