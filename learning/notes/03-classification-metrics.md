# Classification metrics: precision, recall, F1, support

## English

**Question I had:** what are the four columns of `classification_report`?

For one class (say `plastic`):

- **Precision** = correct "plastic" predictions / all things predicted "plastic". Low precision: it calls other things plastic.
- **Recall** = correct "plastic" predictions / all real plastic items. Low recall: it misses plastic.
- **F1** = `2·P·R / (P + R)`, the harmonic mean. It is high only when both are high.
- **Support** = how many real items of that class are in the test set. It is a count, not a quality score; small support means a noisier estimate.

Averages: **macro** gives every class the same weight; **weighted** weights by support; **accuracy** counts all items equally.

**Why it matters here:** the waste data is imbalanced (`clothes` about 5,300 images, `brown-glass` about 600). Accuracy is dominated by `clothes`. For the scratch CNN, accuracy was 83.1% but macro-F1 only 0.809, which is the more honest summary of how it treats rare classes. I report both, plus per-class F1.

## Tiếng Việt

**Câu hỏi của mình:** bốn cột của `classification_report` là gì?

Với một lớp (ví dụ `plastic`):

- **Precision** = số dự đoán "plastic" đúng / tất cả những gì bị đoán là "plastic". Precision thấp: hay gọi thứ khác là nhựa.
- **Recall** = số dự đoán "plastic" đúng / tất cả món nhựa thật. Recall thấp: bỏ sót nhựa.
- **F1** = `2·P·R / (P + R)`, trung bình điều hòa. Chỉ cao khi cả hai cùng cao.
- **Support** = số mẫu thật của lớp đó trong tập test. Đây là số đếm, không phải điểm chất lượng; support nhỏ nghĩa là ước lượng nhiễu hơn.

Các trung bình: **macro** cho mọi lớp trọng số bằng nhau; **weighted** tính theo support; **accuracy** đếm mọi mẫu như nhau.

**Vì sao quan trọng ở đây:** dữ liệu rác mất cân bằng (`clothes` khoảng 5.300 ảnh, `brown-glass` khoảng 600). Accuracy bị `clothes` chi phối. Với CNN từ đầu, accuracy là 83,1% nhưng macro-F1 chỉ 0,809, đây là tóm tắt trung thực hơn về cách mô hình đối xử với lớp hiếm. Mình báo cáo cả hai, kèm F1 từng lớp.
