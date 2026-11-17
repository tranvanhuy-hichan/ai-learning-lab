"use strict";
/* AI Learning Lab front-end: vanilla JS, no build step. Bilingual (EN / VI). */

const REPO = "https://github.com/tranvanhuy-hichan/ai-learning-lab";

/* ----------------------------------------------------------------- translations */
const I18N = {
  en: {
    "waste.unknown": "Outside the supported waste categories / cannot identify",
    "waste.uncertain": "Not confident enough to classify",
    "waste.rejectNote": "The model cannot reliably classify this image as waste. Scores below only compare its 12 known classes; they do not confirm the object is waste.",
    "waste.uncertainNote": "The model is unsure. Try a clear photo of a single waste item.",
    "study.nav": "Experiments",
    "study.title": "Measured experiments",
    "study.desc": "Kaggle training results and a local IMDB baseline. These are separate experiments; the interactive tools above keep their installed models.",
    "study.stats": "Mean ± sample standard deviation in percentage points. n is the number of seeds; a single run has no estimated spread.",
    "study.report": "Full report and limitations",
    "study.waste": "Waste classification ablation",
    "study.seg": "Segmentation: historical recipe",
    "study.imdb": "IMDB comparison",
    "study.compression": "Post-training compression (one run each)",
    "study.model": "Variant / model",
    "study.score": "Accuracy (%)",
    "study.miou": "mIoU (%)",
    "study.border": "Border IoU (%)",
    "study.size": "File MB",
    "study.seeds": "n",
    "study.segNote": "B→C changes learning rate AND epochs. D→E changes resolution AND batch size; E has only one seed and uses a different evaluation resolution. Border weighting improves border IoU while reducing mIoU in these runs.",
    "study.compressionNote": "INT8 loses substantial accuracy. Pruned checkpoints retain dense shapes; file size does not measure inference speed. CPU latency is not measured.",
    "study.error": "Experiment tables could not be loaded.",
    "camera.open": "Take a photo",
    "camera.title": "Camera",
    "camera.shoot": "Capture",
    "camera.cancel": "Cancel",
    "camera.unavailable": "Camera needs HTTPS or localhost and a supported browser. You can still upload a photo.",
    "camera.error": "Cannot open camera. Check camera permissions and whether another app is using it.",
    "camera.captureError": "Could not capture a photo. Please try again.",
    "hero.badge": "Learning in public",
    "hero.title": "What I am learning about AI, and the models I built to test it",
    "hero.lead": "I study from courses and books, write a short note, then train and measure a model to see whether I really understood. Each section below runs one of those models on your own input.",
    "nav.waste": "Waste", "nav.seg": "Segmentation", "nav.sent": "Sentiment", "nav.fc": "Forecast",
    "nav.journey": "Journey", "theme.title": "Light / dark",
    "tbl.task": "Task", "tbl.model": "Best model", "tbl.result": "Result", "tbl.note": "For comparison",
    "tbl.cap": "Table 1. Test-set results. Every number is a single training run with one seed, so small gaps are not conclusive.",
    "tbl.waste": "12-class accuracy", "tbl.seg": "mIoU", "tbl.sent": "accuracy", "tbl.fc": "mean absolute error",
    "tbl.scratch": "CNN from scratch",
    "tab.waste": "Waste classification", "tab.seg": "Pet segmentation", "tab.sent": "Review sentiment", "tab.fc": "Temperature forecast",
    learned: "What I learned",
    "drop.title": "Drop an image here", "drop.sub": "or click to choose (JPG/PNG, up to 8 MB)",
    "stat.waste": "waste classifier accuracy", "stat.seg": "pet segmentation mIoU", "stat.sent": "review sentiment accuracy", "stat.fc": "forecast error (MAE)",
    "fact.acc": "Accuracy", "fact.f1": "Macro-F1", "fact.params": "parameters", "fact.miou": "mIoU", "fact.border": "Border IoU", "fact.mae": "MAE", "fact.base": "Same as today",

    "waste.title": "Waste classifier",
    "waste.desc": "Drop a photo of one waste item. A fine-tuned EfficientNet-B0 says which of 12 classes it belongs to.",
    "waste.l1": "Transfer learning: reuse ImageNet features and retrain the head",
    "waste.l2": "Class-weighted loss for an imbalanced dataset",
    "waste.l3": "Read accuracy together with macro-F1 and per-class results",
    "waste.pred": "Prediction", "waste.tip": "What to do with it",
    "waste.note": "Trained on photos of a single item on a plain background. Messy real-world photos may do worse and I have not measured that. It always picks one of the 12 classes.",
    "waste.off": "The waste classifier weights are not installed on this server.",

    "seg.title": "Pet segmentation",
    "seg.desc": "Every pixel gets a label: pet, background or border. Compare a U-Net written from scratch with one that uses a pretrained ResNet18 encoder.",
    "seg.l1": "U-Net: encoder, decoder and skip connections",
    "seg.l2": "A pretrained encoder lifts mIoU, but several settings changed at once",
    "seg.l3": "mIoU and per-class IoU expose the thin border class",
    "seg.model": "Model", "seg.examples": "Try an example", "seg.opacity": "Mask opacity",
    "seg.pet": "Pet", "seg.background": "Background", "seg.border": "Border",
    "seg.note": "Trained only on cats and dogs (Oxford-IIIT Pet). Other animals or objects will not be segmented reliably.",
    "seg.off": "No segmentation weights are installed on this server.",

    "sent.title": "Movie review sentiment",
    "sent.desc": "Type a movie review. An LSTM and a Transformer (DistilBERT) both say whether it is positive or negative, so you can compare them.",
    "sent.l1": "LSTM gates keep context that a plain RNN forgets",
    "sent.l2": "Self-attention sees the whole text at once; pretraining does the rest",
    "sent.l3": "A bigger model costs more: about 24x the parameters for roughly 4 points of accuracy",
    "sent.lang": "Both models were trained on English IMDB reviews, so write in English. Vietnamese text will give meaningless results.",
    "sent.ph": "Type an English movie review…", "sent.examples": "Examples", "sent.go": "Analyze", "sent.busy": "Analyzing…",
    "sent.ex1": "Clear praise", "sent.ex2": "Clear complaint", "sent.ex3": "Twist: hate → good", "sent.ex4": "Twist: slow → won me over",
    "sent.lstm": "BiLSTM (RNN family)", "sent.distil": "DistilBERT (Transformer)",
    "sent.positive": "Positive", "sent.negative": "Negative", "sent.tokens": "tokens read", "sent.cut": "text was cut",
    "sent.nodistil": "DistilBERT is an optional 268 MB download and is not installed on this server, so only the LSTM is shown.",
    "sent.empty": "Please type a review first.",

    "fc.title": "Next-day temperature",
    "fc.desc": "Pick a 30-day window of Melbourne's daily minimum temperatures. An RNN and an LSTM predict tomorrow; compare them with simple baselines and the real value.",
    "fc.l1": "A plain RNN is enough when the signal is short-range",
    "fc.l2": "Split by time, never at random, to avoid peeking at the future",
    "fc.l3": "First beat a baseline: “same as today”",
    "fc.pick": "Choose a window", "fc.history": "Last 30 days", "fc.today": "today", "fc.tomorrow": "tomorrow",
    "fc.model": "Model", "fc.forecast": "Forecast", "fc.error": "Error",
    "fc.rnn": "RNN", "fc.lstm": "LSTM", "fc.base": "Same as today", "fc.mean7": "7-day average", "fc.actual": "Actual tomorrow",
    "fc.custom": "Use your own 30 values", "fc.custom.ph": "30 temperatures in °C, separated by commas", "fc.custom.go": "Forecast",
    "fc.custom.bad": "I need exactly 30 numbers between -60 and 60.",
    "fc.note": "Over the whole test set the RNN and LSTM are about 15% better than “same as today”. On a single window either one can lose.",
    "fc.off": "The forecast models are not installed on this server.",

    "journey.title": "From a lesson to a working demo",
    "journey.lead": "This site is the last step of a loop I repeat for every topic.",
    "journey.s1.t": "Learn", "journey.s1.d": "Courses, books and docs. I write down what I did not understand.",
    "journey.s2.t": "Note", "journey.s2.d": "A short bilingual note per topic and a dated log entry.",
    "journey.s3.t": "Apply", "journey.s3.d": "Train a model, measure it honestly, and add it to this app.",
    "journey.roadmap": "Roadmap", "journey.log": "Learning log", "journey.notes": "Notes",
    "foot.1": "Built while learning AI. Results come from single training runs on public datasets and are for education only.",
    "foot.2": "Uploaded images are processed in memory and are not saved by this app.",
    "err.net": "Cannot reach the server.", "err.generic": "Something went wrong",
  },
  vi: {
    "waste.unknown": "Ngoài các nhóm rác hỗ trợ / không xác định được",
    "waste.uncertain": "Chưa đủ tin cậy để phân loại",
    "waste.rejectNote": "Mô hình không thể phân loại ảnh này là rác một cách đáng tin cậy. Các điểm bên dưới chỉ so sánh 12 lớp đã học, không xác nhận vật thể là rác.",
    "waste.uncertainNote": "Mô hình chưa chắc chắn. Hãy thử ảnh rõ của một món rác riêng lẻ.",
    "study.nav": "Thí nghiệm",
    "study.title": "Kết quả thí nghiệm",
    "study.desc": "Kết quả huấn luyện trên Kaggle và baseline IMDB chạy cục bộ. Đây là các thí nghiệm riêng; các công cụ phía trên dùng mô hình đang được cài.",
    "study.stats": "Trung bình ± độ lệch chuẩn mẫu, theo điểm phần trăm. n là số seed; một lượt chạy chưa có ước lượng độ dao động.",
    "study.report": "Báo cáo đầy đủ và giới hạn",
    "study.waste": "Ablation phân loại rác",
    "study.seg": "Phân đoạn: cấu hình cũ",
    "study.imdb": "So sánh IMDB",
    "study.compression": "Nén sau huấn luyện (mỗi phương pháp một lượt)",
    "study.model": "Biến thể / mô hình",
    "study.score": "Accuracy (%)",
    "study.miou": "mIoU (%)",
    "study.border": "IoU viền (%)",
    "study.size": "Dung lượng MB",
    "study.seeds": "n",
    "study.segNote": "B→C đổi cả learning rate VÀ số epoch. D→E đổi cả độ phân giải VÀ batch size; E chỉ có một seed và đánh giá ở độ phân giải khác. Tăng trọng số viền giúp IoU viền tăng nhưng mIoU giảm trong các lượt này.",
    "study.compressionNote": "INT8 làm giảm mạnh accuracy. Checkpoint sau pruning vẫn có tensor cùng kích thước; dung lượng tệp không thể hiện tốc độ suy luận. Chưa đo độ trễ CPU.",
    "study.error": "Không tải được bảng thí nghiệm.",
    "camera.open": "Chụp ảnh",
    "camera.title": "Camera",
    "camera.shoot": "Chụp",
    "camera.cancel": "Hủy",
    "camera.unavailable": "Camera cần HTTPS hoặc localhost và trình duyệt hỗ trợ. Bạn vẫn có thể tải ảnh lên.",
    "camera.error": "Không mở được camera. Hãy kiểm tra quyền truy cập và ứng dụng khác đang dùng camera.",
    "camera.captureError": "Không chụp được ảnh. Hãy thử lại.",
    "hero.badge": "Học công khai",
    "hero.title": "Mình đang học gì về AI, và các mô hình mình xây để kiểm chứng",
    "hero.lead": "Mình học từ môn học và sách, viết một ghi chú ngắn, rồi huấn luyện và đo một mô hình để xem mình đã thật sự hiểu chưa. Mỗi phần bên dưới chạy một mô hình như vậy trên dữ liệu của chính bạn.",
    "nav.waste": "Phân loại rác", "nav.seg": "Phân đoạn", "nav.sent": "Cảm xúc", "nav.fc": "Dự báo",
    "nav.journey": "Hành trình", "theme.title": "Sáng / tối",
    "tbl.task": "Bài toán", "tbl.model": "Mô hình tốt nhất", "tbl.result": "Kết quả", "tbl.note": "So sánh",
    "tbl.cap": "Bảng 1. Kết quả trên tập test. Mỗi con số là một lần huấn luyện với một seed, nên chênh lệch nhỏ chưa mang tính kết luận.",
    "tbl.waste": "accuracy 12 lớp", "tbl.seg": "mIoU", "tbl.sent": "accuracy", "tbl.fc": "sai số tuyệt đối trung bình",
    "tbl.scratch": "CNN từ đầu",
    "tab.waste": "Phân loại rác", "tab.seg": "Phân đoạn thú cưng", "tab.sent": "Cảm xúc review", "tab.fc": "Dự báo nhiệt độ",
    learned: "Mình đã học được",
    "drop.title": "Thả ảnh vào đây", "drop.sub": "hoặc bấm để chọn (JPG/PNG, tối đa 8 MB)",
    "stat.waste": "độ chính xác phân loại rác", "stat.seg": "mIoU phân đoạn thú cưng", "stat.sent": "độ chính xác cảm xúc review", "stat.fc": "sai số dự báo (MAE)",
    "fact.acc": "Độ chính xác", "fact.f1": "Macro-F1", "fact.params": "tham số", "fact.miou": "mIoU", "fact.border": "IoU viền", "fact.mae": "MAE", "fact.base": "Bằng hôm nay",

    "waste.title": "Phân loại rác",
    "waste.desc": "Thả ảnh một món rác. EfficientNet-B0 đã fine-tune cho biết nó thuộc 1 trong 12 loại.",
    "waste.l1": "Transfer learning: dùng lại đặc trưng ImageNet và huấn luyện lại phần đầu",
    "waste.l2": "Loss có trọng số lớp cho dữ liệu mất cân bằng",
    "waste.l3": "Đọc accuracy cùng macro-F1 và kết quả từng lớp",
    "waste.pred": "Dự đoán", "waste.tip": "Nên xử lý thế nào",
    "waste.note": "Mô hình học từ ảnh một món đồ trên nền đơn giản. Ảnh thực tế lộn xộn có thể kém hơn và mình chưa đo điều đó. Nó luôn chọn một trong 12 lớp.",
    "waste.off": "Máy chủ này chưa cài trọng số bộ phân loại rác.",

    "seg.title": "Phân đoạn thú cưng",
    "seg.desc": "Mỗi pixel được gán nhãn: con vật, nền hoặc viền. So sánh U-Net tự viết từ đầu với U-Net dùng encoder ResNet18 pretrained.",
    "seg.l1": "U-Net: encoder, decoder và skip connection",
    "seg.l2": "Encoder pretrained giúp tăng mIoU, nhưng nhiều thiết lập thay đổi cùng lúc",
    "seg.l3": "mIoU và IoU từng lớp cho thấy lớp viền mỏng là điểm yếu",
    "seg.model": "Mô hình", "seg.examples": "Thử một ảnh mẫu", "seg.opacity": "Độ phủ mask",
    "seg.pet": "Con vật", "seg.background": "Nền", "seg.border": "Viền",
    "seg.note": "Chỉ huấn luyện trên chó và mèo (Oxford-IIIT Pet). Động vật hay vật thể khác sẽ không được phân đoạn đáng tin cậy.",
    "seg.off": "Máy chủ này chưa cài trọng số phân đoạn.",

    "sent.title": "Cảm xúc review phim",
    "sent.desc": "Gõ một review phim. LSTM và Transformer (DistilBERT) cùng cho biết review tích cực hay tiêu cực để bạn so sánh.",
    "sent.l1": "Các cổng của LSTM giữ ngữ cảnh mà RNN thuần quên mất",
    "sent.l2": "Self-attention nhìn toàn bộ văn bản cùng lúc; pretrained lo phần còn lại",
    "sent.l3": "Mô hình lớn hơn tốn kém hơn: gấp khoảng 24 lần tham số để hơn khoảng 4 điểm accuracy",
    "sent.lang": "Cả hai mô hình được huấn luyện trên review IMDB tiếng Anh, hãy viết bằng tiếng Anh. Văn bản tiếng Việt sẽ cho kết quả vô nghĩa.",
    "sent.ph": "Gõ một review phim bằng tiếng Anh…", "sent.examples": "Ví dụ", "sent.go": "Phân tích", "sent.busy": "Đang phân tích…",
    "sent.ex1": "Khen rõ ràng", "sent.ex2": "Chê rõ ràng", "sent.ex3": "Chuyển ý: ghét → hay", "sent.ex4": "Chuyển ý: chán → chinh phục",
    "sent.lstm": "BiLSTM (họ RNN)", "sent.distil": "DistilBERT (Transformer)",
    "sent.positive": "Tích cực", "sent.negative": "Tiêu cực", "sent.tokens": "token đã đọc", "sent.cut": "văn bản bị cắt bớt",
    "sent.nodistil": "DistilBERT là bản tải thêm tùy chọn (268 MB) và chưa được cài trên máy chủ này, nên chỉ hiển thị LSTM.",
    "sent.empty": "Hãy gõ một review trước.",

    "fc.title": "Dự báo nhiệt độ ngày mai",
    "fc.desc": "Chọn một cửa sổ 30 ngày nhiệt độ thấp nhất hằng ngày ở Melbourne. RNN và LSTM dự đoán ngày mai; so với các mốc đơn giản và giá trị thật.",
    "fc.l1": "RNN thuần là đủ khi tín hiệu phụ thuộc gần",
    "fc.l2": "Chia theo thời gian, không ngẫu nhiên, để không nhìn trộm tương lai",
    "fc.l3": "Trước hết phải thắng một mốc đơn giản: “bằng hôm nay”",
    "fc.pick": "Chọn một cửa sổ", "fc.history": "30 ngày gần nhất", "fc.today": "hôm nay", "fc.tomorrow": "ngày mai",
    "fc.model": "Mô hình", "fc.forecast": "Dự báo", "fc.error": "Sai số",
    "fc.rnn": "RNN", "fc.lstm": "LSTM", "fc.base": "Bằng hôm nay", "fc.mean7": "Trung bình 7 ngày", "fc.actual": "Thực tế ngày mai",
    "fc.custom": "Dùng 30 giá trị của riêng bạn", "fc.custom.ph": "30 nhiệt độ (°C), cách nhau bằng dấu phẩy", "fc.custom.go": "Dự báo",
    "fc.custom.bad": "Cần đúng 30 số trong khoảng từ -60 đến 60.",
    "fc.note": "Trên toàn bộ tập test, RNN và LSTM tốt hơn “bằng hôm nay” khoảng 15%. Với một cửa sổ riêng lẻ, mô hình nào cũng có thể thua.",
    "fc.off": "Máy chủ này chưa cài mô hình dự báo.",

    "journey.title": "Từ bài học đến bản demo chạy được",
    "journey.lead": "Trang này là bước cuối của vòng lặp mình lặp lại với mỗi chủ đề.",
    "journey.s1.t": "Học", "journey.s1.d": "Môn học, sách và tài liệu. Mình ghi lại những chỗ chưa hiểu.",
    "journey.s2.t": "Ghi chú", "journey.s2.d": "Một ghi chú song ngữ ngắn cho mỗi chủ đề và một mục nhật ký có ngày.",
    "journey.s3.t": "Áp dụng", "journey.s3.d": "Huấn luyện mô hình, đo lường trung thực, rồi thêm vào ứng dụng này.",
    "journey.roadmap": "Lộ trình", "journey.log": "Nhật ký học", "journey.notes": "Ghi chú",
    "foot.1": "Xây dựng trong lúc học AI. Kết quả đến từ các lần huấn luyện đơn lẻ trên dữ liệu công khai và chỉ phục vụ học tập.",
    "foot.2": "Ảnh tải lên được xử lý trong bộ nhớ và ứng dụng này không lưu lại.",
    "err.net": "Không kết nối được máy chủ.", "err.generic": "Có lỗi xảy ra",
  },
};

const WASTE_VI = {
  battery: "Pin", biological: "Rác hữu cơ", "brown-glass": "Thủy tinh nâu", cardboard: "Bìa carton", clothes: "Quần áo",
  "green-glass": "Thủy tinh xanh", metal: "Kim loại", paper: "Giấy", plastic: "Nhựa", shoes: "Giày dép", trash: "Rác thông thường",
  "white-glass": "Thủy tinh trong",
};
const SENT_EXAMPLES = [
  ["sent.ex1", "A wonderful, moving film with brilliant acting."],
  ["sent.ex2", "Total waste of time, awful acting and a boring plot."],
  ["sent.ex3", "I expected to hate it, but it turned out to be surprisingly good."],
  ["sent.ex4", "The first hour was slow and dull, however the ending completely won me over."],
];
const SEG_COLORS = { pet: "#e63c3c", background: "#3c3cc8", border: "#fadc3c" };
const FC_COLORS = { rnn: "#1f5fbf", lstm: "#c2561b", base: "#8a8a85", actual: "#2c6e49" };

/* ----------------------------------------------------------------- helpers */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};
const state = {
  lang: store.get("lang") || ((navigator.language || "").toLowerCase().startsWith("vi") ? "vi" : "en"),
  theme: store.get("theme") || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"),
  study: null, models: null, waste: null, seg: { file: null, model: null, res: null, img: null, mask: null },
  sent: { results: null }, fc: { samples: [], idx: 0, values: null, actual: null, preds: null },
};
const t = (key) => (I18N[state.lang] && I18N[state.lang][key]) || I18N.en[key] || key;

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) el.append(c.nodeType ? c : document.createTextNode(c));
  return el;
}
function svg(tag, attrs = {}, text) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text != null) el.textContent = text;
  return el;
}
async function api(path, opts) {
  let r;
  try { r = await fetch(path, opts); } catch { throw new Error(t("err.net")); }
  if (!r.ok) {
    let detail = "";
    try { const j = await r.json(); detail = typeof j.detail === "string" ? j.detail : (j.detail && j.detail[0] && j.detail[0].msg) || ""; } catch { /* not json */ }
    throw new Error(detail || `${t("err.generic")} (${r.status})`);
  }
  return r.json();
}
function showMsg(id, text, info = false) {
  const el = $(id);
  el.hidden = !text; el.textContent = text || ""; el.classList.toggle("info", info);
}
const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const wasteName = (l) => (state.lang === "vi" ? WASTE_VI[l] || l : l.replace("-", " "));
const sentName = (l) => t(l === "positive" ? "sent.positive" : "sent.negative");

function renderBars(box, rows) {
  box.replaceChildren(...rows.map((r) => {
    const fill = h("div", { class: "fill" });
    if (r.color) fill.style.background = r.color;
    const row = h("div", { class: "bar" + (r.top ? " best" : "") }, h("span", {}, r.name), h("div", { class: "track" }, fill), h("span", { class: "v" }, r.text));
    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${Math.max(1, r.value * 100)}%`; }));
    return row;
  }));
}
function setupDrop(drop, input, onFile) {
  input.addEventListener("change", () => { if (input.files[0]) onFile(input.files[0]); input.value = ""; });
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) onFile(f); });
}
const loadImage = (src) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });

/* ----------------------------------------------------------------- camera */
function setupCamera() {
  const modal = $("#camera-modal"), video = $("#camera-video");
  const shoot = $("#camera-shoot");
  let stream = null, target = null, opener = null, request = 0;
  function close() {
    request++;
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null; video.srcObject = null; target = null;
    modal.close();
    if (opener) opener.focus();
  }
  async function open(button, onFile) {
    opener = button; target = onFile;
    const token = ++request;
    showMsg("#camera-msg", ""); shoot.disabled = true;
    modal.showModal(); $("#camera-close").focus();
    if (!navigator.mediaDevices?.getUserMedia) {
      showMsg("#camera-msg", t("camera.unavailable")); return;
    }
    try {
      const incoming = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false,
      });
      if (token !== request) { incoming.getTracks().forEach((track) => track.stop()); return; }
      stream = incoming; video.srcObject = stream;
      await video.play();
      if (token === request) shoot.disabled = false;
    } catch {
      if (token !== request) return;
      if (stream) stream.getTracks().forEach((track) => track.stop());
      stream = null; video.srcObject = null;
      showMsg("#camera-msg", t("camera.error"));
    }
  }
  [[$("#waste-camera-open"), runWaste], [$("#seg-camera-open"), runSeg]].forEach(([button, handler]) => {
    button.addEventListener("click", () => open(button, handler));
  });
  $("#camera-close").addEventListener("click", close);
  modal.addEventListener("cancel", (event) => { event.preventDefault(); close(); });
  modal.addEventListener("click", (event) => { if (event.target === modal) close(); });
  window.addEventListener("pagehide", close);
  document.addEventListener("visibilitychange", () => { if (document.hidden && modal.open) close(); });
  shoot.addEventListener("click", () => {
    if (!video.videoWidth || !video.videoHeight || !target) return;
    const canvas = $("#camera-canvas"), handler = target, token = request;
    const scale = Math.min(1, 1600 / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.round(video.videoWidth * scale); canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    shoot.disabled = true;
    canvas.toBlob((blob) => {
      if (token !== request) return;
      if (!blob) { shoot.disabled = false; showMsg("#camera-msg", t("camera.captureError")); return; }
      close(); handler(new File([blob], "camera.jpg", { type: "image/jpeg" }));
    }, "image/jpeg", 0.9);
  });
}

/* ----------------------------------------------------------------- stats & facts */
function fact(label, value) { return h("span", {}, `${label} `, h("b", {}, value)); }
function renderStats() {
  const m = (state.models && state.models.metrics) || {};
  const link = (id, key) => h("a", { href: `#${id}` }, t(key));
  const rows = [];
  if (m.waste) rows.push([link("waste", "tab.waste"), "EfficientNet-B0", pct(m.waste.accuracy, 1), `${t("tbl.waste")}`]);
  if (m.resnet_unet) rows.push([link("seg", "tab.seg"), "ResNet18-encoder U-Net", m.resnet_unet.miou.toFixed(3), `${t("tbl.seg")}; U-Net ${m.unet.miou.toFixed(3)}`]);
  if (m.distilbert) rows.push([link("sent", "tab.sent"), "DistilBERT", pct(m.distilbert.accuracy, 1), `${t("tbl.sent")}; BiLSTM ${pct(m.lstm.accuracy, 1)}`]);
  if (m.lstm_temp) rows.push([link("fc", "tab.fc"), "LSTM", `${m.lstm_temp.mae.toFixed(2)} °C`, `${t("tbl.fc")}; RNN ${m.rnn_temp.mae.toFixed(2)}, ${t("fact.base")} ${m.persistence_mae.toFixed(2)}`]);
  $("#results-body").replaceChildren(...rows.map((r) => h("tr", {}, ...r.map((c) => h("td", {}, c)))));
  const f = (id, chips) => $(id).replaceChildren(...chips);
  if (m.waste) f("#facts-waste", [fact(t("fact.acc"), pct(m.waste.accuracy, 1)), fact(t("fact.f1"), m.waste.macro_f1.toFixed(3)), fact(t("fact.params"), `${m.waste.params_m}M`)]);
  if (m.unet && m.resnet_unet) f("#facts-seg", [fact("U-Net " + t("fact.miou"), m.unet.miou.toFixed(3)), fact("ResNet-U-Net " + t("fact.miou"), m.resnet_unet.miou.toFixed(3)), fact(t("fact.border"), `${m.unet.border_iou.toFixed(2)} → ${m.resnet_unet.border_iou.toFixed(2)}`)]);
  if (m.lstm && m.distilbert) f("#facts-sent", [fact("BiLSTM", pct(m.lstm.accuracy, 1)), fact("DistilBERT", pct(m.distilbert.accuracy, 1)), fact(t("fact.params"), `${m.lstm.params_m}M vs ${m.distilbert.params_m}M`)]);
  if (m.rnn_temp) f("#facts-fc", [fact("RNN " + t("fact.mae"), `${m.rnn_temp.mae.toFixed(2)} °C`), fact("LSTM " + t("fact.mae"), `${m.lstm_temp.mae.toFixed(2)} °C`), fact(t("fact.base"), `${m.persistence_mae.toFixed(2)} °C`)]);
}

/* ----------------------------------------------------------------- waste */
function renderWaste() {
  const r = state.waste;
  if (!r) return;
  $("#waste-out").hidden = false;
  const rejected = r.status === "unknown" || r.status === "uncertain";
  $("#waste-label").textContent = rejected ? t("waste." + r.status) : wasteName(r.label);
  $("#waste-pct").hidden = rejected;
  $("#waste-pct").textContent = pct(r.top[0].prob, 1);
  $("#waste-tip").closest(".tip").hidden = rejected;
  showMsg("#waste-verdict-msg", rejected ? t(r.status === "unknown" ? "waste.rejectNote" : "waste.uncertainNote") : "", true);
  $("#waste-tip").textContent = r.tip[state.lang] || r.tip.en;
  renderBars($("#waste-bars"), r.top.map((x, i) => ({ name: wasteName(x.label), value: x.prob, text: pct(x.prob, 1), top: i === 0 })));
}
async function runWaste(file) {
  const prev = $("#waste-preview");
  prev.src = URL.createObjectURL(file); prev.hidden = false; $("#waste-drop").classList.add("has-img");
  const fd = new FormData(); fd.append("file", file);
  try { state.waste = await api("/api/waste", { method: "POST", body: fd }); showMsg("#waste-msg", ""); renderWaste(); }
  catch (e) { $("#waste-out").hidden = true; showMsg("#waste-msg", e.message); }
}

/* ----------------------------------------------------------------- segmentation */
const segLabel = (n) => n.replace(/\s*\(.*?(\d+px)\)/, " · $1");
function renderSegModels() {
  const list = (state.models && state.models.segmenters) || [];
  if (!state.seg.model || !list.includes(state.seg.model)) state.seg.model = list[0] || null;
  $("#seg-models").replaceChildren(...list.map((n) => h("button", {
    type: "button", "aria-pressed": String(n === state.seg.model),
    onclick: () => { state.seg.model = n; renderSegModels(); if (state.seg.file) runSeg(state.seg.file); },
  }, segLabel(n))));
}
function drawSeg() {
  const s = state.seg;
  if (!s.img) return;
  const c = $("#seg-canvas"), ctx = c.getContext("2d");
  c.width = s.img.naturalWidth; c.height = s.img.naturalHeight;
  ctx.drawImage(s.img, 0, 0);
  if (s.mask) { ctx.globalAlpha = $("#seg-opacity").value / 100; ctx.drawImage(s.mask, 0, 0, c.width, c.height); ctx.globalAlpha = 1; }
}
function renderSeg() {
  $("#seg-legend").replaceChildren(...["pet", "background", "border"].map((k) => h("span", {}, h("i", { style: `background:${SEG_COLORS[k]}` }), t("seg." + k))));
  const r = state.seg.res;
  if (!r) return;
  renderBars($("#seg-bars"), ["pet", "background", "border"].map((k) => ({ name: t("seg." + k), value: r.stats[k] / 100, text: `${r.stats[k]}%`, color: SEG_COLORS[k] })));
}
async function runSeg(file) {
  const s = state.seg;
  s.file = file;
  const fd = new FormData(); fd.append("file", file); if (s.model) fd.append("model", s.model);
  try {
    s.img = await loadImage(URL.createObjectURL(file));
    s.res = await api("/api/segment", { method: "POST", body: fd });
    s.mask = await loadImage(s.res.mask);
    showMsg("#seg-msg", ""); $("#seg-out").hidden = false; drawSeg(); renderSeg();
  } catch (e) { $("#seg-out").hidden = true; showMsg("#seg-msg", e && e.message ? e.message : t("err.generic")); }
}
async function runSegExample(name) {
  try { const b = await (await fetch(`/examples/${name}`)).blob(); await runSeg(new File([b], name, { type: b.type || "image/jpeg" })); }
  catch { showMsg("#seg-msg", t("err.net")); }
}
function renderSegExamples() {
  const ex = (state.models && state.models.examples) || [];
  $("#seg-examples").replaceChildren(...ex.map((n) => h("button", { type: "button", title: n, onclick: () => runSegExample(n) }, h("img", { src: `/examples/${n}`, alt: n, loading: "lazy" }))));
}

/* ----------------------------------------------------------------- sentiment */
function renderSentExamples() {
  $("#sent-examples").replaceChildren(...SENT_EXAMPLES.map(([k, text]) => h("button", {
    type: "button", class: "pill", onclick: () => { $("#sent-text").value = text; updateCount(); runSent(); },
  }, t(k))));
}
const updateCount = () => { $("#sent-count").textContent = `${$("#sent-text").value.length}/5000`; };
function renderSent() {
  const res = state.sent.results;
  if (!res) return;
  $("#sent-out").hidden = false;
  $("#sent-out").replaceChildren(...res.map((r) => {
    const name = t(r.model === "distilbert" ? "sent.distil" : "sent.lstm");
    if (r.error) return h("div", { class: "sentcard" }, h("div", { class: "who" }, h("b", {}, name)), h("div", { class: "msg" }, r.error));
    const neg = h("i", { class: "neg" }), pos = h("i", { class: "pos" });
    requestAnimationFrame(() => requestAnimationFrame(() => { neg.style.width = `${r.probs.negative * 100}%`; pos.style.width = `${r.probs.positive * 100}%`; }));
    return h("div", { class: "sentcard" },
      h("div", { class: "who" }, h("b", {}, name)),
      h("div", { class: "res " + r.label }, `${sentName(r.label)} · ${pct(r.probs[r.label], 1)}`),
      h("div", { class: "meter" }, neg, pos),
      h("div", { class: "nums" }, h("span", {}, `${sentName("negative")} ${pct(r.probs.negative, 1)}`), h("span", {}, `${sentName("positive")} ${pct(r.probs.positive, 1)}`)),
      h("div", { class: "meta" }, `${r.tokens_used} ${t("sent.tokens")}${r.truncated ? " · " + t("sent.cut") : ""}`));
  }));
}
async function runSent() {
  const text = $("#sent-text").value.trim();
  if (!text) { showMsg("#sent-msg", t("sent.empty")); return; }
  const models = (state.models && state.models.sentiment) || [];
  const btn = $("#sent-go");
  btn.disabled = true; btn.textContent = t("sent.busy");
  const one = (m) => api("/api/sentiment", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, model: m }) })
    .catch((e) => ({ model: m, error: e.message }));
  state.sent.results = await Promise.all(models.map(one));
  btn.disabled = false; btn.textContent = t("sent.go");
  showMsg("#sent-msg", models.includes("distilbert") ? "" : t("sent.nodistil"), true);
  renderSent();
}

/* ----------------------------------------------------------------- forecast */
function drawChart() {
  const { values, actual, preds } = state.fc;
  const root = $("#fc-chart");
  root.replaceChildren();
  if (!values) return;
  const W = 680, H = 270, L = 46, R = 52, T = 16, B = 34;
  const all = [...values, ...(preds ? [preds.rnn, preds.lstm, preds.persistence] : []), ...(actual != null ? [actual] : [])];
  const lo = Math.floor(Math.min(...all) - 1), hi = Math.ceil(Math.max(...all) + 1);
  const x = (i) => L + ((W - L - R) * i) / 31, y = (v) => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  for (let k = 0; k <= 4; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    root.append(svg("line", { class: "grid", x1: L, x2: W - R, y1: y(v), y2: y(v) }), svg("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }, `${v.toFixed(0)}°`));
  }
  root.append(svg("line", { x1: x(29), x2: x(29), y1: T, y2: H - B, stroke: "currentColor", "stroke-dasharray": "3 4", opacity: 0.3 }));
  root.append(svg("polyline", { class: "hist", points: values.map((v, i) => `${x(i)},${y(v)}`).join(" ") }));
  [[0, `-30 d`, "start", 0], [29, t("fc.today"), "end", 3], [30, t("fc.tomorrow"), "start", -3]].forEach(([i, s, anchor, dx]) => root.append(svg("text", { x: x(i) + dx, y: H - 10, "text-anchor": anchor }, s)));
  if (preds) {
    root.append(svg("line", { x1: x(29), y1: y(values[29]), x2: x(30), y2: y(preds.lstm), stroke: FC_COLORS.lstm, "stroke-dasharray": "3 3", opacity: 0.6 }));
    root.append(svg("circle", { cx: x(30) - 9, cy: y(preds.rnn), r: 5.5, fill: FC_COLORS.rnn }));
    root.append(svg("circle", { cx: x(30), cy: y(preds.lstm), r: 5.5, fill: FC_COLORS.lstm }));
    root.append(svg("circle", { cx: x(30) + 9, cy: y(preds.persistence), r: 5, fill: "none", stroke: FC_COLORS.base, "stroke-width": 2 }));
  }
  if (actual != null) root.append(svg("path", { d: `M${x(30) + 20},${y(actual) - 7} l7,7 l-7,7 l-7,-7 z`, fill: FC_COLORS.actual }));
}
function renderForecast() {
  const { values, actual, preds } = state.fc;
  $("#fc-legend").replaceChildren(...[["fc.history", "var(--ink)"], ["fc.rnn", FC_COLORS.rnn], ["fc.lstm", FC_COLORS.lstm], ["fc.base", FC_COLORS.base], ["fc.actual", FC_COLORS.actual]]
    .map(([k, c]) => h("span", {}, h("i", { style: `background:${c}` }), t(k))));
  drawChart();
  const tbl = $("#fc-table");
  if (!preds) { tbl.replaceChildren(); return; }
  const row = (k, v, col, err) => h("div", { class: "tr" }, h("span", { class: "name" }, h("i", { style: `background:${col}` }), t(k)), h("span", { class: "val" }, `${v.toFixed(1)} °C`), h("span", { class: "err" }, err));
  const err = (v) => (actual == null ? "" : `${v - actual >= 0 ? "+" : ""}${(v - actual).toFixed(1)}`);
  tbl.replaceChildren(
    h("div", { class: "tr head" }, h("span", {}, t("fc.model")), h("span", {}, t("fc.forecast")), h("span", {}, actual == null ? "" : t("fc.error"))),
    row("fc.rnn", preds.rnn, FC_COLORS.rnn, err(preds.rnn)), row("fc.lstm", preds.lstm, FC_COLORS.lstm, err(preds.lstm)),
    row("fc.base", preds.persistence, FC_COLORS.base, err(preds.persistence)), row("fc.mean7", preds.mean7, "#a58a3a", err(preds.mean7)),
    ...(actual != null ? [row("fc.actual", actual, FC_COLORS.actual, "")] : []));
}
async function runForecast(values, actual) {
  Object.assign(state.fc, { values, actual, preds: null });
  renderForecast();
  try {
    state.fc.preds = await api("/api/forecast", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ values }) });
    showMsg("#fc-msg", ""); renderForecast();
  } catch (e) { showMsg("#fc-msg", e.message); }
}
function renderFcSamples() {
  $("#fc-samples").replaceChildren(...state.fc.samples.map((s, i) => h("button", {
    type: "button", class: "pill", "aria-pressed": String(i === state.fc.idx),
    onclick: () => { state.fc.idx = i; renderFcSamples(); runForecast(s.values, s.actual); },
  }, `${s.start.slice(5)} → ${s.end.slice(5)} · ${s.start.slice(0, 4)}`)));
}
function parseCustom() {
  const nums = $("#fc-custom").value.split(/[\s,;]+/).filter(Boolean).map(Number);
  if (nums.length !== 30 || nums.some((n) => !Number.isFinite(n) || n < -60 || n > 60)) { showMsg("#fc-msg", t("fc.custom.bad")); return; }
  state.fc.idx = -1; renderFcSamples(); runForecast(nums, null);
}

/* ----------------------------------------------------------------- chrome: language, theme, tabs */
function renderStudy() {
  const box = $("#study-tables"), data = state.study;
  if (!data) return;
  box.replaceChildren();
  const score = (row) => `${(row.mean * 100).toFixed(3)}${row.std == null ? "" : ` ± ${(row.std * 100).toFixed(3)}`}`;
  function table(title, headers, rows, note) {
    const el = h("table", { class: "results study-table" },
      h("caption", {}, t(title)),
      h("thead", {}, h("tr", {}, headers.map((key) => h("th", { scope: "col" }, t(key))))),
      h("tbody", {}, rows.map((row) => h("tr", {}, row.map((value) => h("td", {}, String(value)))))));
    box.append(h("div", { class: "study-scroll", tabindex: "0" }, el));
    if (note) box.append(h("p", { class: "caption" }, t(note)));
  }
  table("study.waste", ["study.model", "study.score", "study.seeds"],
    ["baseline", "label_smoothing", "cutmix", "epochs20", "size192"].map((name) => [name, score(data.waste[name]), data.waste[name].n]));
  table("study.seg", ["study.model", "study.miou", "study.border", "study.seeds"],
    ["A_scratch", "B_pretrained_encoder", "C_recipe", "D_border_weight", "E_256px"].map((name) =>
      [name, score(data.segmentation[name]), score(data.border[name]), data.segmentation[name].n]), "study.segNote");
  table("study.imdb", ["study.model", "study.score", "study.seeds"],
    Object.entries(data.imdb).map(([name, row]) => [name, score(row), row.n]));
  table("study.compression", ["study.model", "study.score", "study.size"],
    Object.entries(data.compression).map(([name, row]) => [name, (row.test_acc * 100).toFixed(3), row.size_mb.toFixed(3)]), "study.compressionNote");
}

function applyI18n() {
  document.documentElement.lang = state.lang;
  document.title = state.lang === "vi" ? "AI Learning Lab · Học và thử AI" : "AI Learning Lab · Learn and try AI";
  $$("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  $$("[data-i18n-ph]").forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
  $$("[data-i18n-title]").forEach((el) => { el.title = t(el.dataset.i18nTitle); });
  $$("#lang button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === state.lang)));
  $$("[data-repo]").forEach((a) => { a.href = REPO + a.dataset.repo; });
  renderStudy(); renderStats(); renderSentExamples(); renderSegModels(); renderSeg(); renderWaste(); renderSent(); renderFcSamples(); renderForecast();
  if ($("#sent-go") && !$("#sent-go").disabled) $("#sent-go").textContent = t("sent.go");
}
async function init() {
  document.documentElement.dataset.theme = state.theme;
  $$("#lang button").forEach((b) => b.addEventListener("click", () => { state.lang = b.dataset.lang; store.set("lang", state.lang); applyI18n(); }));
  $("#theme").addEventListener("click", () => { state.theme = state.theme === "dark" ? "light" : "dark"; store.set("theme", state.theme); document.documentElement.dataset.theme = state.theme; });
  setupCamera();
  setupDrop($("#waste-drop"), $("#waste-file"), runWaste);
  setupDrop($("#seg-drop"), $("#seg-file"), runSeg);
  $("#seg-opacity").addEventListener("input", drawSeg);
  $("#sent-go").addEventListener("click", runSent);
  $("#sent-text").addEventListener("input", updateCount);
  $("#sent-text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runSent(); });
  $("#fc-custom-go").addEventListener("click", parseCustom);
  updateCount();
  api("/study-results.json").then((data) => { state.study = data; renderStudy(); })
    .catch(() => { $("#study-tables").textContent = t("study.error"); });
  applyI18n();

  try { state.models = await api("/api/models"); } catch (e) { ["#waste-msg", "#seg-msg", "#sent-msg", "#fc-msg"].forEach((id) => showMsg(id, e.message)); return; }
  const m = state.models;
  if (!m.waste) showMsg("#waste-msg", t("waste.off"), true);
  if (!m.segmenters.length) showMsg("#seg-msg", t("seg.off"), true);
  if (!m.sentiment.length) showMsg("#sent-msg", "No sentiment model is installed on this server.", true);
  renderSegExamples();
  if (m.forecast) {
    try { state.fc.samples = await api("/api/forecast/samples"); } catch { state.fc.samples = []; }
    if (state.fc.samples.length) { renderFcSamples(); runForecast(state.fc.samples[0].values, state.fc.samples[0].actual); }
  } else showMsg("#fc-msg", t("fc.off"), true);
  applyI18n();
}
init();
