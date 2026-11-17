"""Build the report and web tables from saved measurements, without retraining."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"


def read(name):
    return json.loads((RESULTS / name).read_text())


def aggregate(runs, metric, nested=None):
    groups = defaultdict(list)
    for key, run in runs.items():
        value = run[metric] if nested is None else run[metric][nested]
        groups[key.split("/")[0]].append(value)
    return {name: {"mean": statistics.mean(values),
                   "std": statistics.stdev(values) if len(values) > 1 else None,
                   "n": len(values)} for name, values in groups.items()}


def cell(row):
    spread = f" ± {row['std'] * 100:.3f}" if row['std'] is not None else " (single run)"
    return f"{row['mean'] * 100:.3f}{spread}"


def main():
    ab, seg, comp = read("ablation.json"), read("seg_ablation.json"), read("compression.json")
    summary = {
        "waste": aggregate(ab, "test_acc"),
        "students": aggregate(comp["students"], "test_acc"),
        "segmentation": aggregate(seg, "mIoU"),
        "border": aggregate(seg, "IoU", "border"),
        "compression": comp["post"], "teacher": comp["teacher"],
        "imdb": {"BiLSTM": {"mean": 0.8766, "n": 1, "std": None},
                 "TF-IDF + Logistic Regression": {"mean": read("tfidf_imdb.json")["logreg_1-2gram"]["test_acc"], "n": 1, "std": None},
                 "TF-IDF + Linear SVM": {"mean": read("tfidf_imdb.json")["linear_svm_1-2gram"]["test_acc"], "n": 1, "std": None},
                 "DistilBERT": {"mean": 0.91592, "n": 1, "std": None}},
    }
    # Read sequence measurements too, so rerunning reflects their actual source files.
    for label, filename in [("BiLSTM", "lstm"), ("DistilBERT", "distilbert")]:
        path = HERE.parent / "sequence-modeling" / "outputs" / f"{filename}_imdb_results.json"
        summary["imdb"][label]["mean"] = json.loads(path.read_text())["test_acc"]
    text = ["# Downloaded Kaggle results", "",
            "Source: `hichandut/model-study`; downloaded after CLI reported COMPLETE on 2026-11-17 (Asia/Bangkok).",
            "Raw files: [ablation](ablation.json), [compression](compression.json), [segmentation](seg_ablation.json).",
            "Regenerate: `python experiments/summarize_results.py` from `projects/model-study`.", "",
            "All spreads below are sample standard deviations across seeds, in percentage points; they are not confidence intervals.",
            "Test data was not used to select the teacher: it is `epochs20`, chosen by mean validation accuracy among 128 px variants.", "",
            "## Waste classification", "", "| Variant | Test accuracy (%) | Seeds |", "|---|---:|---:|"]
    for name in ("baseline", "label_smoothing", "cutmix", "epochs20", "size192"):
        row = summary["waste"][name]; text.append(f"| {name} | {cell(row)} | {row['n']} |")
    text += ["", "192 px has the highest observed mean test accuracy; the gap from baseline is "
             f"{100 * (summary['waste']['size192']['mean'] - summary['waste']['baseline']['mean']):.3f} percentage points.",
             "These variants change one baseline setting each; they do not test combinations of changes.",
             "", "## Small students", "", "| Student | Test accuracy (%) | Seeds |", "|---|---:|---:|"]
    for name, row in summary["students"].items(): text.append(f"| {name} | {cell(row)} | {row['n']} |")
    text += ["", "## Post-training compression (one teacher / one run per method)", "",
             "| Method | Test accuracy (%) | File MB | gzip MB |", "|---|---:|---:|---:|"]
    for name, row in comp["post"].items():
        text.append(f"| {name} | {row['test_acc']*100:.3f} | {row['size_mb']:.3f} | {format(row['gzip_mb'], '.3f') if 'gzip_mb' in row else '—'} |")
    text += ["", "An em dash means gzip size was not measured. ONNX INT8 loses substantial accuracy here; it should not replace the current model.",
             "Pruning leaves dense tensor shapes and ordinary checkpoint sizes unchanged. Smaller gzip files do not establish faster inference.",
             "CPU latency has not been measured for these downloaded models.", "", "## Segmentation: historical recipe", "",
             "| Step | mIoU (%) | Border IoU (%) | Seeds | Epochs | Batch |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("A_scratch", "B_pretrained_encoder", "C_recipe", "D_border_weight", "E_256px"):
        row=summary["segmentation"][name]; cfg=next(r for k,r in seg.items() if k.split('/')[0]==name)
        text.append(f"| {name} | {cell(row)} | {cell(summary['border'][name])} | {row['n']} | {cfg['epochs']} | {cfg['batch']} |")
    text += ["", "A→B replaces the scratch encoder with pretrained ResNet18; model architecture and parameter count also change.",
             "B→C changes both learning rate (0.002→0.001) and epochs (30→25), so this is a recipe comparison.",
             "C→D changes border loss weight to 3: inspect both mIoU and border IoU rather than assuming improvement.",
             "D→E changes resolution (128→256) and batch size (32→16); E has only seed 0. Evaluation resolution changes too.",
             "The downloaded measurements must not be described as the corrected 30-epoch / batch-16 ablation now in the script.",
             "A corrected rerun is still pending; no downloaded segmentation checkpoint has replaced the web model.", "",
             "## IMDB (local results, one run)", "", "| Model | Test accuracy (%) |", "|---|---:|"]
    for name,row in summary["imdb"].items(): text.append(f"| {name} | {row['mean']*100:.3f} |")
    (RESULTS / "REPORT.md").write_text("\n".join(text)+"\n")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (HERE.parent / "lab-web" / "static" / "study-results.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("Generated results/REPORT.md, results/summary.json and web study-results.json")


if __name__ == "__main__":
    main()
