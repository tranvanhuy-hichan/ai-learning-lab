"""Gradio web app: waste classification + pet segmentation.

Run:  python app.py   then open http://127.0.0.1:7860
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from vision_toolkit import inference as inf  # noqa: E402


def classify_waste(image):
    if image is None:
        raise gr.Error("Please upload an image first.")
    probs = inf.predict_waste(image)
    top = max(probs, key=probs.get)
    tip = f"**{top}** ({probs[top]:.0%})\n\n{inf.RECYCLING_TIPS[top]}\n\n*General guidance only; local rules vary.*"
    return probs, tip


def segment(image, model_name):
    if image is None:
        raise gr.Error("Please upload an image first.")
    if not model_name:
        raise gr.Error("No segmentation weights found in models/.")
    overlay, mask, stats = inf.segment_pet(image, model_name)
    legend = " | ".join(f"{k}: {v}%" for k, v in stats.items())
    return overlay, mask, legend


def build_demo() -> gr.Blocks:
    segmenters = inf.available_segmenters()
    examples = sorted(str(p) for p in (ROOT / "examples").glob("pet_*.jpg"))
    with gr.Blocks(title="AI Vision Toolkit") as demo:
        gr.Markdown(
            "# AI Vision Toolkit\n"
            "Two computer-vision models in one app: **waste classification** (EfficientNet-B0) "
            "and **pet segmentation** (U-Net). Models are trained on public datasets; see the README for limits."
        )
        with gr.Tab("Waste classifier"):
            gr.Markdown(
                "Upload a photo of a single waste item (12 classes: battery, biological, glass, cardboard, ...)."
            )
            with gr.Row():
                w_in = gr.Image(type="pil", label="Photo of a waste item")
                with gr.Column():
                    w_label = gr.Label(num_top_classes=5, label="Prediction")
                    w_tip = gr.Markdown()
            gr.Button("Classify", variant="primary").click(classify_waste, w_in, [w_label, w_tip])
        with gr.Tab("Pet segmentation"):
            gr.Markdown("Upload a photo of a cat or dog. Red = pet, blue = background, yellow = border.")
            with gr.Row():
                s_in = gr.Image(type="pil", label="Photo of a cat or dog")
                s_overlay = gr.Image(label="Overlay")
                s_mask = gr.Image(label="Predicted mask")
            s_model = gr.Dropdown(segmenters, value=segmenters[0] if segmenters else None, label="Model")
            s_stats = gr.Textbox(label="Pixel share per class")
            gr.Button("Segment", variant="primary").click(segment, [s_in, s_model], [s_overlay, s_mask, s_stats])
            if examples:
                gr.Examples(examples=examples, inputs=s_in, label="Examples (Oxford-IIIT Pet, CC BY-SA 4.0)")
    return demo


if __name__ == "__main__":
    build_demo().launch()
