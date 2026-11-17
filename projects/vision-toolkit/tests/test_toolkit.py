import numpy as np
import pytest
import torch
from PIL import Image

from vision_toolkit import inference as inf
from vision_toolkit.models import ResNetUNet, UNet, build_waste_classifier


def _img(w=200, h=150):
    return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))


@pytest.mark.parametrize("model,size", [(UNet(3, 32), 128), (ResNetUNet(3), 128), (ResNetUNet(3), 256)])
def test_segmentation_models_keep_spatial_size(model, size):
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, size, size)).shape == (1, 3, size, size)


def test_waste_classifier_output_shape():
    m = build_waste_classifier(12).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 3, 128, 128)).shape == (2, 12)


def test_every_waste_label_has_a_tip():
    assert set(inf.waste_labels()) == set(inf.RECYCLING_TIPS)


@pytest.mark.skipif(not inf.WASTE_WEIGHTS.exists(), reason="waste weights missing")
def test_predict_waste_returns_probability_distribution():
    probs = inf.predict_waste(_img())
    assert set(probs) == set(inf.waste_labels())
    assert abs(sum(probs.values()) - 1) < 1e-4


@pytest.mark.skipif(not inf.available_segmenters(), reason="segmentation weights missing")
@pytest.mark.parametrize("w,h", [(200, 150), (333, 500)])
def test_segment_pet_returns_original_resolution(w, h):
    overlay, mask, stats = inf.segment_pet(_img(w, h), inf.available_segmenters()[0])
    assert overlay.size == mask.size == (w, h)
    assert abs(sum(stats.values()) - 100) < 0.5


def test_gradcam_shape_and_range():
    from vision_toolkit.explain import gradcam, heat, overlay

    m = build_waste_classifier(12).eval()
    x = torch.randn(2, 3, 128, 128)
    cam, logits = gradcam(m, x)
    assert cam.shape == (2, 128, 128) and logits.shape == (2, 12)
    assert float(cam.min()) >= 0 and float(cam.max()) <= 1.0001
    assert heat(cam[0].numpy()).shape == (128, 128, 3)
    assert overlay(_img(200, 150), cam[0].numpy()).size == (200, 150)


@pytest.mark.skipif(not inf.WASTE_WEIGHTS.exists(), reason="waste weights missing")
def test_analyze_waste_returns_a_verdict():
    out = inf.analyze_waste(_img(), explain=True)
    assert out["status"] in {"ok", "uncertain", "unknown"}
    assert abs(sum(out["probs"].values()) - 1) < 1e-4
    assert out["cam"].shape == (128, 128)
