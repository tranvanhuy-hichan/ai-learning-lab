import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from labweb import services
from labweb.main import EXAMPLES, app
from PIL import Image

client = TestClient(app)


def _png(w=120, h=90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_models():
    assert client.get("/api/health").json()["status"] == "ok"
    body = client.get("/api/models").json()
    assert {"waste", "segmenters", "sentiment", "forecast", "metrics", "examples"} <= set(body)
    assert "lstm" in body["sentiment"] and body["forecast"] is True


def test_front_end_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "AI Learning Lab" in r.text
    assert client.get("/app.js").status_code == 200 and client.get("/style.css").status_code == 200


def test_waste_returns_top5_and_bilingual_tip():
    r = client.post("/api/waste", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert len(body["top"]) == 5 and body["top"][0]["label"] == body["label"]
    assert set(body["tip"]) == {"en", "vi"} and body["tip"]["vi"] != body["tip"]["en"]


def test_vietnamese_tip_exists_for_every_class():
    from vision_toolkit import inference as vision

    assert set(services.TIPS_VI) == set(vision.RECYCLING_TIPS)


@pytest.mark.parametrize("payload,status", [(b"not an image", 400), (b"", 400)])
def test_bad_uploads_are_rejected(payload, status):
    r = client.post("/api/waste", files={"file": ("x.png", payload, "image/png")})
    assert r.status_code == status


def test_oversized_upload_is_rejected():
    r = client.post("/api/waste", files={"file": ("x.bin", b"0" * (8 * 1024 * 1024 + 10), "image/png")})
    assert r.status_code == 413


@pytest.mark.skipif(not EXAMPLES.exists(), reason="example images missing")
def test_segment_returns_mask_at_input_size():
    example = sorted(EXAMPLES.glob("pet_*.jpg"))[0]
    size = Image.open(example).size
    r = client.post("/api/segment", files={"file": (example.name, example.read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["mask"].startswith("data:image/png;base64,") and body["size"] == list(size)
    assert abs(sum(body["stats"].values()) - 100) < 0.5
    assert client.get(f"/examples/{example.name}").status_code == 200


def test_sentiment_lstm_gets_clear_cases_right():
    pos = client.post(
        "/api/sentiment", json={"text": "A wonderful, moving film with brilliant acting.", "model": "lstm"}
    ).json()
    neg = client.post(
        "/api/sentiment", json={"text": "Total waste of time, awful acting and a boring plot.", "model": "lstm"}
    ).json()
    assert pos["label"] == "positive" and neg["label"] == "negative"
    assert abs(sum(pos["probs"].values()) - 1) < 1e-3


@pytest.mark.parametrize(
    "body", [{"text": "", "model": "lstm"}, {"text": "x" * 5001, "model": "lstm"}, {"text": "ok", "model": "gpt"}]
)
def test_sentiment_validates_input(body):
    assert client.post("/api/sentiment", json=body).status_code == 422


def test_forecast_samples_and_prediction():
    samples = client.get("/api/forecast/samples").json()
    assert len(samples) >= 3 and all(len(s["values"]) == 30 for s in samples)
    r = client.post("/api/forecast", json={"values": samples[0]["values"]})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"rnn", "lstm", "persistence", "mean7"}
    assert all(-10 < v < 40 for v in body.values())


@pytest.mark.parametrize("values", [[10.0] * 29, [10.0] * 31, [100.0] * 30, ["a"] * 30])
def test_forecast_validates_input(values):
    assert client.post("/api/forecast", json={"values": values}).status_code == 422


def test_metrics_file_is_valid_json():
    json.loads((services.MODELS / "metrics.json").read_text())


def test_waste_reports_a_verdict_and_a_heat_map():
    body = client.post("/api/waste", files={"file": ("x.png", _png(), "image/png")}).json()
    assert body["status"] in {"ok", "uncertain", "unknown"} and body["calibrated"] is True
    assert body["cam"].startswith("data:image/png;base64,")
    assert abs(sum(t["prob"] for t in body["top"]) - 1) < 0.2  # top-5 of a distribution


def test_waste_heat_map_can_be_switched_off():
    body = client.post("/api/waste?explain=false", files={"file": ("x.png", _png(), "image/png")}).json()
    assert body["cam"] is None


@pytest.mark.skipif(not EXAMPLES.exists(), reason="example images missing")
def test_a_cat_photo_is_not_confidently_called_waste():
    example = sorted(EXAMPLES.glob("pet_*.jpg"))[0]
    body = client.post("/api/waste", files={"file": (example.name, example.read_bytes(), "image/jpeg")}).json()
    assert body["status"] in {"unknown", "uncertain"}  # the classifier must not claim a pet is plain waste


def test_sentiment_returns_word_importance_that_points_the_right_way():
    r = client.post(
        "/api/sentiment", json={"text": "A wonderful, moving film with brilliant acting.", "model": "lstm"}
    ).json()
    words = {w["t"]: w["s"] for w in r["words"]}
    assert "wonderful" in words and words["wonderful"] > 0  # removing a praising word lowers "positive"
    assert r["words"][0]["t"] == "a"
