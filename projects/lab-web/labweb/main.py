"""FastAPI app: JSON API for every model, plus the static front-end."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator

from vision_toolkit import inference as vision

from . import __version__, services

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
EXAMPLES = ROOT.parent / "vision-toolkit" / "examples"
MAX_UPLOAD = 8 * 1024 * 1024
MAX_SIDE = 2048

app = FastAPI(title="AI Learning Lab", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")


def _read_image(file: UploadFile) -> Image.Image:
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"Image is larger than {MAX_UPLOAD // 1024 // 1024} MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, "Could not read the file as an image") from exc
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def _png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/models")
def models() -> dict:
    return {
        "waste": vision.WASTE_WEIGHTS.exists(),
        "segmenters": vision.available_segmenters(),
        "sentiment": services.sentiment_models(),
        "forecast": services.forecast_available(),
        "metrics": services.metrics(),
        "examples": sorted(p.name for p in EXAMPLES.glob("pet_*.jpg")) if EXAMPLES.exists() else [],
    }


@app.post("/api/waste")
def waste(file: UploadFile = File(...), explain: bool = True) -> dict:
    if not vision.WASTE_WEIGHTS.exists():
        raise HTTPException(503, "Waste classifier weights are not installed")
    out = services.classify_waste(_read_image(file), explain=explain)
    out["cam"] = _png_data_url(out["cam"]) if out["cam"] is not None else None
    return out


@app.post("/api/segment")
def segment(file: UploadFile = File(...), model: str | None = Form(None)) -> dict:
    img = _read_image(file)
    try:
        out = services.segment(img, model)
    except LookupError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"model": out["model"], "mask": _png_data_url(out["mask"]), "stats": out["stats"], "size": list(img.size)}


class SentimentIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    model: Literal["lstm", "distilbert"] = "lstm"


@app.post("/api/sentiment")
def sentiment(body: SentimentIn) -> dict:
    if body.model not in services.sentiment_models():
        raise HTTPException(503, f"Model '{body.model}' is not available on this server")
    if not body.text.strip():
        raise HTTPException(422, "Text is empty")
    return services.sentiment(body.text, body.model)


class ForecastIn(BaseModel):
    values: list[float] = Field(min_length=30, max_length=30)

    @field_validator("values")
    @classmethod
    def plausible(cls, v: list[float]) -> list[float]:
        if any(not -60 <= x <= 60 for x in v):
            raise ValueError("temperatures must be between -60 and 60 °C")
        return v


@app.get("/api/forecast/samples")
def forecast_samples() -> list[dict]:
    if not services.forecast_available():
        raise HTTPException(503, "Forecast models are not installed")
    return services.forecast_samples()


@app.post("/api/forecast")
def forecast(body: ForecastIn) -> dict:
    if not services.forecast_available():
        raise HTTPException(503, "Forecast models are not installed")
    return services.forecast(body.values)


if EXAMPLES.exists():
    app.mount("/examples", StaticFiles(directory=EXAMPLES), name="examples")
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
