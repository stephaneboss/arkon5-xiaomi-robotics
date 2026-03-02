"""ARKON-5 | Xiaomi Robotics-0 VLA Inference Server
S25 Lumiere Trading Pipeline - FastAPI endpoint for MEXC signals
"""
import asyncio
import base64
import logging
import re
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arkon5")

model = None
processor = None
model_lock = asyncio.Lock()
stats = {"inferences": 0, "total_latency_ms": 0.0, "errors": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ARKON-5 server starting - model loads on first request (lazy)")
    yield
    logger.info("ARKON-5 server shutting down")


app = FastAPI(
    title="ARKON-5 Xiaomi Robotics-0",
    description="S25 Lumiere VLA inference server for crypto trading signals",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_model():
    global model, processor
    if model is not None:
        return model, processor
    async with model_lock:
        if model is not None:
            return model, processor
        logger.info("Loading Xiaomi Robotics-0 model...")
        t0 = time.time()
        try:
            from xiaomi_model import XiaomiRoboticsModel, XiaomiRoboticsProcessor
            model = XiaomiRoboticsModel.from_pretrained(
                "/app/xiaomi_model", torch_dtype=torch.bfloat16
            ).cuda().eval()
            processor = XiaomiRoboticsProcessor.from_pretrained("/app/xiaomi_model")
        except Exception:
            logger.warning("Native loader failed, falling back to AutoModel")
            from transformers import AutoModel, AutoProcessor
            model = AutoModel.from_pretrained(
                "/app/xiaomi_model",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).cuda().eval()
            processor = AutoProcessor.from_pretrained(
                "/app/xiaomi_model", trust_remote_code=True
            )
        elapsed = time.time() - t0
        logger.info(f"Model loaded in {elapsed:.1f}s")
        return model, processor


class AnalyzeRequest(BaseModel):
    image_b64: str
    command: str = "Analyze this chart and describe what you see"


class TradingSignalRequest(BaseModel):
    image_b64: str
    pair: str = "BTC/USDT"
    timeframe: str = "1h"
    context: Optional[str] = None


class TradingSignalResponse(BaseModel):
    action: str
    conf: float
    tp: Optional[float]
    sl: Optional[float]
    reason: str
    model: str
    latency_ms: float


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/info")
async def info():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        }
    return {
        "model": "xiaomi-robotics-0",
        "version": "1.0.0",
        "pipeline": "ARKON-5 S25 Lumiere",
        "model_loaded": model is not None,
        "gpu": gpu_info,
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    t0 = time.time()
    try:
        m, p = await asyncio.wait_for(get_model(), timeout=120)
        img = Image.open(BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
        inputs = p(images=img, text=req.command, return_tensors="pt")
        inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = m.generate(**inputs, max_new_tokens=256)
        result = p.decode(out[0], skip_special_tokens=True)
        latency = (time.time() - t0) * 1000
        stats["inferences"] += 1
        stats["total_latency_ms"] += latency
        return {"result": result, "latency_ms": round(latency, 1)}
    except Exception as e:
        stats["errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


def _parse_signal(raw: str):
    action, conf, tp, sl, reason = "HOLD", 0.5, None, None, raw.strip()
    try:
        if m := re.search(r"ACTION:\s*(BUY|SELL|HOLD)", raw, re.I):
            action = m.group(1).upper()
        if m := re.search(r"CONFIDENCE:\s*([\d.]+)", raw, re.I):
            conf = min(1.0, max(0.0, float(m.group(1))))
        if m := re.search(r"TP:\s*([\d.]+)", raw, re.I):
            tp = float(m.group(1))
        if m := re.search(r"SL:\s*([\d.]+)", raw, re.I):
            sl = float(m.group(1))
        if m := re.search(r"REASON:\s*(.+)", raw, re.I):
            reason = m.group(1).strip()
    except Exception:
        pass
    return action, conf, tp, sl, reason


@app.post("/trading-signal", response_model=TradingSignalResponse)
async def trading_signal(req: TradingSignalRequest):
    t0 = time.time()
    prompt = (
        f"You are a crypto trading AI. Analyze this {req.timeframe} chart for {req.pair}. "
        f"Context: {req.context if req.context else 'none'} "
        "Respond ONLY in this format: "
        "ACTION: BUY|SELL|HOLD | CONFIDENCE: 0.XX | TP: price_or_null | SL: price_or_null | REASON: one sentence"
    )
    try:
        m, p = await asyncio.wait_for(get_model(), timeout=120)
        img = Image.open(BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
        inputs = p(images=img, text=prompt, return_tensors="pt")
        inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = m.generate(**inputs, max_new_tokens=128)
        raw = p.decode(out[0], skip_special_tokens=True)
        action, conf, tp, sl, reason = _parse_signal(raw)
        latency = (time.time() - t0) * 1000
        stats["inferences"] += 1
        stats["total_latency_ms"] += latency
        return TradingSignalResponse(
            action=action, conf=conf, tp=tp, sl=sl,
            reason=reason, model="xiaomi-robotics-0",
            latency_ms=round(latency, 1),
        )
    except Exception as e:
        stats["errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    avg_lat = stats["total_latency_ms"] / stats["inferences"] if stats["inferences"] > 0 else 0
    gpu_util = 0
    if torch.cuda.is_available():
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            gpu_util = int(r.stdout.strip())
        except Exception:
            pass
    return {
        "inferences": stats["inferences"],
        "errors": stats["errors"],
        "avg_latency_ms": round(avg_lat, 1),
        "gpu_utilization_pct": gpu_util,
    }
