# fastapi_server.py
import os, time
from typing import List, Dict, Any
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, Request, Query
from fastapi.responses import JSONResponse
from ultralytics import YOLO

APP_TITLE = "YOLOv11x-pose FastAPI Server"
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolo11n-pose.pt")
YOLO_DEVICE  = os.getenv("YOLO_DEVICE",  None)   # None -> auto
IMG_SIZE     = int(os.getenv("YOLO_IMG_SIZE", "640"))
CONF_THRESH  = float(os.getenv("YOLO_CONF", "0.25"))

# --- Optional: TurboJPEG (faster jpeg decode) ---
try:
    from turbojpeg import TurboJPEG, TJPF_BGR
    _jpeg = TurboJPEG()
    def decode_jpeg_bytes(image_bytes: bytes):
        t0 = time.perf_counter()
        frame = _jpeg.decode(image_bytes, pixel_format=TJPF_BGR)
        t1 = time.perf_counter()
        return frame, (t1 - t0) * 1000.0
    _DECODE_IMPL = "turbojpeg"
except Exception:
    _jpeg = None
    def decode_jpeg_bytes(image_bytes: bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        t0 = time.perf_counter()
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        t1 = time.perf_counter()
        return frame, (t1 - t0) * 1000.0
    _DECODE_IMPL = "opencv"

app = FastAPI(title=APP_TITLE)

# Load model & warmup
try:
    model = YOLO(YOLO_WEIGHTS)
    _ = model.predict(source=np.zeros((64,64,3), dtype=np.uint8),
                      imgsz=64, conf=0.1, device=YOLO_DEVICE, verbose=False)
except Exception as e:
    raise RuntimeError(f"YOLO load failed: {e}")

@app.get("/")
def root():
    return {"ok": True, "model": YOLO_WEIGHTS, "device": YOLO_DEVICE or "auto", "jpeg_decoder": _DECODE_IMPL}

def _run_infer(frame, imgsz: int, conf: float):
    t2 = time.perf_counter()
    results = model.predict(source=frame, imgsz=imgsz, conf=conf, device=YOLO_DEVICE, verbose=False)
    t3 = time.perf_counter()
    res = results[0]
    spd = getattr(res, "speed", {}) if hasattr(res, "speed") else {}
    infer_ms = float(spd.get("inference", (t3 - t2) * 1000.0))
    preprocess_ms = float(spd.get("preprocess", 0.0))
    postprocess_ms = float(spd.get("postprocess", 0.0))

    persons: List[Dict[str, Any]] = []
    if hasattr(res, "keypoints") and res.keypoints is not None and res.keypoints.xy is not None:
        xy = res.keypoints.xy
        cf = getattr(res.keypoints, "conf", None)
        for i in range(xy.shape[0]):
            persons.append({
                "xy": xy[i].detach().cpu().numpy().tolist(),
                "conf": (cf[i].detach().cpu().numpy().tolist() if cf is not None else None),
            })
    return persons, infer_ms, preprocess_ms, postprocess_ms

# 기존 multipart/form-data 업로드 (호환 유지)
@app.post("/infer")
async def infer(
    image: UploadFile = File(..., description="JPEG frame"),
    frame_id: int = Form(...),
    cap_time: float = Form(...),
    imgsz: int = Form(IMG_SIZE),
    conf: float = Form(CONF_THRESH),
):
    t_recv = time.perf_counter()
    image_bytes = await image.read()
    frame, decode_ms = decode_jpeg_bytes(image_bytes)
    if frame is None:
        return JSONResponse({"ok": False, "error": "imdecode failed"}, status_code=400)

    persons, infer_ms, preprocess_ms, postprocess_ms = _run_infer(frame, imgsz, conf)
    t_done = time.perf_counter()
    return JSONResponse({
        "ok": True,
        "frame_id": frame_id,
        "cap_time": cap_time,
        "persons": persons,
        "times": {
            "decode_ms": decode_ms,
            "preprocess_ms": preprocess_ms,
            "infer_ms": infer_ms,
            "postprocess_ms": postprocess_ms,
            "server_total_ms": (t_done - t_recv) * 1000.0,
        },
    })

# RAW 바이너리 업로드 (application/octet-stream) - 폼 파싱 오버헤드 제거
@app.post("/infer_bytes")
async def infer_bytes(
    request: Request,
    frame_id: int = Query(...),
    cap_time: float = Query(...),
    imgsz: int = Query(IMG_SIZE),
    conf: float = Query(CONF_THRESH),
):
    t_recv = time.perf_counter()
    image_bytes = await request.body()
    frame, decode_ms = decode_jpeg_bytes(image_bytes)
    if frame is None:
        return JSONResponse({"ok": False, "error": "imdecode failed"}, status_code=400)

    persons, infer_ms, preprocess_ms, postprocess_ms = _run_infer(frame, imgsz, conf)
    t_done = time.perf_counter()
    return JSONResponse({
        "ok": True,
        "frame_id": frame_id,
        "cap_time": cap_time,
        "persons": persons,
        "times": {
            "decode_ms": decode_ms,
            "preprocess_ms": preprocess_ms,
            "infer_ms": infer_ms,
            "postprocess_ms": postprocess_ms,
            "server_total_ms": (t_done - t_recv) * 1000.0,
        },
    })
