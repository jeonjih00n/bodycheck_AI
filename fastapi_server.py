# fastapi_server.py
import os, time
from typing import List, Dict, Any
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO

APP_TITLE = "YOLOv11x-pose FastAPI Server"
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolo11n-pose.pt")  # yolo11n/s/x-pose.pt 등 교체 가능
YOLO_DEVICE  = os.getenv("YOLO_DEVICE",  None)               # None=auto('cuda:0' if available else 'cpu')
IMG_SIZE     = int(os.getenv("YOLO_IMG_SIZE", "640"))
CONF_THRESH  = float(os.getenv("YOLO_CONF", "0.25"))

app = FastAPI(title=APP_TITLE)

# 모델 로드 + 워밍업
try:
    model = YOLO(YOLO_WEIGHTS)
    _ = model.predict(source=np.zeros((64,64,3), dtype=np.uint8),
                      imgsz=64, conf=0.1, device=YOLO_DEVICE, verbose=False)
except Exception as e:
    raise RuntimeError(f"YOLO load failed: {e}")

@app.get("/")
def root():
    return {"ok": True, "model": YOLO_WEIGHTS, "device": YOLO_DEVICE or "auto"}

@app.post("/infer")
async def infer(
    image: UploadFile = File(..., description="JPEG frame"),
    frame_id: int = Form(...),
    cap_time: float = Form(...),   # 클라이언트 캡처 시각(perf_counter)
    imgsz: int = Form(IMG_SIZE),
    conf: float = Form(CONF_THRESH),
):
    t_recv = time.perf_counter()

    # 1) 디코딩
    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    t0 = time.perf_counter()
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"ok": False, "error": "imdecode failed"}, status_code=400)
    t1 = time.perf_counter()
    decode_ms = (t1 - t0) * 1000.0

    # 2) 추론
    t2 = time.perf_counter()
    results = model.predict(source=frame, imgsz=imgsz, conf=conf,
                            device=YOLO_DEVICE, verbose=False)
    t3 = time.perf_counter()
    res = results[0]
    spd = getattr(res, "speed", {}) if hasattr(res, "speed") else {}
    infer_ms = float(spd.get("inference", (t3 - t2) * 1000.0))
    preprocess_ms = float(spd.get("preprocess", 0.0))
    postprocess_ms = float(spd.get("postprocess", 0.0))

    # 3) 키포인트 수집
    persons: List[Dict[str, Any]] = []
    if hasattr(res, "keypoints") and res.keypoints is not None:
        xy = res.keypoints.xy   # (N, K, 2)
        cf = getattr(res.keypoints, "conf", None)  # (N, K) or None
        if xy is not None:
            for i in range(xy.shape[0]):
                persons.append({
                    "xy": xy[i].detach().cpu().numpy().tolist(),
                    "conf": (cf[i].detach().cpu().numpy().tolist() if cf is not None else None)
                })

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
            "t_recv": t_recv,
            "t_done": t_done
        }
    })
