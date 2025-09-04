# client_stream_http.py
import argparse, time, threading
from typing import List, Optional, Tuple
from queue import Queue, Full, Empty
import requests, cv2, numpy as np

COCO_SKELETON: List[Tuple[int,int]] = [
    (5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4),
]

def draw_pose(img, persons_xy, persons_conf, min_conf=0.25, r=3, t=2):
    for p_idx, kpts in enumerate(persons_xy or []):
        confs = None if persons_conf is None else persons_conf[p_idx]
        K = len(kpts)
        # 선
        for a,b in COCO_SKELETON:
            if a < K and b < K:
                x1,y1 = kpts[a]; x2,y2 = kpts[b]
                c1 = 1.0 if confs is None else float(confs[a])
                c2 = 1.0 if confs is None else float(confs[b])
                if c1 >= min_conf and c2 >= min_conf:
                    cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),t)
        # 점
        for j in range(K):
            x,y = kpts[j]
            c = 1.0 if confs is None else float(confs[j])
            if c >= min_conf:
                cv2.circle(img,(int(x),int(y)),r,(0,255,255),-1)

class ResultStore:
    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.res = None
    def set(self, d):
        with self.lock: self.res = d
    def get(self):
        with self.lock: return None if self.res is None else self.res.copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--target_fps", type=float, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--queue_size", type=int, default=2)
    ap.add_argument("--jpeg_q", type=int, default=85)
    ap.add_argument("--min_kpt_conf", type=float, default=0.25)
    args = ap.parse_args()

    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source)) if is_cam else cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("[오류] 소스 열기 실패"); return
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    sess = requests.Session()
    q = Queue(maxsize=args.queue_size)
    store = ResultStore()
    stop = False

    def worker():
        while not stop:
            try:
                item = q.get(timeout=0.1)
            except Empty:
                continue
            frame_id, frame, cap_time = item
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_q])
            if not ok:
                q.task_done(); continue

            t_req0 = time.perf_counter()
            files = {"image": ("frame.jpg", buf.tobytes(), "image/jpeg")}
            data = {
                "frame_id": str(frame_id),
                "cap_time": str(cap_time),
                "imgsz": str(args.imgsz),
                "conf": str(args.conf),
            }
            try:
                r = sess.post(args.server + "/infer", files=files, data=data, timeout=10)
                t_req1 = time.perf_counter()
                if r.ok:
                    resp = r.json()
                    server_total = float(resp["times"]["server_total_ms"])
                    rtt_ms = (t_req1 - t_req0)*1000.0
                    net_ms = max(0.0, rtt_ms - server_total)  # 대충의 네트워크 왕복 추정
                    store.set({
                        "frame_id": resp["frame_id"],
                        "cap_time": float(resp["cap_time"]),
                        "persons": resp["persons"],
                        "infer_ms": float(resp["times"]["infer_ms"]),
                        "server_total_ms": server_total,
                        "net_ms": net_ms,
                    })
            except Exception:
                pass
            q.task_done()

    th = threading.Thread(target=worker, daemon=True); th.start()

    period = 1.0/max(1e-6,args.target_fps)
    frame_id = 0
    cv2.namedWindow("Client Live (sent) + delayed pose", cv2.WINDOW_NORMAL)
    print("[도움말] q 종료, p 일시정지/재개")

    paused = False
    last_disp = time.perf_counter()
    while True:
        loop0 = time.perf_counter()
        if not paused:
            ok, frame = cap.read()
            if not ok: break
            cap_time = time.perf_counter()

            try:
                q.put((frame_id, frame.copy(), cap_time), timeout=0.005)
            except Full:
                pass

            display = frame.copy()
            res = store.get()
            if res is not None:
                age_ms = (time.perf_counter() - res["cap_time"])*1000.0
                persons_xy = [np.array(p["xy"], dtype=np.float32) for p in (res["persons"] or [])]
                persons_conf = [np.array(p["conf"], dtype=np.float32) if p["conf"] is not None else None
                                for p in (res["persons"] or [])] if res["persons"] else None
                draw_pose(display, persons_xy, persons_conf, args.min_kpt_conf)
                lines = [
                    f"Sent frame id: {frame_id}",
                    f"Latest result from frame id: {res['frame_id']}  (age: {age_ms:6.1f} ms)",
                    f"Server infer: {res['infer_ms']:5.1f} ms | Server total: {res['server_total_ms']:5.1f} ms | Net≈{res['net_ms']:5.1f} ms"
                ]
                for i,t in enumerate(lines):
                    cv2.putText(display,t,(12,30+24*i),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)

            cv2.imshow("Client Live (sent) + delayed pose", display)
            frame_id += 1

        key = cv2.waitKey(1)&0xFF
        if key==ord('q') or key==27: break
        elif key==ord('p'): paused = not paused

        sleep = period - (time.perf_counter()-loop0)
        if sleep>0: time.sleep(sleep)

    # 종료
    stop = True
    try: th.join(timeout=1.0)
    except: pass
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
