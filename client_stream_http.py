# client_stream_http.py
import argparse, time, threading
from typing import List, Tuple
from queue import Queue, Full, Empty
import requests, cv2, numpy as np
from collections import deque

COCO_SKELETON: List[Tuple[int,int]] = [
    (5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4),
]

def draw_pose(img, persons_xy, persons_conf, min_conf=0.25, r=3, t=2):
    for p_idx, kpts in enumerate(persons_xy or []):
        confs = None if persons_conf is None else persons_conf[p_idx]
        K = len(kpts)
        for a,b in COCO_SKELETON:
            if a < K and b < K:
                x1,y1 = kpts[a]; x2,y2 = kpts[b]
                c1 = 1.0 if confs is None else float(confs[a])
                c2 = 1.0 if confs is None else float(confs[b])
                if c1 >= min_conf and c2 >= min_conf:
                    cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),t)
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
    ap.add_argument("--jpeg_q", type=int, default=75)
    ap.add_argument("--min_kpt_conf", type=float, default=0.25)
    ap.add_argument("--transport", choices=["form","raw"], default="raw")
    ap.add_argument("--send_long", type=int, default=640)
    ap.add_argument("--use_turbojpeg", action="store_true")
    ap.add_argument("--log_csv", type=str, default=None)
    ap.add_argument("--avg_window", type=int, default=120, help="이동평균 윈도우(프레임 수)")
    args = ap.parse_args()

    # Optional TurboJPEG
    jpeg = None
    TJPF_BGR_CONST = None
    if args.use_turbojpeg:
        try:
            from turbojpeg import TurboJPEG, TJPF_BGR
            jpeg = TurboJPEG()
            TJPF_BGR_CONST = TJPF_BGR
            print("[info] Using TurboJPEG for encode")
        except Exception as e:
            print(f"[warn] TurboJPEG unavailable ({e}); fallback to OpenCV.")

    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source)) if is_cam else cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("[오류] 소스 열기 실패"); return
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    sess = requests.Session()
    q = Queue(maxsize=args.queue_size)
    store = ResultStore()
    stop = False

    # ---- 간단 시간 동기화(오프셋 추정: server_time - client_time) ----
    offset_lock = threading.Lock()
    offset = 0.0  # seconds (server ≈ client + offset)
    def sync_once():
        nonlocal offset
        try:
            t0 = time.time()
            r = sess.get(args.server + "/now", timeout=2)
            t1 = time.time()
            if r.ok:
                srv = float(r.json()["server_time"])
                # Cristian's algorithm
                curr = t0 + (t1 - t0)/2.0
                est = srv - curr
                # EMA
                with offset_lock:
                    alpha = 0.3
                    offset = alpha * est + (1 - alpha) * offset
        except Exception:
            pass

    # 초기 3회 샘플링 후 주기적 갱신
    for _ in range(3): sync_once()
    def timesync_worker():
        while not stop:
            sync_once()
            time.sleep(2.0)
    threading.Thread(target=timesync_worker, daemon=True).start()

    # ---- CSV ----
    csv_fp = None
    if args.log_csv:
        csv_fp = open(args.log_csv, "w", encoding="utf-8")
        csv_fp.write("frame_id,age_ms,send_ms,return_ms,server_infer_ms,server_total_ms,decode_ms,rtt_ms,disp_fps\n")

    # ---- 이동평균용 버퍼 ----
    win = args.avg_window
    buf_send = deque(maxlen=win)
    buf_ret  = deque(maxlen=win)
    buf_inf  = deque(maxlen=win)
    buf_tot  = deque(maxlen=win)

    # ---- 전송 워커 ----
    def worker():
        while not stop:
            try:
                item = q.get(timeout=0.1)
            except Empty:
                continue
            frame_id, frame_small, cap_time, scale = item

            # JPEG encode
            t_enc0 = time.perf_counter()
            if jpeg is not None:
                try:
                    payload_bytes = jpeg.encode(frame_small, quality=args.jpeg_q, pixel_format=TJPF_BGR_CONST)
                except Exception:
                    ok, buf = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_q])
                    if not ok: q.task_done(); continue
                    payload_bytes = buf.tobytes()
            else:
                ok, buf = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_q])
                if not ok: q.task_done(); continue
                payload_bytes = buf.tobytes()
            t_enc1 = time.perf_counter()
            encode_ms = (t_enc1 - t_enc0) * 1000.0

            # HTTP request
            t_req0_c = time.time()  # client wall clock
            try:
                if args.transport == "form":
                    files = {"image": ("frame.jpg", payload_bytes, "image/jpeg")}
                    data = {"frame_id": str(frame_id), "cap_time": str(cap_time),
                            "imgsz": str(args.imgsz), "conf": str(args.conf)}
                    r = sess.post(args.server + "/infer", files=files, data=data, timeout=10)
                else:
                    params = {"frame_id": frame_id, "cap_time": cap_time,
                              "imgsz": args.imgsz, "conf": args.conf}
                    r = sess.post(args.server + "/infer_bytes",
                                  params=params, data=payload_bytes,
                                  headers={"Content-Type": "application/octet-stream"},
                                  timeout=10)
                t_req1_c = time.time()
                if r.ok:
                    resp = r.json()
                    times = resp["times"]
                    # 서버 시각
                    srv_recv = float(times["server_recv_ts"])
                    srv_done = float(times["server_done_ts"])
                    decode_ms = float(times["decode_ms"])
                    infer_ms  = float(times["infer_ms"])
                    server_total_ms = float(times["server_total_ms"])
                    # 클라-서버 오프셋으로 편도 분리
                    with offset_lock:
                        off = offset
                    # 클라 시각을 서버 시각으로 환산
                    send_srv = t_req0_c + off
                    recv_srv = t_req1_c + off
                    send_ms = max(0.0, (srv_recv - send_srv) * 1000.0)
                    ret_ms  = max(0.0, (recv_srv - srv_done) * 1000.0)
                    rtt_ms  = (t_req1_c - t_req0_c) * 1000.0

                    store.set({
                        "frame_id": resp["frame_id"],
                        "cap_time": float(resp["cap_time"]),
                        "persons": resp["persons"],
                        "infer_ms": infer_ms,
                        "server_total_ms": server_total_ms,
                        "decode_ms": decode_ms,
                        "send_ms": send_ms,
                        "return_ms": ret_ms,
                        "rtt_ms": rtt_ms,
                        "scale": scale,
                    })

                    # 이동평균 버퍼
                    buf_send.append(send_ms)
                    buf_ret.append(ret_ms)
                    buf_inf.append(infer_ms)
                    buf_tot.append(server_total_ms)

                    # CSV
                    if csv_fp:
                        csv_fp.write(f"{resp['frame_id']},0,{send_ms:.3f},{ret_ms:.3f},{infer_ms:.3f},{server_total_ms:.3f},{decode_ms:.3f},{rtt_ms:.3f},0\n")
            except Exception:
                pass
            q.task_done()

    threading.Thread(target=worker, daemon=True).start()

    period = 1.0/max(1e-6,args.target_fps)
    frame_id = 0
    cv2.namedWindow("Client Live (sent) + delayed pose", cv2.WINDOW_NORMAL)
    print("[도움말] q 종료, p 일시정지/재개")

    paused = False
    last_disp = time.perf_counter()

    # 전체 평균용 누적
    cnt = 0
    sum_send = sum_ret = sum_inf = sum_tot = 0.0

    while True:
        loop0 = time.perf_counter()
        if not paused:
            ok, frame = cap.read()
            if not ok: break

            h, w = frame.shape[:2]
            scale = min(1.0, float(args.send_long) / max(h, w))
            frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale<1.0 else frame

            cap_time = time.perf_counter()
            try:
                q.put((frame_id, frame_small.copy(), cap_time, scale), timeout=0.005)
            except Full:
                pass

            display = frame.copy()
            res = store.get()
            if res is not None:
                age_ms = (time.perf_counter() - res["cap_time"])*1000.0
                s = float(res.get("scale", 1.0))
                persons_xy = []
                for p in (res["persons"] or []):
                    k = np.array(p["xy"], dtype=np.float32)
                    if s > 0 and s != 1.0:
                        k[:,0] = k[:,0] / s
                        k[:,1] = k[:,1] / s
                    persons_xy.append(k)
                persons_conf = [ (np.array(p["conf"], dtype=np.float32) if p["conf"] is not None else None)
                                 for p in (res["persons"] or []) ] if res["persons"] else None
                draw_pose(display, persons_xy, persons_conf, args.min_kpt_conf)

                now = time.perf_counter()
                disp_fps = 1.0 / max(1e-6, (now - last_disp)); last_disp = now

                # 이동평균
                def avg(dq): return (sum(dq)/len(dq)) if dq else 0.0
                avg_send = avg(buf_send); avg_ret = avg(buf_ret)
                avg_inf  = avg(buf_inf);  avg_tot = avg(buf_tot)

                # 전체 평균 누적
                sum_send += res["send_ms"]; sum_ret += res["return_ms"]
                sum_inf  += res["infer_ms"]; sum_tot += res["server_total_ms"]; cnt += 1

                lines = [
                    f"Sent frame id: {frame_id}",
                    f"Latest result id: {res['frame_id']} | age: {age_ms:6.1f} ms",
                    f"ONE-WAY  send: {res['send_ms']:5.1f} ms  | return: {res['return_ms']:5.1f} ms",
                    f"AVG({len(buf_tot):>3}f) send: {avg_send:5.1f}  return: {avg_ret:5.1f}  infer: {avg_inf:5.1f}  srvTotal: {avg_tot:5.1f}",
                    f"Display FPS: {disp_fps:4.1f} | imgsz={args.imgsz} | send_long={args.send_long}px | jpeg_q={args.jpeg_q} | {args.transport}",
                ]
                for i,t in enumerate(lines):
                    cv2.putText(display,t,(12,30+24*i),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)

                # CSV 최신 값을 업데이트(나중에 원하는 포맷으로 조정 가능)
                if csv_fp:
                    csv_fp.write(f"{res['frame_id']},{age_ms:.3f},{res['send_ms']:.3f},{res['return_ms']:.3f},{res['infer_ms']:.3f},{res['server_total_ms']:.3f},{res['decode_ms']:.3f},{res['rtt_ms']:.3f},{disp_fps:.2f}\n")

            cv2.imshow("Client Live (sent) + delayed pose", display)
            frame_id += 1

        key = cv2.waitKey(1)&0xFF
        if key==ord('q') or key==27: break
        elif key==ord('p'): paused = not paused

        sleep = period - (time.perf_counter()-loop0)
        if sleep>0: time.sleep(sleep)

    # 종료
    stop = True
    cap.release(); cv2.destroyAllWindows()
    if csv_fp: csv_fp.close()

    if cnt>0:
        print("[요약 평균(전체 구간)]")
        print(f" send(one-way): {sum_send/cnt:.2f} ms")
        print(f" return(one-way): {sum_ret/cnt:.2f} ms")
        print(f" infer(server): {sum_inf/cnt:.2f} ms")
        print(f" server_total: {sum_tot/cnt:.2f} ms")

if __name__ == "__main__":
    main()
