# client_stream_http.py (latest-only, single in-flight + mapping CSV/plots)
import argparse, time, threading
import requests, cv2, numpy as np
from collections import deque
from typing import List, Tuple

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
        self._lock = threading.Lock()
        self._res = None
    def set(self, d):
        with self._lock: self._res = d
    def get(self):
        with self._lock: return None if self._res is None else self._res.copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--target_fps", type=float, default=30)

    # 저지연 프리셋
    ap.add_argument("--imgsz", type=int, default=416)      # 서버 모델 입력 크기
    ap.add_argument("--send_long", type=int, default=480)  # 전송 프레임의 긴 변(px)
    ap.add_argument("--jpeg_q", type=int, default=70)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min_kpt_conf", type=float, default=0.25)
    ap.add_argument("--transport", choices=["form","raw"], default="raw")
    ap.add_argument("--use_turbojpeg", action="store_true")

    # 로깅/그래프
    ap.add_argument("--log_csv", type=str, default=None)
    ap.add_argument("--avg_window", type=int, default=120)
    ap.add_argument("--map_csv", type=str, default="mapping.csv",
                    help="표시 프레임(n) ↔ 추론 프레임(m) 매핑 CSV 저장 파일명")
    ap.add_argument("--map_plot_prefix", type=str, default="lag_plot",
                    help="그래프 파일 접두사(…_map.png, …_lag_frames.png)")
    args = ap.parse_args()

    # TurboJPEG (있으면 사용)
    jpeg = None
    TJPF_BGR_CONST = None
    if args.use_turbojpeg:
        try:
            from turbojpeg import TurboJPEG, TJPF_BGR
            jpeg = TurboJPEG()
            TJPF_BGR_CONST = TJPF_BGR
            print("[info] Using TurboJPEG for encode")
        except Exception as e:
            print(f"[warn] TurboJPEG unavailable ({e}); fallback to OpenCV)")

    # 캡처
    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source)) if is_cam else cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("[오류] 소스 열기 실패"); return
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    # HTTP 세션
    sess = requests.Session()

    # 서버 시계 동기화(간단 오프셋)
    offset_lock = threading.Lock()
    offset = 0.0  # server_time ≈ client_time + offset
    def sync_once():
        nonlocal offset
        try:
            t0 = time.time()
            r = sess.get(args.server + "/now", timeout=2)
            t1 = time.time()
            if r.ok:
                srv = float(r.json()["server_time"])
                curr = t0 + (t1 - t0)/2.0
                est = srv - curr
                with offset_lock:
                    alpha = 0.3
                    offset = alpha*est + (1-alpha)*offset
        except: pass
    for _ in range(3): sync_once()
    def timesync_worker():
        while True:
            sync_once(); time.sleep(2.0)
    threading.Thread(target=timesync_worker, daemon=True).start()

    # 최신 프레임 공유 버퍼(항상 최신만 유지)
    latest = {"frame_small": None, "cap_time": 0.0, "scale": 1.0, "id": -1}
    latest_lock = threading.Lock()

    # in-flight 제어: 단 1개 요청만
    inflight = threading.Event(); inflight.clear()

    store = ResultStore()

    # CSV & 이동평균
    csv_fp = None
    if args.log_csv:
        csv_fp = open(args.log_csv, "w", encoding="utf-8")
        csv_fp.write("frame_id,age_ms,send_ms,return_ms,server_infer_ms,server_total_ms,decode_ms,rtt_ms,disp_fps\n")
    win = args.avg_window
    buf_send, buf_ret, buf_inf, buf_tot = deque(maxlen=win), deque(maxlen=win), deque(maxlen=win), deque(maxlen=win)

    # 표시-추론 매핑 로깅용 버퍼
    map_display_ids, map_result_ids, map_lag_frames, map_age_ms = [], [], [], []

    # 송신 스레드(항상 최신 프레임만 전송)
    def sender():
        last_sent_id = -1
        while True:
            if inflight.is_set():
                time.sleep(0.001); continue

            with latest_lock:
                fid = latest["id"]
                frame_small = None if latest["frame_small"] is None else latest["frame_small"].copy()
                cap_time = latest["cap_time"]; scale = latest["scale"]

            if frame_small is None or fid <= last_sent_id:
                time.sleep(0.001); continue

            inflight.set()
            # JPEG 인코딩
            if jpeg is not None:
                try:
                    payload_bytes = jpeg.encode(frame_small, quality=args.jpeg_q, pixel_format=TJPF_BGR_CONST)
                except Exception:
                    ok, buf = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_q])
                    if not ok: inflight.clear(); continue
                    payload_bytes = buf.tobytes()
            else:
                ok, buf = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_q])
                if not ok: inflight.clear(); continue
                payload_bytes = buf.tobytes()

            t_req0_c = time.time()
            try:
                if args.transport == "form":
                    files = {"image": ("frame.jpg", payload_bytes, "image/jpeg")}
                    data  = {"frame_id": str(fid), "cap_time": str(cap_time),
                             "imgsz": str(args.imgsz), "conf": str(args.conf)}
                    r = sess.post(args.server + "/infer", files=files, data=data, timeout=10)
                else:
                    params = {"frame_id": fid, "cap_time": cap_time, "imgsz": args.imgsz, "conf": args.conf}
                    r = sess.post(args.server + "/infer_bytes", params=params,
                                  data=payload_bytes, headers={"Content-Type":"application/octet-stream"},
                                  timeout=10)
                t_req1_c = time.time()
                if r.ok:
                    resp  = r.json()
                    times = resp["times"]
                    srv_recv = float(times["server_recv_ts"])
                    srv_done = float(times["server_done_ts"])
                    decode_ms = float(times["decode_ms"])
                    infer_ms  = float(times["infer_ms"])
                    server_total_ms = float(times["server_total_ms"])
                    rtt_ms = (t_req1_c - t_req0_c) * 1000.0
                    with offset_lock: off = offset
                    send_ms = max(0.0, (srv_recv - (t_req0_c + off)) * 1000.0)
                    ret_ms  = max(0.0, ((t_req1_c + off) - srv_done) * 1000.0)

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

                    buf_send.append(send_ms); buf_ret.append(ret_ms)
                    buf_inf.append(infer_ms); buf_tot.append(server_total_ms)

                    if csv_fp:
                        csv_fp.write(f"{resp['frame_id']},0,{send_ms:.3f},{ret_ms:.3f},{infer_ms:.3f},{server_total_ms:.3f},{decode_ms:.3f},{rtt_ms:.3f},0\n")
                last_sent_id = fid
            except Exception:
                pass
            finally:
                inflight.clear()

    threading.Thread(target=sender, daemon=True).start()

    # 표시 루프
    period = 1.0/max(1e-6, args.target_fps)
    frame_id = 0
    last_disp = time.perf_counter()
    cv2.namedWindow("Client Live (latest-only) + delayed pose", cv2.WINDOW_NORMAL)
    print("[도움말] q 종료, p 일시정지/재개")

    paused = False
    while True:
        loop0 = time.perf_counter()
        if not paused:
            ok, frame = cap.read()
            if not ok: break

            # 전송용 다운스케일(긴 변 send_long)
            h, w = frame.shape[:2]
            scale = min(1.0, float(args.send_long) / max(h, w))
            frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale<1.0 else frame

            # 최신 공유 버퍼 갱신 (항상 최근 것만 유지)
            with latest_lock:
                latest["frame_small"] = frame_small
                latest["cap_time"] = time.perf_counter()
                latest["scale"] = scale
                latest["id"] = frame_id

            # 화면 표시
            display = frame.copy()
            res = store.get()
            if res is not None:
                age_ms = (time.perf_counter() - res["cap_time"]) * 1000.0
                s = float(res.get("scale", 1.0))
                persons_xy = []
                for p in (res["persons"] or []):
                    k = np.array(p["xy"], dtype=np.float32)
                    if s > 0 and s != 1.0:
                        k[:,0] /= s; k[:,1] /= s
                    persons_xy.append(k)
                persons_conf = [ (np.array(p["conf"], dtype=np.float32) if p["conf"] is not None else None)
                                 for p in (res["persons"] or []) ] if res["persons"] else None
                draw_pose(display, persons_xy, persons_conf, args.min_kpt_conf)

                # 이동 평균 및 HUD
                now = time.perf_counter()
                disp_fps = 1.0 / max(1e-6, (now - last_disp)); last_disp = now
                def avg(dq): return (sum(dq)/len(dq)) if dq else 0.0

                # === 매핑 기록 (표시 n ↔ 추론 m) ===
                lag_frames = frame_id - int(res["frame_id"])
                map_display_ids.append(int(frame_id))
                map_result_ids.append(int(res["frame_id"]))
                map_lag_frames.append(int(lag_frames))
                map_age_ms.append(float(age_ms))

                lines = [
                    f"Sent frame id (latest): {frame_id}",
                    f"Latest result id: {res['frame_id']} | age: {age_ms:6.1f} ms | lag: {lag_frames} frames",
                    f"ONE-WAY  send: {res['send_ms']:5.1f} ms | return: {res['return_ms']:5.1f} ms",
                    f"AVG({len(buf_tot):>3}f) send: {avg(buf_send):5.1f}  return: {avg(buf_ret):5.1f}  infer: {avg(buf_inf):5.1f}  srvTotal: {avg(buf_tot):5.1f}",
                    f"FPS {disp_fps:4.1f} | imgsz={args.imgsz} | send_long={args.send_long}px | jpeg_q={args.jpeg_q} | {args.transport}"
                ]
                for i,t in enumerate(lines):
                    cv2.putText(display,t,(12,30+24*i),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)

                # CSV (기존 로그 파일과 별개)
                if csv_fp:
                    csv_fp.write(f"{res['frame_id']},{age_ms:.3f},{res['send_ms']:.3f},{res['return_ms']:.3f},{res['infer_ms']:.3f},{res['server_total_ms']:.3f},{res['decode_ms']:.3f},{res['rtt_ms']:.3f},{disp_fps:.2f}\n")

            cv2.imshow("Client Live (latest-only) + delayed pose", display)
            frame_id += 1

        key = cv2.waitKey(1)&0xFF
        if key in (ord('q'),27): break
        elif key==ord('p'): paused = not paused

        sleep = period - (time.perf_counter()-loop0)
        if sleep>0: time.sleep(sleep)

    # 종료
    cap.release(); cv2.destroyAllWindows()
    if csv_fp: csv_fp.close()

    # === 매핑 CSV/그래프 저장 ===
    try:
        with open(args.map_csv, "w", encoding="utf-8") as f:
            f.write("display_frame_id,result_frame_id,lag_frames,age_ms\n")
            for n, m, k, a in zip(map_display_ids, map_result_ids, map_lag_frames, map_age_ms):
                f.write(f"{n},{m},{k},{a:.3f}\n")
        print(f"[저장] 매핑 CSV: {args.map_csv}")
    except Exception as e:
        print(f"[경고] 매핑 CSV 저장 실패: {e}")

    try:
        import matplotlib.pyplot as plt

        # (1) m vs n 산점도 + y=x 기준선
        plt.figure(figsize=(7,6))
        plt.scatter(map_display_ids, map_result_ids, s=10)
        if map_display_ids and map_result_ids:
            lo = min(min(map_display_ids), min(map_result_ids))
            hi = max(max(map_display_ids), max(map_result_ids))
            plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel("Displayed frame index (n)")
        plt.ylabel("Inferred frame index (m)")
        plt.title("Mapping: displayed (n) vs inferred (m)")
        out1 = f"{args.map_plot_prefix}_map.png"
        plt.tight_layout(); plt.savefig(out1); plt.close()
        print(f"[저장] 그래프: {out1}")

        # (2) 프레임 지연 k=n-m 라인 그래프
        plt.figure(figsize=(7,4))
        plt.plot(map_display_ids, map_lag_frames)
        plt.xlabel("Displayed frame index (n)")
        plt.ylabel("Lag (frames) = n - m")
        plt.title("Frame-lag over time")
        out2 = f"{args.map_plot_prefix}_lag_frames.png"
        plt.tight_layout(); plt.savefig(out2); plt.close()
        print(f"[저장] 그래프: {out2}")
    except Exception as e:
        print(f"[경고] 그래프 저장 건너뜀(아마 matplotlib 미설치): {e}")

if __name__ == "__main__":
    main()
