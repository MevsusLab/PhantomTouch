import os
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import math
import sys
import time
import urllib.request
import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mtp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

CAM = 0
CAM_W, CAM_H = 640, 480
MARGIN = 0.15
SMOOTH = 0.35
SNAP = 2.0
PINCH_ON, PINCH_OFF = 0.38, 0.55
COOLDOWN = 0.25
SCROLL_K = 0.12
SCROLL_DEAD = 1.5
SCROLL_SMOOTH = 0.5
SCROLL_MAX = 6
MODEL = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
WIN = "AiMouse"

WRIST, THUMB, INDEX, MID_MCP = 0, 4, 8, 9
PIPS = (6, 10, 14, 18)
TIPS = (8, 12, 16, 20)
PALM = [0, 5, 9, 13, 17]

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SW, SH = pyautogui.size()


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def get_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL)
    if not os.path.isfile(path):
        print("downloading %s ..." % MODEL)
        try:
            urllib.request.urlretrieve(MODEL_URL, path)
        except OSError as e:
            sys.exit("download failed (%s), grab it manually from %s" % (e, MODEL_URL))
    return path


def open_cam():
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM)
    if not cap.isOpened():
        sys.exit("camera %d is busy or missing" % CAM)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def to_px(lms, w, h):
    return np.array([[l.x * w, l.y * h] for l in lms], np.float32)


def pinch_dist(p):
    palm = np.linalg.norm(p[MID_MCP] - p[WRIST])
    if palm < 1e-3:
        return math.inf
    return float(np.linalg.norm(p[THUMB] - p[INDEX]) / palm)


def is_fist(p):
    for pip, tip in zip(PIPS, TIPS):
        if np.linalg.norm(p[tip] - p[WRIST]) >= np.linalg.norm(p[pip] - p[WRIST]):
            return False
    return True


def draw_hand(img, p):
    for c in HandLandmarksConnections.HAND_CONNECTIONS:
        a = tuple(np.round(p[c.start]).astype(int))
        b = tuple(np.round(p[c.end]).astype(int))
        cv2.line(img, a, b, (200, 200, 200), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(p):
        if i == THUMB or i in TIPS:
            cv2.circle(img, (int(x), int(y)), 7, (0, 215, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (int(x), int(y)), 4, (120, 255, 120), -1, cv2.LINE_AA)


def main():
    opts = vision.HandLandmarkerOptions(
        base_options=mtp.BaseOptions(model_asset_path=get_model()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6)

    cap = open_cam()
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    cx = cy = None
    pinching = False
    armed = False
    last_click = 0.0
    last_palm = None
    vel = 0.0
    acc = 0.0
    warmup = 0
    fps = 0.0
    bad = 0
    t0 = prev = time.perf_counter()
    prev_ts = -1

    print("screen %dx%d, camera %d, press q to quit" % (SW, SH, CAM))

    with vision.HandLandmarker.create_from_options(opts) as hl:
        while True:
            ok, frame = cap.read()
            if not ok:
                bad += 1
                if bad > 30:
                    print("lost the camera feed")
                    break
                time.sleep(0.01)
                continue
            bad = 0

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            ts = int((time.perf_counter() - t0) * 1000)
            if ts <= prev_ts:
                ts = prev_ts + 1
            prev_ts = ts

            res = hl.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            hands = res.hand_landmarks

            state, col, d = "no hand", (140, 140, 140), None

            if not hands:
                cx = cy = None
                last_palm = None
                pinching = False
                vel = acc = 0.0
            else:
                p = to_px(hands[0], w, h)
                draw_hand(frame, p)
                d = pinch_dist(p)

                if is_fist(p):
                    state, col = "scroll", (255, 200, 60)
                    cx = cy = None
                    pinching = False
                    palm = float(np.mean(p[PALM, 1]))
                    if last_palm is None:
                        last_palm = palm
                        warmup = 2
                        vel = acc = 0.0
                    else:
                        dy = last_palm - palm
                        last_palm = palm
                        if warmup:
                            warmup -= 1
                        else:
                            vel += (dy - vel) * SCROLL_SMOOTH
                            if abs(vel) > SCROLL_DEAD:
                                acc += vel * SCROLL_K
                            n = int(acc)
                            if n:
                                acc -= n
                                pyautogui.scroll(int(clamp(n, -SCROLL_MAX, SCROLL_MAX)))
                    cv2.putText(frame, "up" if vel > 0 else "down" if vel < 0 else "-",
                                (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2,
                                cv2.LINE_AA)
                else:
                    last_palm = None
                    vel = acc = 0.0

                    now = time.perf_counter()
                    if d > PINCH_OFF:
                        pinching = False
                        armed = True
                    elif d < PINCH_ON and not pinching:
                        pinching = True
                        if armed and now - last_click >= COOLDOWN:
                            pyautogui.click()
                            last_click = now
                            armed = False
                            cv2.circle(frame, (int(p[INDEX][0]), int(p[INDEX][1])),
                                       18, (0, 0, 255), 3, cv2.LINE_AA)

                    state = "click" if pinching else "move"
                    col = (80, 80, 255) if pinching else (90, 230, 90)

                    fx, fy = p[INDEX]
                    tx = float(np.interp(fx, (MARGIN * w, (1 - MARGIN) * w), (0, SW - 1)))
                    ty = float(np.interp(fy, (MARGIN * h, (1 - MARGIN) * h), (0, SH - 1)))

                    if cx is None:
                        cx, cy = tx, ty
                    elif not pinching:
                        cx += (tx - cx) * SMOOTH
                        cy += (ty - cy) * SMOOTH
                        if abs(tx - cx) < SNAP:
                            cx = tx
                        if abs(ty - cy) < SNAP:
                            cy = ty

                    pyautogui.moveTo(int(clamp(cx, 0, SW - 1)), int(clamp(cy, 0, SH - 1)))
                    cv2.circle(frame, (int(fx), int(fy)), 11, col, 2, cv2.LINE_AA)

            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 0:
                fps += (1 / dt - fps) * 0.15

            mx, my = int(MARGIN * w), int(MARGIN * h)
            cv2.rectangle(frame, (mx, my), (w - mx, h - my), (255, 170, 0), 2)
            cv2.putText(frame, state, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2,
                        cv2.LINE_AA)
            info = "fps %.0f" % fps
            if d is not None and math.isfinite(d):
                info += "   pinch %.2f" % d
            cv2.putText(frame, info, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (230, 230, 230), 1, cv2.LINE_AA)
            cv2.imshow(WIN, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
