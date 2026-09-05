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
# Two-stage filter: landmarks remove camera noise, adaptive cursor smoothing
# stays precise for small motions but catches up quickly on large motions.
LANDMARK_SMOOTH = 0.38
CURSOR_SLOW = 0.12
CURSOR_FAST = 0.68
CURSOR_SPEED_REF = 0.10  # fraction of the screen diagonal
CURSOR_DEADZONE = 2.5
CLICK_ON, CLICK_OFF = 0.34, 0.50
CLICK_CONFIRM_FRAMES = 3
RELEASE_CONFIRM_FRAMES = 2
COOLDOWN = 0.30
# Ring + little finger gesture: upward = scroll up, downward = scroll down.
SCROLL_K = 0.55
SCROLL_DEAD = 0.65
SCROLL_SMOOTH = 0.42
SCROLL_MAX = 14
# Warn only when the index fingertip reaches/leaves the physical camera frame.
INDEX_EDGE_GUARD = 0.025
INDEX_WARNING_HOLD = 0.45
MODEL = "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
WIN = "AiMouse"

WRIST, THUMB, INDEX, MIDDLE, MID_MCP = 0, 4, 8, 12, 9
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
    """Open the camera only after a backend successfully returns a real frame."""
    if sys.platform == "win32":
        backends = ((cv2.CAP_MSMF, "MSMF"),
                    (cv2.CAP_DSHOW, "DirectShow"),
                    (cv2.CAP_ANY, "automatic"))
    else:
        backends = ((cv2.CAP_ANY, "automatic"),)

    for backend, name in backends:
        cap = cv2.VideoCapture(CAM, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Some webcams need a short warm-up before the first valid frame.
        for _ in range(25):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                print("camera backend: %s" % name)
                return cap
            time.sleep(0.04)
        cap.release()

    raise RuntimeError(
        "camera %d opened but produced no frames; close apps using the camera "
        "and check Windows camera permissions" % CAM)


def to_px(lms, w, h):
    return np.array([[l.x * w, l.y * h] for l in lms], np.float32)


def click_dist(p):
    """Scale-independent distance from thumb tip to middle-finger tip."""
    palm = np.linalg.norm(p[MID_MCP] - p[WRIST])
    if palm < 1e-3:
        return math.inf
    return float(np.linalg.norm(p[THUMB] - p[MIDDLE]) / palm)


def finger_angle(p, mcp, pip, tip):
    """Angle at the PIP joint: about 180° straight, smaller when folded."""
    a = p[mcp] - p[pip]
    b = p[tip] - p[pip]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 0.0
    cosine = clamp(float(np.dot(a, b) / denom), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def is_scroll_gesture(p):
    """Recognize two raised outer fingers even when the hand is slightly rotated."""
    index_angle = finger_angle(p, 5, 6, 8)
    middle_angle = finger_angle(p, 9, 10, 12)
    ring_angle = finger_angle(p, 13, 14, 16)
    little_angle = finger_angle(p, 17, 18, 20)

    # MediaPipe often reports the short little finger as only partly straight.
    # Tolerant angle thresholds keep the gesture usable with a rotated hand.
    outer_up = ring_angle > 110 and little_angle > 105
    inner_folded = index_angle < 165 and middle_angle < 165
    return outer_up and inner_folded


def index_tip_out_of_frame(p, w, h):
    """Return True only when the index fingertip touches/leaves camera view."""
    x, y = p[INDEX]
    gx, gy = INDEX_EDGE_GUARD * w, INDEX_EDGE_GUARD * h
    return x <= gx or x >= w - gx or y <= gy or y >= h - gy


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
    filtered_tip = None
    clicking = False
    armed = False
    close_frames = release_frames = 0
    last_click = 0.0
    last_palm = None
    vel = 0.0
    acc = 0.0
    warmup = 0
    fps = 0.0
    bad = 0
    t0 = prev = time.perf_counter()
    prev_ts = -1
    index_warning_until = 0.0

    print("screen %dx%d, camera %d, press q to quit" % (SW, SH, CAM))

    with vision.HandLandmarker.create_from_options(opts) as hl:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or not frame.size:
                bad += 1
                if bad >= 30:
                    print("camera feed interrupted, reconnecting...")
                    cap.release()
                    try:
                        cap = open_cam()
                        bad = 0
                    except RuntimeError as error:
                        print(error)
                        break
                else:
                    time.sleep(0.02)
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
                filtered_tip = None
                last_palm = None
                clicking = False
                close_frames = release_frames = 0
                vel = acc = 0.0
                if time.perf_counter() < index_warning_until:
                    state, col = "show index finger", (0, 70, 255)
            else:
                p = to_px(hands[0], w, h)
                draw_hand(frame, p)
                d = click_dist(p)
                if index_tip_out_of_frame(p, w, h):
                    index_warning_until = time.perf_counter() + INDEX_WARNING_HOLD

                if time.perf_counter() < index_warning_until:
                    state, col = "show index finger", (0, 70, 255)
                    cx = cy = None
                    filtered_tip = None
                    last_palm = None
                    clicking = False
                    armed = False
                    close_frames = release_frames = 0
                    vel = acc = 0.0
                elif is_scroll_gesture(p):
                    state, col = "scroll: ring + little", (255, 200, 60)
                    cx = cy = None
                    filtered_tip = None
                    clicking = False
                    close_frames = release_frames = 0
                    # Follow the two active fingertips instead of the whole palm.
                    scroll_y = float(np.mean(p[[16, 20], 1]))
                    if last_palm is None:
                        last_palm = scroll_y
                        warmup = 2
                        vel = acc = 0.0
                    else:
                        dy = last_palm - scroll_y
                        last_palm = scroll_y
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
                    if d < CLICK_ON:
                        close_frames += 1
                        release_frames = 0
                    elif d > CLICK_OFF:
                        release_frames += 1
                        close_frames = 0
                    else:
                        close_frames = release_frames = 0

                    if release_frames >= RELEASE_CONFIRM_FRAMES:
                        clicking = False
                        armed = True
                    elif (close_frames >= CLICK_CONFIRM_FRAMES and not clicking):
                        clicking = True
                        if armed and now - last_click >= COOLDOWN:
                            pyautogui.click()
                            last_click = now
                            armed = False
                            cv2.circle(frame, tuple(p[MIDDLE].astype(int)),
                                       18, (0, 0, 255), 3, cv2.LINE_AA)

                    state = "click" if clicking else "move"
                    col = (80, 80, 255) if clicking else (90, 230, 90)

                    raw_tip = p[INDEX].copy()
                    if filtered_tip is None:
                        filtered_tip = raw_tip
                    else:
                        filtered_tip += (raw_tip - filtered_tip) * LANDMARK_SMOOTH
                    fx, fy = filtered_tip
                    tx = float(np.interp(fx, (MARGIN * w, (1 - MARGIN) * w), (0, SW - 1)))
                    ty = float(np.interp(fy, (MARGIN * h, (1 - MARGIN) * h), (0, SH - 1)))
                    tx, ty = clamp(tx, 0, SW - 1), clamp(ty, 0, SH - 1)

                    if cx is None:
                        cx, cy = tx, ty
                    elif not clicking:
                        dx, dy = tx - cx, ty - cy
                        distance = math.hypot(dx, dy)
                        if distance > CURSOR_DEADZONE:
                            speed = clamp(distance / (math.hypot(SW, SH) *
                                                       CURSOR_SPEED_REF), 0.0, 1.0)
                            alpha = CURSOR_SLOW + (CURSOR_FAST - CURSOR_SLOW) * speed
                            cx += dx * alpha
                            cy += dy * alpha

                    pyautogui.moveTo(int(clamp(cx, 0, SW - 1)), int(clamp(cy, 0, SH - 1)))
                    cv2.circle(frame, tuple(filtered_tip.astype(int)), 11, col, 2,
                               cv2.LINE_AA)
                    cv2.line(frame, tuple(p[THUMB].astype(int)),
                             tuple(p[MIDDLE].astype(int)), col, 2, cv2.LINE_AA)

            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 0:
                fps += (1 / dt - fps) * 0.15

            mx, my = int(MARGIN * w), int(MARGIN * h)
            cv2.rectangle(frame, (mx, my), (w - mx, h - my), (255, 170, 0), 2)
            cv2.putText(frame, state, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2,
                        cv2.LINE_AA)
            if state == "show index finger":
                message = "SHOW INDEX FINGER"
                (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
                x, y = (w - tw) // 2, h // 2
                cv2.rectangle(frame, (x - 14, y - th - 14),
                              (x + tw + 14, y + 14), (0, 0, 0), -1)
                cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.85, col, 2, cv2.LINE_AA)
            info = "fps %.0f" % fps
            if d is not None and math.isfinite(d):
                info += "   thumb-middle %.2f" % d
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
