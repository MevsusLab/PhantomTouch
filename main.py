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
from phantom_config import PhantomSettings, load_settings

CAM = 0
CAM_W, CAM_H = 640, 480
MARGIN = 0.15


LANDMARK_SMOOTH = 0.38
CURSOR_SLOW = 0.12
CURSOR_FAST = 0.68
CURSOR_SPEED_REF = 0.10
CURSOR_DEADZONE = 2.5
CLICK_ON, CLICK_OFF = 0.34, 0.50
CLICK_CONFIRM_FRAMES = 3
RELEASE_CONFIRM_FRAMES = 2
COOLDOWN = 0.30


SCROLL_DEAD_SPEED = 0.12
SCROLL_GAIN = 32.0
SCROLL_SMOOTH = 0.42
SCROLL_MAX_RATE = 55.0
SCROLL_ACTIVATE_FRAMES = 2
SCROLL_RELEASE_FRAMES = 3

FIST_CONFIRM_FRAMES = 3
FIST_RELEASE_FRAMES = 4
APP_SWITCH_COOLDOWN = 0.9

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


SCROLL_FINGERS = ((13, 14, 15, 16), (17, 18, 19, 20))
SCROLL_COLORS = ((255, 80, 40), (40, 80, 255))

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


def open_cam(camera_index=CAM):
    """Open the selected camera only after a backend returns a real frame."""
    if sys.platform == "win32":
        backends = ((cv2.CAP_MSMF, "MSMF"),
                    (cv2.CAP_DSHOW, "DirectShow"),
                    (cv2.CAP_ANY, "automatic"))
    else:
        backends = ((cv2.CAP_ANY, "automatic"),)

    for backend, name in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


        for _ in range(25):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                print("camera backend: %s" % name)
                return cap
            time.sleep(0.04)
        cap.release()

    raise RuntimeError(
        "camera %d opened but produced no frames; close apps using the camera "
        "and check Windows camera permissions" % camera_index)


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


def finger_extension(p, mcp, pip, dip, tip):
    """Measure finger reach relative to its joint-chain length."""
    chain_length = sum(float(np.linalg.norm(p[b] - p[a]))
                       for a, b in ((mcp, pip), (pip, dip), (dip, tip)))
    if chain_length < 1e-3:
        return 0.0
    return float(np.linalg.norm(p[tip] - p[mcp]) / chain_length)


def scroll_pose(p):
    """Hold index/middle folded; ring/little may freely flex or extend."""
    index_angle = finger_angle(p, 5, 6, 8)
    middle_angle = finger_angle(p, 9, 10, 12)



    return index_angle < 150 and middle_angle < 150 and not is_fist(p)


def scroll_finger_position(p):
    """Average ring/little extension relative to their stable MCP knuckles."""
    return sum(finger_extension(p, *chain) for chain in SCROLL_FINGERS) / 2.0


def is_scroll_gesture(p):
    return scroll_pose(p)


def is_fist(p):
    """Recognize a closed fist independently of hand size and mild rotation."""
    palm_size = float(np.linalg.norm(p[MID_MCP] - p[WRIST]))
    if palm_size < 1e-3:
        return False

    angles = (
        finger_angle(p, 5, 6, 8),
        finger_angle(p, 9, 10, 12),
        finger_angle(p, 13, 14, 16),
        finger_angle(p, 17, 18, 20),
    )


    tip_near_palm = all(
        np.linalg.norm(p[tip] - p[MID_MCP]) / palm_size < limit
        for tip, limit in zip(TIPS, (1.35, 1.25, 1.30, 1.45))
    )


    thumb_tucked = np.linalg.norm(p[THUMB] - p[MID_MCP]) / palm_size < 0.70
    return max(angles) < 150 and tip_near_palm and thumb_tucked


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


    for chain, color in zip(SCROLL_FINGERS, SCROLL_COLORS):
        points = np.round(p[list(chain)]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [points], False, (20, 20, 20), 8, cv2.LINE_AA)
        cv2.polylines(img, [points], False, color, 5, cv2.LINE_AA)
        for point in points[:, 0]:
            cv2.circle(img, tuple(point), 6, color, -1, cv2.LINE_AA)


def draw_scroll_indicator(img, p, velocity):
    """Draw a compact palm-anchored indicator of finger movement."""
    anchor = np.mean(p[[13, 17]], axis=0).astype(int)
    magnitude = int(clamp(abs(velocity) * 25.0, 4, 30))
    direction = -1 if velocity > SCROLL_DEAD_SPEED else (
        1 if velocity < -SCROLL_DEAD_SPEED else 0)
    color = (80, 240, 80) if direction < 0 else (
        (80, 170, 255) if direction > 0 else (180, 180, 180))
    start = (int(anchor[0] + 24), int(anchor[1]))
    end = (start[0], start[1] + direction * magnitude)
    cv2.circle(img, start, 5, color, -1, cv2.LINE_AA)
    if direction:
        cv2.arrowedLine(img, start, end, color, 4, cv2.LINE_AA, tipLength=0.4)


def main(settings=None, stop_event=None, status_callback=None):
    """Run gesture control until q/window close or stop_event is set."""
    settings = settings or load_settings()
    if not isinstance(settings, PhantomSettings):
        settings = PhantomSettings.from_mapping(settings)
    landmark_smooth = settings.cursor_smoothing
    cursor_fast = settings.cursor_responsiveness
    cursor_slow = min(CURSOR_SLOW, cursor_fast)
    scroll_gain = settings.scroll_sensitivity
    scroll_dead_speed = settings.scroll_dead_zone

    opts = vision.HandLandmarkerOptions(
        base_options=mtp.BaseOptions(model_asset_path=get_model()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6)

    cap = open_cam(settings.camera_index)
    if settings.debug_overlay:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    if status_callback:
        status_callback("running")

    cx = cy = None
    filtered_tip = None
    clicking = False
    armed = False
    close_frames = release_frames = 0
    last_click = 0.0
    last_palm = None
    last_scroll_time = None
    vel = 0.0
    acc = 0.0
    scroll_active = False
    scroll_on_frames = scroll_off_frames = 0
    fist_frames = fist_release_frames = 0
    fist_armed = True
    last_app_switch = 0.0
    fps = 0.0
    bad = 0
    t0 = prev = time.perf_counter()
    prev_ts = -1
    index_warning_until = 0.0

    print("screen %dx%d, camera %d, press q to quit" % (SW, SH, CAM))

    try:
        with vision.HandLandmarker.create_from_options(opts) as hl:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                ok, frame = cap.read()
                if not ok or frame is None or not frame.size:
                    bad += 1
                    if bad >= 30:
                        print("camera feed interrupted, reconnecting...")
                        cap.release()
                        try:
                            cap = open_cam(settings.camera_index)
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
                    last_palm = last_scroll_time = None
                    scroll_active = False
                    scroll_on_frames = scroll_off_frames = 0
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
                    else:
                        now = time.perf_counter()
                        fist_now = is_fist(p)
                        if fist_now:
                            fist_frames += 1
                            fist_release_frames = 0
                        else:
                            fist_frames = 0
                            fist_release_frames += 1
                            if fist_release_frames >= FIST_RELEASE_FRAMES:
                                fist_armed = True


                        gesture_now = not fist_now and is_scroll_gesture(p)
                        if gesture_now:
                            scroll_on_frames += 1
                            scroll_off_frames = 0
                            if scroll_on_frames >= SCROLL_ACTIVATE_FRAMES:
                                scroll_active = True
                        else:
                            scroll_off_frames += 1
                            scroll_on_frames = 0
                            if scroll_off_frames >= SCROLL_RELEASE_FRAMES or fist_now:
                                scroll_active = False

                        if fist_now:
                            state, col = "fist: next application", (180, 90, 255)
                            cx = cy = None
                            filtered_tip = None
                            last_palm = last_scroll_time = None
                            vel = acc = 0.0
                            clicking = False
                            close_frames = release_frames = 0
                            if (fist_frames >= FIST_CONFIRM_FRAMES and fist_armed and
                                    now - last_app_switch >= APP_SWITCH_COOLDOWN):
                                if settings.fist_alt_tab:
                                    pyautogui.hotkey("alt", "tab")
                                last_app_switch = now
                                fist_armed = False
                        elif scroll_active:
                            state, col = "scroll: ring + little", (255, 200, 60)
                            cx = cy = None
                            filtered_tip = None
                            clicking = False
                            close_frames = release_frames = 0

                            now = time.perf_counter()



                            finger_pos = scroll_finger_position(p)
                            if last_palm is None or last_scroll_time is None:
                                last_palm = finger_pos
                                last_scroll_time = now
                                vel = acc = 0.0
                            else:
                                dt_scroll = clamp(now - last_scroll_time, 1 / 120, 0.10)
                                raw_speed = (finger_pos - last_palm) / dt_scroll
                                last_palm = finger_pos
                                last_scroll_time = now
                                vel += (raw_speed - vel) * SCROLL_SMOOTH

                                effective_speed = (math.copysign(
                                    abs(vel) - scroll_dead_speed, vel)
                                    if abs(vel) > scroll_dead_speed else 0.0)
                                rate = clamp(effective_speed * scroll_gain,
                                             -SCROLL_MAX_RATE, SCROLL_MAX_RATE)
                                acc += rate * dt_scroll
                                n = math.trunc(acc)
                                if n:
                                    acc -= n
                                    pyautogui.scroll(n)

                            direction = "UP" if vel > scroll_dead_speed else (
                                "DOWN" if vel < -scroll_dead_speed else "HOLD")
                            draw_scroll_indicator(frame, p, vel)
                            cv2.putText(frame, direction, (12, h - 16),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2,
                                        cv2.LINE_AA)
                        else:
                            last_palm = None
                            last_scroll_time = None
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
                                filtered_tip += (raw_tip - filtered_tip) * landmark_smooth
                            fx, fy = filtered_tip
                            tx = float(np.interp(fx, (MARGIN * w, (1 - MARGIN) * w),
                                                 (0, SW - 1)))
                            ty = float(np.interp(fy, (MARGIN * h, (1 - MARGIN) * h),
                                                 (0, SH - 1)))
                            tx, ty = clamp(tx, 0, SW - 1), clamp(ty, 0, SH - 1)

                            if cx is None:
                                cx, cy = tx, ty
                            elif not clicking:
                                dx, dy = tx - cx, ty - cy
                                distance = math.hypot(dx, dy)
                                if distance > CURSOR_DEADZONE:
                                    speed = clamp(distance / (math.hypot(SW, SH) *
                                                               CURSOR_SPEED_REF), 0.0, 1.0)
                                    alpha = cursor_slow + (cursor_fast - cursor_slow) * speed
                                    cx += dx * alpha
                                    cy += dy * alpha

                            pyautogui.moveTo(int(clamp(cx, 0, SW - 1)),
                                             int(clamp(cy, 0, SH - 1)))
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
                if settings.debug_overlay:
                    cv2.imshow(WIN, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                        break
                else:
                    time.sleep(0.001)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
