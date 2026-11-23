import argparse
import csv
import hashlib
import os
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

from counting import (
    CATEGORIES, CLASSES, DIRECTION_LABELS, REVIEW_LAST_FRAME, VEHICLE_LABELS,
    CountingLine, load_reviewed_counts,
)
from sample_data import ASSETS, VIDEO, WEIGHTS, prepare

config_dir = ASSETS / "settings"
config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")

import cv2
import numpy as np
import torch
from filelock import FileLock
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

COLORS = {"car": (75, 220, 110), "motorcycle": (65, 190, 255)}
ROOT = Path(__file__).resolve().parent
WINDOW = "VnTraffic | Q / Esc"
SUMMARY_SECONDS = 5
PANEL_WIDTH = 408
PANEL_COLUMNS = (208, 330)


def font(size, bold=False):
    fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    filename = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(fonts / filename), size)


def counting_line_label():
    text = "Vạch đếm"
    label_font = font(13, True)
    left, top, right, bottom = label_font.getbbox(text)
    label = Image.new("RGB", (right - left + 12, bottom - top + 2), "#141e2e")
    ImageDraw.Draw(label).text((6 - left, 1 - top), text, font=label_font, fill="white")
    return cv2.cvtColor(np.asarray(label), cv2.COLOR_RGB2BGR)


def count_rows(counts):
    rows = [[counts.get((vehicle, direction), 0) for direction in DIRECTION_LABELS]
            for vehicle in VEHICLE_LABELS]
    totals = [sum(column) for column in zip(*rows)]
    return [*rows, totals]


def draw_count_table(draw, top, rows):
    draw.rounded_rectangle((20, top, 388, top + 166), radius=10, fill="#1d2a3d", outline="#3d4d65")
    heading_font = font(17, True)
    draw.text((36, top + 20), "Loại xe", font=heading_font, anchor="lm", fill="#c3cfe1")
    for x, direction in zip(PANEL_COLUMNS, DIRECTION_LABELS.values()):
        draw.text((x, top + 20), direction, font=heading_font, anchor="mm", fill="#c3cfe1")

    labels = (*VEHICLE_LABELS.values(), "Tổng")
    label_colors = (*(COLORS[vehicle][::-1] for vehicle in VEHICLE_LABELS), "white")
    for i, (label, color, values) in enumerate(zip(labels, label_colors, rows)):
        y = top + 40 + i * 42
        draw.line((34, y, 374, y), fill="#39495f")
        bold = i == len(labels) - 1
        draw.text((36, y + 21), label, font=font(20, bold), anchor="lm", fill=color)
        for x, value in zip(PANEL_COLUMNS, values):
            draw.text((x, y + 21), str(value), font=font(22, bold), anchor="mm", fill="white")


def draw_reviewed_comparison(draw, review):
    draw.text((20, 274), f"Đối chiếu Ground Truth · khung 0–{REVIEW_LAST_FRAME}",
              font=font(20, True), anchor="lm", fill="white")
    top = 296
    draw.rounded_rectangle((20, top, 388, top + 160), radius=10, fill="#1d2a3d", outline="#3d4d65")
    columns = (34, 126, 234, 329)
    anchors = ("lm", "lm", "mm", "mm")
    header_font = font(14, True)
    for x, anchor, label in zip(columns, anchors, ("Loại xe", "Chiều đi", "Tham chiếu", "Demo đã duyệt")):
        draw.text((x, top + 17), label, font=header_font, anchor=anchor, fill="#c3cfe1")
    rows = [(VEHICLE_LABELS[vehicle], DIRECTION_LABELS[direction],
             review.reference[vehicle, direction], review.demo[vehicle, direction])
            for vehicle, direction in CATEGORIES]
    rows.append(("Tổng", "Cả hai chiều", sum(review.reference.values()), sum(review.demo.values())))
    for index, row in enumerate(rows):
        y = top + 34 + 25 * index
        draw.line((32, y, 376, y), fill="#39495f")
        row_font = font(16, index == len(rows) - 1)
        for x, anchor, value in zip(columns, anchors, row):
            draw.text((x, y + 12), str(value), font=row_font, anchor=anchor, fill="white")
    draw.multiline_text((20, 467),
                        "Số tham chiếu dùng quỹ đạo Ground Truth và\n"
                        "loại xe đã xác nhận thủ công. Lượt ở khung 507\n"
                        "nằm ngoài phạm vi đối chiếu.",
                        font=font(15), spacing=3, fill="#c3cfe1")


def counter_panel(height, counts, review=None):
    panel = Image.new("RGB", (PANEL_WIDTH, height), "#141e2e")
    draw = ImageDraw.Draw(panel)
    draw.line((0, 0, 0, height), fill="#55657c", width=2)
    draw.text((20, 44), "Lượt qua vạch", font=font(30, True), anchor="lm", fill="white")
    system_rows = count_rows(counts)
    draw_count_table(draw, 82, system_rows)

    if review is not None:
        draw_reviewed_comparison(draw, review)

    return cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)


def open_window(width, height):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if os.name == "nt":
        from ctypes import byref, windll
        from ctypes.wintypes import RECT

        work_area = RECT()
        if windll.user32.SystemParametersInfoW(0x0030, 0, byref(work_area), 0):  # SPI_GETWORKAREA
            available_width = work_area.right - work_area.left
            available_height = work_area.bottom - work_area.top
            scale = min(1, 0.94 * available_width / width, 0.88 * available_height / height)
            width, height = round(width * scale), round(height * scale)
            cv2.moveWindow(WINDOW, work_area.left + (available_width - width) // 2,
                           work_area.top + (available_height - height) // 2)
    cv2.resizeWindow(WINDOW, width, height)


def window_closed():
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def process(headless, folder, cache, total_cached, review):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = YOLO(str(WEIGHTS))
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        cap.release()
        raise RuntimeError(f"Cannot open video: {VIDEO}")
    expected = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    config = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    tracker = BYTETracker(args=config, frame_rate=round(fps))
    width = 960
    height = 2 * round(width * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    output_size = (width + PANEL_WIDTH, height)
    writer = cv2.VideoWriter(str(folder / "writing.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    line = CountingLine(width, height)
    line_label = counting_line_label()
    label_height, label_width = line_label.shape[:2]
    label_x = line.x - label_width // 2
    label_y = min(line.bottom + 2, height - label_height)
    counts = Counter()
    canvas = np.empty((height, output_size[0], 3), dtype=np.uint8)
    canvas[:, width:] = counter_panel(height, counts)
    trails = defaultdict(lambda: deque(maxlen=18))
    frames = 0
    started = time.perf_counter()
    try:
        if not writer.isOpened():
            raise RuntimeError("Cannot create output video.")
        if not headless:
            open_window(*output_size)
        with (folder / "writing.csv").open("w", newline="", encoding="utf-8-sig") as event_file:
            events = csv.writer(event_file)
            events.writerow(["frame_zero_based", "time_sec", "track_id", "vehicle", "direction"])
            while True:
                frame_started = time.perf_counter()
                ok, original = cap.read()
                if not ok:
                    if frames != expected:
                        raise RuntimeError(f"Video ended early at frame {frames} of {expected}.")
                    break
                frame = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
                if frames < total_cached:
                    row = cache.execute("SELECT boxes FROM detections WHERE frame=?", (frames,)).fetchone()
                    detections = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 6).copy()
                else:
                    result = model.predict(frame, classes=list(CLASSES), conf=0.1, imgsz=640,
                                           device="cpu", verbose=False)[0]
                    detections = result.boxes.data.cpu().numpy().astype(np.float32, copy=False)
                    with cache:
                        cache.execute("INSERT INTO detections VALUES (?, ?)", (frames, detections.tobytes()))
                tracks = tracker.update(Boxes(detections, (height, width)), frame)
                changed = False
                for track in tracks:
                    left, top, right, bottom, tid = map(int, track[:5])
                    cls = int(track[6])
                    name = CLASSES[cls]
                    color = COLORS[name]
                    point = ((left + right) // 2, bottom)
                    direction = line.update(tid, point)
                    if direction:
                        counts[name, direction] += 1
                        events.writerow([frames, round(frames / fps, 3), tid, name, direction])
                        changed = True
                    trail = trails[tid]
                    if trail:
                        for a, b in zip(trail, list(trail)[1:] + [point]):
                            cv2.line(frame, a, b, color, 1)
                    trail.append(point)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 1)
                    cv2.circle(frame, point, 2, color, cv2.FILLED, cv2.LINE_AA)
                    label = f"{name} #{tid}"
                    position = (max(0, min(left, width - 125)), max(15, top - 3))
                    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                                0.42, (20, 20, 20), 3, cv2.LINE_AA)
                    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                                0.42, color, 1, cv2.LINE_AA)
                for color, thickness in (((25, 25, 25), 5), ((240, 245, 255), 2)):
                    cv2.line(frame, (line.x, line.top), (line.x, line.bottom),
                             color, thickness, cv2.LINE_AA)
                frame[label_y:label_y + label_height, label_x:label_x + label_width] = line_label
                if changed:
                    canvas[:, width:] = counter_panel(height, counts)
                canvas[:, :width] = frame
                writer.write(canvas)
                frames += 1
                if frames % 60 == 0 or frames == expected:
                    print(f"Processed {frames}/{expected} video frames.", flush=True)
                if not headless and frames > total_cached:
                    cv2.imshow(WINDOW, canvas)
                    delay = max(1, round(1000 * (1 / fps - (time.perf_counter() - frame_started))))
                    key = cv2.waitKey(delay) & 0xff
                    if key in (27, ord("q"), ord("Q")) or window_closed():
                        headless = True
                        break
        if frames == 0:
            raise RuntimeError("Video contains no readable frames.")
        complete = frames == expected
        if complete:
            canvas[:, width:] = counter_panel(height, counts, review)
            for _ in range(round(fps * SUMMARY_SECONDS)):
                writer.write(canvas)
        writer.release()
        (folder / "writing.mp4").replace(folder / ("tracked.mp4" if complete else "partial.mp4"))
        (folder / "writing.csv").replace(folder / ("events.csv" if complete else "partial.csv"))
        if complete:
            with cache:
                cache.execute("UPDATE run SET complete=1")
        elapsed = time.perf_counter() - started
        print(f"{frames} frames in {elapsed:.1f} s, {frames / elapsed:.1f} FPS, {sum(counts.values())} crossings")
        if complete:
            print(f"Ground Truth frames 0-{REVIEW_LAST_FRAME}: {sum(review.reference.values())} reference, "
                  f"{sum(review.demo.values())} reviewed demo, {review.matched} correct matches")
        print(f"Results: {folder}", flush=True)
        if complete and not headless:
            cv2.imshow(WINDOW, canvas)
            while cv2.waitKey(100) < 0 and not window_closed():
                pass
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()


def run(headless, resume=None):
    from contextlib import closing
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    if resume is None:
        folder = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=outputs))
    else:
        folder = Path(resume).resolve()
        if not (folder / "checkpoint.sqlite").is_file():
            raise ValueError(f"No resumable checkpoint in {folder}")
    print(f"Run folder: {folder}", flush=True)
    fingerprint = hashlib.sha256()
    for source in ("demo.py", "counting.py", "sample_data.py", "requirements.txt"):
        fingerprint.update((ROOT / source).read_bytes().replace(b"\r\n", b"\n"))
    with FileLock(str(folder / "run.lock"), timeout=0), closing(sqlite3.connect(folder / "checkpoint.sqlite")) as cache:
        with cache:
            cache.execute("CREATE TABLE IF NOT EXISTS run (signature TEXT, complete INTEGER)")
            cache.execute("CREATE TABLE IF NOT EXISTS detections (frame INTEGER PRIMARY KEY, boxes BLOB NOT NULL)")
            state = cache.execute("SELECT signature, complete FROM run").fetchone()
            if state is None:
                cache.execute("INSERT INTO run VALUES (?, 0)", (fingerprint.hexdigest(),))
            elif state[1] and all((folder / name).is_file() for name in ("tracked.mp4", "events.csv")):
                print(f"Already complete. Results: {folder}")
                return
            elif state[0] != fingerprint.hexdigest():
                raise ValueError("Demo configuration changed. Start a new run without --resume.")
        cached = cache.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        print(f"Restoring {cached} cached frames." if cached else "Starting from frame 0.", flush=True)
        review = load_reviewed_counts(ROOT / "doi-chieu-da-xac-nhan.csv")
        prepare()
        process(headless, folder, cache, cached, review)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Save results without opening a window")
    parser.add_argument("--resume", type=Path, metavar="RUN_FOLDER", help="Continue a previous run")
    args = parser.parse_args()
    run(args.headless, args.resume)
