"""Match one demo run to the confirmed VnTraffic crossings in frames 0–500."""

import argparse
import csv
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from counting import CLASSES, REFERENCE_FILE, REVIEW_LAST_FRAME, CountingLine, load_reference_events
from demo import BYTETracker, Boxes, IterableSimpleNamespace, YAML, check_yaml

MAX_SECONDS = 0.40
MAX_Y_PIXELS = 24


@dataclass(frozen=True)
class Event:
    track_id: int
    frame: int
    crossing_time: float
    crossing_y: float
    vehicle: str
    direction: str
    review_id: str = ""
    gap: int = 0

    def key(self):
        return self.frame, self.track_id, self.vehicle, self.direction


def read_run_geometry(run, fps=30):
    """Replay this run's detections through the same tracker and crossing rule."""
    config = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    tracker = BYTETracker(args=config, frame_rate=fps)
    line = CountingLine(960, 540)
    history = defaultdict(list)
    events = []
    checkpoint = (run / "checkpoint.sqlite").resolve()
    with closing(sqlite3.connect(checkpoint.as_uri() + "?mode=ro", uri=True)) as db:
        records = db.execute("SELECT frame, boxes FROM detections WHERE frame <= ? ORDER BY frame",
                             (REVIEW_LAST_FRAME,))
        total_frames = 0
        for frame, blob in records:
            if frame != total_frames:
                raise ValueError(f"Missing detections for frame {total_frames}")
            total_frames += 1
            boxes = np.frombuffer(blob, dtype=np.float32).reshape(-1, 6).copy()
            for track in tracker.update(Boxes(boxes, (540, 960))):
                left, _, right, bottom, track_id = map(int, track[:5])
                x = (left + right) // 2
                samples = history[track_id]
                samples.append((frame, x, bottom))
                direction = line.update(track_id, (x, bottom))
                if direction:
                    t, y, gap = crossing_geometry(samples, line.x, direction)
                    vehicle = CLASSES[int(track[6])]
                    events.append(Event(track_id, frame, t / fps, y, vehicle, direction, gap=gap))
        if total_frames != REVIEW_LAST_FRAME + 1:
            raise ValueError(f"Expected {REVIEW_LAST_FRAME + 1} frames, got {total_frames}")
    with (run / "events.csv").open(newline="", encoding="utf-8-sig") as stream:
        expected = [(int(row["frame_zero_based"]), int(row["track_id"]), row["vehicle"], row["direction"])
                    for row in csv.DictReader(stream) if 0 <= int(row["frame_zero_based"]) <= REVIEW_LAST_FRAME]
    if [event.key() for event in events] != expected:
        raise ValueError("Tracker replay differs from the exported events.csv")
    return events


def crossing_geometry(samples, line_x, direction):
    sign = 1 if direction == "L-R" else -1
    for after, before in zip(reversed(samples[1:]), reversed(samples[:-1])):
        f0, x0, y0 = before
        f1, x1, y1 = after
        if (x0 - line_x) * sign <= 0 < (x1 - line_x) * sign:
            fraction = (line_x - x0) / (x1 - x0)
            return f0 + fraction * (f1 - f0), y0 + fraction * (y1 - y0), f1 - f0
    raise ValueError("Confirmed crossing has no segment through the line")


def associate(reference, predictions):
    """Match by direction, time and position, before checking the predicted class."""
    ref_options, demo_options = defaultdict(set), defaultdict(set)
    n_ref, n_demo = len(reference), len(predictions)
    missing = 3 * (min(n_ref, n_demo) + 1)
    costs = np.full((n_ref, n_demo + n_ref), 2.0 * missing)
    costs[:, n_demo:] = missing
    for i, truth in enumerate(reference):
        for j, demo in enumerate(predictions):
            seconds = abs(truth.crossing_time - demo.crossing_time)
            pixels = abs(truth.crossing_y - demo.crossing_y)
            if truth.direction == demo.direction and seconds <= MAX_SECONDS and pixels <= MAX_Y_PIXELS:
                delay = min(abs(truth.frame - demo.frame) / 12, 1)
                costs[i, j] = seconds / MAX_SECONDS + pixels / MAX_Y_PIXELS + 0.25 * delay
                ref_options[i].add(j)
                demo_options[j].add(i)

    indices, matches = linear_sum_assignment(costs)
    matched_demo = set()
    rows = []
    for i, j in zip(indices, matches):
        truth = reference[i]
        if j >= n_demo:
            rows.append((truth, None, "reference_only", sorted(ref_options[i]), []))
        else:
            demo = predictions[j]
            matched_demo.add(j)
            rows.append((truth, demo, "correct" if truth.vehicle == demo.vehicle else "wrong_class",
                         sorted(ref_options[i] - {j}), sorted(demo_options[j] - {i})))
    for j, demo in enumerate(predictions):
        if j not in matched_demo:
            rows.append((None, demo, "demo_only", [], sorted(demo_options[j])))
    return sorted(rows, key=lambda row: (row[0] or row[1]).crossing_time)


def write_comparison(path, reference, predictions, rows):
    fields = ("review_id", "gt_id", "gt_frame", "gt_crossing_time", "gt_crossing_y", "gt_vehicle",
              "demo_id", "demo_frame", "demo_crossing_time", "demo_crossing_y", "demo_vehicle",
              "direction", "result", "other_demo_ids", "other_review_ids", "demo_crossing_gap_frames")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for gt, demo, status, other_demo, other_gt in rows:
            writer.writerow([
                *([gt.review_id, gt.track_id, gt.frame, gt.crossing_time, gt.crossing_y, gt.vehicle] if gt else [""] * 6),
                *([demo.track_id, demo.frame, round(demo.crossing_time, 3), round(demo.crossing_y, 2), demo.vehicle]
                  if demo else [""] * 5),
                (gt or demo).direction, status,
                " ".join(str(predictions[j].track_id) for j in other_demo),
                " ".join(reference[i].review_id for i in other_gt),
                demo.gap if demo else "",
            ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Folder with checkpoint.sqlite and events.csv")
    args = parser.parse_args()
    reference = [Event(**row) for row in load_reference_events(Path(__file__).with_name(REFERENCE_FILE))]
    predictions = read_run_geometry(args.run)
    rows = associate(reference, predictions)
    output = args.run / "doi-chieu-ba-lop.csv"
    write_comparison(output, reference, predictions, rows)
    correct = sum(row[2] == "correct" for row in rows)
    print(f"{len(reference)} reference, {len(predictions)} demo, {correct} correct in frames 0-{REVIEW_LAST_FRAME}")
    print(f"Precision {correct / max(1, len(predictions)):.1%}, recall {correct / max(1, len(reference)):.1%}")
    print(output)


if __name__ == "__main__":
    main()
