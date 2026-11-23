import csv
from collections import Counter
from dataclasses import dataclass

CLASSES = {2: "car", 3: "motorcycle"}
VEHICLE_LABELS = {"car": "Ô tô con", "motorcycle": "Xe máy"}
DIRECTION_LABELS = {"L-R": "Trái → phải", "R-L": "Phải → trái"}
CATEGORIES = tuple((vehicle, direction) for vehicle in CLASSES.values() for direction in DIRECTION_LABELS)
REVIEW_LAST_FRAME = 500


@dataclass
class TrackPosition:
    point: tuple[int, int]
    side: int
    crossed_segment: bool = False


class CountingLine:
    """Confirm the crossing only after the track leaves the band on the far side."""

    def __init__(self, width, height):
        self.x = round(width * 580 / 960)
        self.top = round(height * 315 / 540)
        self.bottom = round(height * 522 / 540)
        self.margin = max(2, round(width * 4 / 960))
        self.positions = {}

    def update(self, track_id, point):
        x, y = point
        offset = x - self.x
        side = (1 if offset > 0 else -1) if abs(offset) > self.margin else 0
        state = self.positions.get(track_id)
        if state is None or state.side == 0:
            self.positions[track_id] = TrackPosition(point, side)
            return None

        px, py = state.point
        state.point = point
        if offset * state.side > 0:
            state.crossed_segment = False
        elif offset * state.side < 0 <= (px - self.x) * state.side:
            crossing_y = py + (y - py) * (self.x - px) / (x - px)
            state.crossed_segment = self.top <= crossing_y <= self.bottom

        if side and side != state.side:
            crossed_segment = state.crossed_segment
            state.side = side
            state.crossed_segment = False
            if crossed_segment:
                return "L-R" if side == 1 else "R-L"
        return None


def load_manual_counts(path):
    """Read independently counted totals. An empty sheet or cell stays unknown."""
    if not path.is_file():
        return {}

    counts = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["vehicle", "direction", "manual_count"]:
            raise ValueError(f"{path.name}: keep the template's three column names.")
        for row in rows:
            category = row["vehicle"], row["direction"]
            value = row["manual_count"]
            if category not in CATEGORIES or category in counts or value is None or None in row:
                raise ValueError(f"{path.name}:{rows.line_num}: invalid or duplicate category.")
            value = value.strip()
            if value and (not value.isascii() or not value.isdecimal()):
                raise ValueError(f"{path.name}:{rows.line_num}: enter a whole number or leave the cell blank.")
            counts[category] = int(value) if value else None
    return counts


@dataclass(frozen=True)
class ReviewedCounts:
    reference: Counter
    demo: Counter
    matched: int


def load_reviewed_counts(path):
    """Summarize the reviewed crossing events for cars and motorcycles."""
    reference, demo = Counter(), Counter()
    matched = 0
    directions = {label: code for code, label in DIRECTION_LABELS.items()}
    confirmed_types = {label: code for code, label in VEHICLE_LABELS.items()}
    required = {
        "chieu_di", "gt_id", "gt_khung_xac_nhan", "demo_id", "demo_khung_xac_nhan",
        "loai_xe_xac_nhan", "demo_loai_yolo", "xac_nhan_ghep", "ket_qua_ghep",
    }
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if not required.issubset(rows.fieldnames or ()):
            raise ValueError(f"{path.name}: missing reviewed crossing columns.")
        for row in rows:
            try:
                direction = directions[row["chieu_di"]]
                gt_in_scope = bool(row["gt_id"]) and 0 <= int(row["gt_khung_xac_nhan"]) <= REVIEW_LAST_FRAME
                demo_in_scope = bool(row["demo_id"]) and 0 <= int(row["demo_khung_xac_nhan"]) <= REVIEW_LAST_FRAME
                gt_type = confirmed_types.get(row["loai_xe_xac_nhan"])
                demo_type = row["demo_loai_yolo"]
                if demo_in_scope and demo_type not in VEHICLE_LABELS:
                    raise ValueError(f"invalid demo vehicle: {demo_type!r}")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path.name}:{rows.line_num}: invalid crossing data.") from error

            if gt_in_scope and gt_type:
                reference[gt_type, direction] += 1
            if demo_in_scope:
                demo[demo_type, direction] += 1
            decision = row["xac_nhan_ghep"]
            same_vehicle = decision == "Cùng xe" or (
                not decision and row["ket_qua_ghep"] == "Đã ghép theo thời gian và vị trí"
            )
            if gt_in_scope and demo_in_scope and gt_type == demo_type and same_vehicle:
                matched += 1

    return ReviewedCounts(reference, demo, matched)
