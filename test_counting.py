import tempfile
import unittest
from collections import Counter
from pathlib import Path

from counting import CountingLine, REFERENCE_FILE, load_manual_counts, load_reference_events, load_reviewed_counts


class CountingLineTests(unittest.TestCase):
    def assert_crossings(self, offsets, expected):
        line = CountingLine(960, 540)
        results = [line.update(42, (line.x + dx, dy)) for dx, dy in offsets]
        self.assertEqual([result for result in results if result is not None], expected)

    def test_stop_on_line_and_continue_once(self):
        self.assert_crossings(
            [(dx, 400) for dx in (-10, -3, 0, 0, 0, 3, 8, 14)], ["L-R"]
        )

    def test_brush_line_and_return_to_same_side(self):
        self.assert_crossings([(dx, 400) for dx in (12, 3, 0, -2, 0, 5, 13)], [])

    def test_above_end_of_segment_then_turn_inside_band(self):
        self.assert_crossings([(-8, 303), (-2, 303), (2, 303), (2, 326), (8, 328)], [])

    def test_passage_on_segment_then_move_past_endpoint(self):
        self.assert_crossings([(10, 321), (2, 317), (-2, 317), (-3, 307), (-10, 301)], ["R-L"])

    def test_start_in_band(self):
        self.assert_crossings([(dx, 400) for dx in (0, 0, 3, 10)], [])

    def test_distinct_crossings_of_same_track(self):
        self.assert_crossings([(dx, 400) for dx in (-9, 10, 11, 0, -9)], ["L-R", "R-L"])

    def test_tracks_have_independent_state(self):
        line = CountingLine(960, 540)
        self.assertIsNone(line.update(1, (line.x - 9, 400)))
        self.assertIsNone(line.update(2, (line.x + 9, 400)))
        self.assertEqual(line.update(1, (line.x + 9, 400)), "L-R")
        self.assertEqual(line.update(2, (line.x - 9, 400)), "R-L")


class ManualCountTests(unittest.TestCase):
    def test_empty_cell_and_zero_are_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "counts.csv"
            path.write_text("vehicle,direction,manual_count\ncar,L-R,\ncar,R-L,0\n", encoding="utf-8-sig")
            self.assertEqual(load_manual_counts(path), {("car", "L-R"): None, ("car", "R-L"): 0})

    def test_invalid_number_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "counts.csv"
            path.write_text("vehicle,direction,manual_count\ncar,L-R,-1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manual_counts(path)


class ReviewedCountTests(unittest.TestCase):
    def test_two_class_baseline(self):
        path = Path(__file__).resolve().with_name("doi-chieu-da-xac-nhan.csv")
        review = load_reviewed_counts(path)
        self.assertEqual(review.reference, {("car", "R-L"): 7, ("motorcycle", "L-R"): 9, ("motorcycle", "R-L"): 2})
        self.assertEqual(review.demo, {("car", "L-R"): 2, ("car", "R-L"): 6,
                                      ("motorcycle", "L-R"): 11, ("motorcycle", "R-L"): 1})
        self.assertEqual(review.matched, 16)

    def test_three_class_reference(self):
        path = Path(__file__).resolve().with_name(REFERENCE_FILE)
        events = load_reference_events(path)
        self.assertEqual(len(events), 21)
        self.assertEqual(len({(row["track_id"], row["frame"]) for row in events}), 21)
        counts = Counter((row["vehicle"], row["direction"]) for row in events)
        self.assertEqual(
            tuple(counts[vehicle, direction]
                  for vehicle in ("car", "motorcycle", "truck") for direction in ("L-R", "R-L")),
            (0, 7, 9, 2, 1, 2),
        )
        self.assertEqual([row["review_id"] for row in events if row["vehicle"] == "truck"],
                         ["L06", "L12", "L22"])

    def test_scope_and_manual_pair_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.csv"
            path.write_text(
                "chieu_di,gt_id,gt_khung_xac_nhan,demo_id,demo_khung_xac_nhan,"
                "loai_xe_xac_nhan,demo_loai_yolo,xac_nhan_ghep,ket_qua_ghep\n"
                "Trái → phải,1,500,9,500,Ô tô con,car,Cùng xe,Ghép chưa chắc chắn\n"
                "Trái → phải,2,500,10,507,Ô tô con,car,Cùng xe,Ghép chưa chắc chắn\n"
                "Phải → trái,3,400,11,401,Loại khác,motorcycle,Cùng xe,Ghép chưa chắc chắn\n"
                "Phải → trái,4,401,12,402,Xe máy,motorcycle,Khác xe,Đã ghép theo thời gian và vị trí\n",
                encoding="utf-8-sig",
            )
            review = load_reviewed_counts(path)
            self.assertEqual(review.reference, {("car", "L-R"): 2, ("motorcycle", "R-L"): 1})
            self.assertEqual(review.demo, {("car", "L-R"): 1, ("motorcycle", "R-L"): 2})
            self.assertEqual(review.matched, 1)


if __name__ == "__main__":
    unittest.main()
