import unittest

from evaluate import Event, associate, crossing_geometry


class EvaluationTests(unittest.TestCase):
    def test_wrong_class_counts_as_mismatch_and_duplicate_stays_unmatched(self):
        truth = Event(10, 62, 2.029, 459.62, "truck", "L-R", "L06")
        whole_truck = Event(19, 63, 2.05, 460, "car", "L-R")
        cab = Event(16, 58, 1.765, 447.94, "car", "L-R")
        rows = associate([truth], [cab, whole_truck])
        self.assertEqual([(gt.track_id if gt else None, demo.track_id, status)
                          for gt, demo, status, _, _ in rows],
                         [(None, 16, "demo_only"), (10, 19, "wrong_class")])

    def test_geometry_and_direction_take_priority_over_predicted_class(self):
        truth = Event(52, 286, 9.383, 331, "truck", "R-L", "L12")
        wrong_class = Event(193, 286, 9.383, 332, "car", "R-L")
        wrong_direction = Event(12, 286, 9.383, 331, "truck", "L-R")
        distant = Event(333, 286, 9.45, 370, "truck", "R-L")
        rows = associate([truth], [wrong_direction, wrong_class, distant])
        paired = [(demo.track_id, status) for gt, demo, status, _, _ in rows if gt]
        self.assertEqual(paired, [(193, "wrong_class")])

    def test_crossing_position_uses_segment_between_observations(self):
        samples = [(1, 585, 329), (2, 580, 331), (4, 575, 335)]
        self.assertEqual(crossing_geometry(samples, 580, "R-L"), (2, 331, 2))


if __name__ == "__main__":
    unittest.main()
