import unittest

import numpy as np

import main


def scroll_hand(outer_flexed=False):
    """Synthetic hand with folded inner fingers and movable ring/little fingers."""
    p = np.zeros((21, 2), dtype=np.float64)
    p[0] = (0, 40)
    p[1:5] = [(8, 32), (15, 25), (20, 20), (24, 18)]
    p[5], p[9], p[13], p[17] = (-18, 0), (-6, 0), (6, 0), (18, 2)

    # Index/middle are bent but deliberately remain outside the compact-fist
    # radius. This is the held scroll pose, not an Alt+Tab fist.
    p[6:9] = [(-18, -12), (-30, -12), (-38, -4)]
    p[10:13] = [(-6, -14), (-18, -14), (-26, -5)]

    if outer_flexed:
        p[14:17] = [(6, -14), (17, -14), (20, -4)]
        p[18:21] = [(18, -11), (28, -10), (31, -1)]
    else:
        p[14:17] = [(6, -14), (6, -27), (6, -39)]
        p[18:21] = [(18, -11), (18, -21), (18, -30)]
    return p


def fist_hand():
    p = np.zeros((21, 2), dtype=np.float64)
    p[0] = (0, 40)
    p[1:5] = [(8, 30), (12, 20), (8, 12), (2, 9)]
    for mcp, pip, dip, tip, x in (
        (5, 6, 7, 8, -18), (9, 10, 11, 12, -6),
        (13, 14, 15, 16, 6), (17, 18, 19, 20, 18),
    ):
        p[mcp] = (x, 0)
        p[pip] = (x, -12)
        p[dip] = (x + 9, -12)
        p[tip] = (x + 11, -2)
    return p


class ScrollGestureTests(unittest.TestCase):
    def test_extension_changes_scroll_metric(self):
        extended = main.scroll_finger_position(scroll_hand(False))
        flexed = main.scroll_finger_position(scroll_hand(True))
        self.assertGreater(extended - flexed, 0.20)

    def test_whole_hand_translation_does_not_change_metric(self):
        p = scroll_hand(False)
        moved = p + np.array([137.0, -83.0])
        self.assertAlmostEqual(main.scroll_finger_position(p),
                               main.scroll_finger_position(moved), places=12)

    def test_scroll_stays_active_at_both_stroke_ends(self):
        for p in (scroll_hand(False), scroll_hand(True)):
            self.assertTrue(main.is_scroll_gesture(p))
            self.assertFalse(main.is_fist(p))

    def test_fist_remains_distinct(self):
        p = fist_hand()
        self.assertTrue(main.is_fist(p))
        self.assertFalse(main.is_scroll_gesture(p))


if __name__ == "__main__":
    unittest.main()
