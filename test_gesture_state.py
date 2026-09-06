import unittest

from gesture_state import PauseController


class PauseControllerTests(unittest.TestCase):
    def test_open_palm_requires_hold_and_release_between_toggles(self):
        state = PauseController(1.0, 0.5, 3.0)
        self.assertFalse(state.update(0.0, True, True).paused)
        self.assertFalse(state.update(0.9, True, True).paused)
        result = state.update(1.0, True, True)
        self.assertTrue(result.paused)
        self.assertTrue(result.changed)
        self.assertTrue(state.update(2.0, True, True).paused)
        state.update(2.1, True, False)
        state.update(2.2, True, True)
        self.assertFalse(state.update(3.2, True, True).paused)

    def test_short_palm_pose_does_not_toggle(self):
        state = PauseController(1.0, 0.5, 3.0)
        state.update(0.0, True, True)
        state.update(0.5, True, False)
        self.assertFalse(state.update(2.0, True, False).paused)

    def test_hand_loss_auto_pauses_and_never_auto_resumes(self):
        state = PauseController(1.0, 0.5, 2.0)
        state.update(0.0, False, False)
        result = state.update(2.0, False, False)
        self.assertTrue(result.paused)
        self.assertEqual(result.reason, "paused: hand lost")
        self.assertTrue(state.update(3.0, True, False).paused)

    def test_emergency_pause(self):
        state = PauseController(1.0, 0.5, 2.0)
        result = state.update(0.0, True, False, True)
        self.assertTrue(result.paused)
        self.assertEqual(result.reason, "emergency pause")


if __name__ == "__main__":
    unittest.main()
