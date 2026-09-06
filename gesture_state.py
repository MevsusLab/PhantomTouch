from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PauseUpdate:
    paused: bool
    changed: bool
    hold_progress: float
    reason: str


class PauseController:
    def __init__(self, hold_seconds: float, debounce_seconds: float,
                 auto_pause_seconds: float) -> None:
        self.hold_seconds = hold_seconds
        self.debounce_seconds = debounce_seconds
        self.auto_pause_seconds = auto_pause_seconds
        self.paused = False
        self.reason = "running"
        self._palm_since = None
        self._no_hand_since = None
        self._last_toggle = float("-inf")
        self._must_release = False

    def update(self, now: float, hand_present: bool, open_palm: bool,
               emergency_paused: bool = False) -> PauseUpdate:
        changed = False
        if emergency_paused:
            if not self.paused or self.reason != "emergency pause":
                changed = not self.paused
                self.paused = True
                self.reason = "emergency pause"
            self._palm_since = None
            return PauseUpdate(self.paused, changed, 0.0, self.reason)

        if hand_present:
            self._no_hand_since = None
        elif self._no_hand_since is None:
            self._no_hand_since = now
        elif (self.auto_pause_seconds > 0 and not self.paused and
              now - self._no_hand_since >= self.auto_pause_seconds):
            self.paused = True
            self.reason = "paused: hand lost"
            self._must_release = False
            self._palm_since = None
            changed = True

        if not open_palm:
            self._palm_since = None
            self._must_release = False
            return PauseUpdate(self.paused, changed, 0.0, self.reason)

        if self._must_release or now - self._last_toggle < self.debounce_seconds:
            return PauseUpdate(self.paused, changed, 0.0, self.reason)

        if self._palm_since is None:
            self._palm_since = now
        progress = min(1.0, max(0.0, (now - self._palm_since) / self.hold_seconds))
        if progress >= 1.0:
            self.paused = not self.paused
            self.reason = "paused" if self.paused else "running"
            self._last_toggle = now
            self._palm_since = None
            self._must_release = True
            changed = True
            progress = 0.0
        return PauseUpdate(self.paused, changed, progress, self.reason)
