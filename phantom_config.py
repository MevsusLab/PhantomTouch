from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phantom_config.json")


@dataclass(frozen=True)
class PhantomSettings:
    cursor_responsiveness: float = 0.68
    cursor_smoothing: float = 0.38
    scroll_sensitivity: float = 32.0
    scroll_dead_zone: float = 0.12
    camera_index: int = 0
    fist_alt_tab: bool = True
    debug_overlay: bool = True

    def __post_init__(self) -> None:
        limits = {
            "cursor_responsiveness": (0.05, 1.0),
            "cursor_smoothing": (0.05, 1.0),
            "scroll_sensitivity": (1.0, 100.0),
            "scroll_dead_zone": (0.0, 1.0),
            "camera_index": (0, 20),
        }
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("%s must be numeric" % name)
            if not low <= value <= high:
                raise ValueError("%s must be between %s and %s" % (name, low, high))
        if not isinstance(self.camera_index, int):
            raise ValueError("camera_index must be an integer")
        for name in ("fist_alt_tab", "debug_overlay"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError("%s must be true or false" % name)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PhantomSettings":
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(path: str = CONFIG_PATH) -> PhantomSettings:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("configuration root must be an object")
        return PhantomSettings.from_mapping(data)
    except FileNotFoundError:
        return PhantomSettings()
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid Phantom configuration: %s" % error) from error


def save_settings(settings: PhantomSettings, path: str = CONFIG_PATH) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(settings.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
