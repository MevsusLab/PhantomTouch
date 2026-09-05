"""Phantom desktop control panel."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

import main as gesture_engine
from phantom_config import PhantomSettings, load_settings, save_settings

NAVY = "#07111f"
PANEL = "#0d1b2d"
PANEL_2 = "#12243a"
YELLOW = "#ffd21c"
TEXT = "#e8edf2"
MUTED = "#8292a6"
DANGER = "#ff4d52"


class PhantomApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Phantom")
        self.geometry("940x700")
        self.minsize(820, 640)
        self.configure(bg=NAVY)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.worker = None
        self.stop_event = threading.Event()
        self.vars = {}
        self._style()
        self._build()
        try:
            self._set_values(load_settings())
        except ValueError as error:
            messagebox.showwarning("Phantom configuration", "%s\nDefaults were loaded." % error)
            self._set_values(PhantomSettings())

    def _style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Phantom.Horizontal.TScale", background=PANEL, troughcolor=NAVY,
                        borderwidth=0, lightcolor=YELLOW, darkcolor=YELLOW)
        style.configure("Phantom.TCheckbutton", background=PANEL, foreground=TEXT,
                        font=("Segoe UI", 10), indicatorcolor=NAVY)
        style.map("Phantom.TCheckbutton", background=[("active", PANEL)],
                  indicatorcolor=[("selected", YELLOW)])
        style.configure("Phantom.TCombobox", fieldbackground=NAVY, background=PANEL_2,
                        foreground=TEXT, arrowcolor=YELLOW, borderwidth=0)

    def _build(self) -> None:
        header = tk.Canvas(self, height=190, bg=NAVY, highlightthickness=0)
        header.pack(fill="x", padx=38, pady=(22, 0))
        header.bind("<Configure>", self._draw_logo)

        content = tk.Frame(self, bg=NAVY)
        content.pack(fill="both", expand=True, padx=42, pady=(0, 30))
        left = tk.Frame(content, bg=PANEL, highlightbackground="#1d3450", highlightthickness=1)
        right = tk.Frame(content, bg=PANEL, highlightbackground="#1d3450", highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self._section_title(left, "POINTER // MOTION")
        self._slider(left, "Cursor responsiveness", "cursor_responsiveness", 0.05, 1.0)
        self._slider(left, "Cursor smoothing", "cursor_smoothing", 0.05, 1.0)
        self._slider(left, "Scroll sensitivity", "scroll_sensitivity", 1.0, 100.0)
        self._slider(left, "Scroll dead zone", "scroll_dead_zone", 0.0, 1.0)

        self._section_title(right, "SYSTEM // CONTROL")
        row = tk.Frame(right, bg=PANEL)
        row.pack(fill="x", padx=22, pady=(10, 18))
        tk.Label(row, text="CAMERA INDEX", bg=PANEL, fg=MUTED,
                 font=("Segoe UI Semibold", 9)).pack(anchor="w")
        self.vars["camera_index"] = tk.StringVar()
        ttk.Combobox(row, textvariable=self.vars["camera_index"], values=tuple(range(8)),
                     state="readonly", style="Phantom.TCombobox").pack(fill="x", pady=(7, 0))
        self._check(right, "Fist gesture triggers Alt+Tab", "fist_alt_tab")
        self._check(right, "Show camera debug overlay", "debug_overlay")

        self.status_var = tk.StringVar(value="●  OFFLINE")
        self.status = tk.Label(right, textvariable=self.status_var, bg=PANEL_2, fg=MUTED,
                               anchor="w", padx=14, pady=11,
                               font=("Consolas", 10, "bold"))
        self.status.pack(fill="x", padx=22, pady=(22, 12))

        buttons = tk.Frame(right, bg=PANEL)
        buttons.pack(fill="x", padx=22, pady=(5, 10))
        self.start_button = self._button(buttons, "START", self.start, YELLOW, NAVY)
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.stop_button = self._button(buttons, "STOP", self.stop, DANGER, TEXT)
        self.stop_button.pack(side="left", fill="x", expand=True, padx=(5, 0))
        utility = tk.Frame(right, bg=PANEL)
        utility.pack(fill="x", padx=22, pady=5)
        self._button(utility, "SAVE SETTINGS", self.save, PANEL_2, TEXT).pack(
            side="left", fill="x", expand=True, padx=(0, 5))
        self._button(utility, "RESET", self.reset, PANEL_2, MUTED).pack(
            side="left", fill="x", expand=True, padx=(5, 0))

    def _draw_logo(self, event) -> None:
        c = event.widget
        c.delete("all")
        w = event.width
        c.create_polygon(0, 18, 18, 0, w, 0, w - 18, 18, fill=PANEL_2)
        c.create_text(w / 2, 75, text="PHANTOM", fill=YELLOW,
                      font=("Impact", 58), anchor="center")
        # Irregular paint runs tied visually to the word's baseline.
        for x, length, width in ((w*.24, 34, 9), (w*.31, 17, 5), (w*.42, 44, 8),
                                 (w*.51, 22, 6), (w*.62, 38, 10), (w*.70, 20, 5)):
            c.create_polygon(x-width, 112, x+width, 112, x+width*.45, 112+length,
                             x, 120+length, x-width*.45, 112+length, fill=YELLOW)
            c.create_oval(x-width*.45, 116+length, x+width*.45, 122+length,
                          fill=YELLOW, outline=YELLOW)
        c.create_line(w*.18, 111, w*.76, 111, fill=YELLOW, width=5)
        c.create_text(w / 2, 169, text="GESTURE CONTROL SYSTEM  //  WINDOWS",
                      fill=MUTED, font=("Consolas", 10, "bold"))

    def _section_title(self, parent, text) -> None:
        tk.Label(parent, text=text, bg=YELLOW, fg=NAVY, anchor="w", padx=14, pady=9,
                 font=("Consolas", 11, "bold")).pack(fill="x")

    def _slider(self, parent, label, key, low, high) -> None:
        block = tk.Frame(parent, bg=PANEL)
        block.pack(fill="x", padx=22, pady=10)
        top = tk.Frame(block, bg=PANEL)
        top.pack(fill="x")
        tk.Label(top, text=label.upper(), bg=PANEL, fg=MUTED,
                 font=("Segoe UI Semibold", 9)).pack(side="left")
        variable = tk.DoubleVar()
        self.vars[key] = variable
        value = tk.Label(top, bg=PANEL, fg=YELLOW, width=6, anchor="e",
                         font=("Consolas", 10, "bold"))
        value.pack(side="right")
        variable.trace_add("write", lambda *_: value.configure(text="%.2f" % variable.get()))
        ttk.Scale(block, from_=low, to=high, variable=variable,
                  style="Phantom.Horizontal.TScale").pack(fill="x", pady=(7, 0))

    def _check(self, parent, label, key) -> None:
        variable = tk.BooleanVar()
        self.vars[key] = variable
        ttk.Checkbutton(parent, text=label, variable=variable,
                        style="Phantom.TCheckbutton").pack(anchor="w", padx=22, pady=9)

    def _button(self, parent, text, command, bg, fg):
        return tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                         activebackground=fg, activeforeground=bg, relief="flat",
                         bd=0, padx=10, pady=10, cursor="hand2",
                         font=("Consolas", 10, "bold"))

    def _settings(self) -> PhantomSettings:
        return PhantomSettings(
            cursor_responsiveness=float(self.vars["cursor_responsiveness"].get()),
            cursor_smoothing=float(self.vars["cursor_smoothing"].get()),
            scroll_sensitivity=float(self.vars["scroll_sensitivity"].get()),
            scroll_dead_zone=float(self.vars["scroll_dead_zone"].get()),
            camera_index=int(self.vars["camera_index"].get()),
            fist_alt_tab=bool(self.vars["fist_alt_tab"].get()),
            debug_overlay=bool(self.vars["debug_overlay"].get()))

    def _set_values(self, settings) -> None:
        for key, value in settings.to_dict().items():
            self.vars[key].set(value)

    def save(self) -> bool:
        try:
            save_settings(self._settings())
            self._status("●  SETTINGS SAVED", YELLOW)
            return True
        except (ValueError, OSError) as error:
            messagebox.showerror("Cannot save settings", str(error))
            return False

    def reset(self) -> None:
        self._set_values(PhantomSettings())
        self._status("●  DEFAULTS LOADED", MUTED)

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if not self.save():
            return
        self.stop_event.clear()
        settings = self._settings()
        self._status("●  STARTING CAMERA", YELLOW)
        self.worker = threading.Thread(target=self._run_engine, args=(settings,), daemon=True)
        self.worker.start()

    def _run_engine(self, settings) -> None:
        try:
            gesture_engine.main(settings=settings, stop_event=self.stop_event,
                                status_callback=lambda text: self.after(
                                    0, self._status, "●  " + text.upper(), YELLOW))
        except Exception as error:
            self.after(0, messagebox.showerror, "Phantom stopped", str(error))
        finally:
            self.after(0, self._status, "●  OFFLINE", MUTED)

    def stop(self) -> None:
        self.stop_event.set()
        self._status("●  STOPPING", DANGER)

    def _status(self, text, color) -> None:
        self.status_var.set(text)
        self.status.configure(fg=color)

    def _close(self) -> None:
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2.5)
        self.destroy()


if __name__ == "__main__":
    PhantomApp().mainloop()
