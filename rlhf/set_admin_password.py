#!/usr/bin/env python
"""
set_admin_password.py -- GUI tool for setting the CAISE-ASR admin panel password.

The password is stored as a PBKDF2-HMAC-SHA256 hash in
_outputs/admin_password.json (or ADMIN_PASSWORD_CFG). Keep this runtime
credential configuration out of Git.

Usage:
    py set_admin_password.py
    py set_admin_password.py --config /custom/path/admin_password.json
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from utils.admin_password import MIN_PASSWORD_LENGTH, password_config_path, save_password_config

_DEFAULT_CFG = password_config_path()
_MIN_LENGTH = MIN_PASSWORD_LENGTH


def _save(cfg_path: Path, password: str) -> None:
    save_password_config(cfg_path, password)


class _App(tk.Tk):
    def __init__(self, cfg_path: Path) -> None:
        super().__init__()
        self._cfg_path = cfg_path

        self.title("CAISE-ASR — Set Admin Password")
        self.resizable(False, False)
        self._build()
        self._center()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        pad = {"padx": 16, "pady": 6}

        tk.Label(self, text="Set Admin Password",
                 font=("Segoe UI", 13, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(18, 4), padx=16)

        tk.Label(self, text=f"Plik: {self._cfg_path}",
                 font=("Segoe UI", 8), fg="#666").grid(
            row=1, column=0, columnspan=2, padx=16, pady=(0, 10))

        tk.Label(self, text="New password:", anchor="w").grid(
            row=2, column=0, sticky="w", **pad)
        self._pw1 = tk.Entry(self, show="●", width=30)
        self._pw1.grid(row=2, column=1, **pad)
        self._pw1.focus_set()

        tk.Label(self, text="Confirm password:", anchor="w").grid(
            row=3, column=0, sticky="w", **pad)
        self._pw2 = tk.Entry(self, show="●", width=30)
        self._pw2.grid(row=3, column=1, **pad)
        self._pw2.bind("<Return>", lambda _: self._submit())

        self._status = tk.Label(self, text="", fg="red",
                                font=("Segoe UI", 9), wraplength=340)
        self._status.grid(row=4, column=0, columnspan=2, padx=16, pady=(2, 0))

        self._btn = tk.Button(
            self, text="Set Password", width=20,
            command=self._submit,
            bg="#2563eb", fg="white",
            activebackground="#1d4ed8", activeforeground="white",
            relief="flat", pady=6,
        )
        self._btn.grid(row=5, column=0, columnspan=2, pady=(10, 18))

    def _center(self) -> None:
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

    def _submit(self) -> None:
        pw1 = self._pw1.get()
        pw2 = self._pw2.get()

        if len(pw1) < _MIN_LENGTH:
            self._status.config(
                text=f"Password must be at least {_MIN_LENGTH} characters.", fg="red")
            return
        if pw1 != pw2:
            self._status.config(text="Passwords do not match.", fg="red")
            self._pw2.delete(0, tk.END)
            self._pw2.focus_set()
            return

        try:
            _save(self._cfg_path, pw1)
        except (OSError, ValueError) as exc:
            self._status.config(text=f"Write error: {exc}", fg="red")
            return

        # Clear fields immediately so the password is not visible in memory
        self._pw1.delete(0, tk.END)
        self._pw2.delete(0, tk.END)

        messagebox.showinfo(
            "Password set",
            f"Password saved to:\n{self._cfg_path}\n\n"
            "Existing admin sessions are invalidated. Keep this file out of Git.",
        )
        self.destroy()


def main() -> int:
    p = argparse.ArgumentParser(
        description="GUI tool for setting the CAISE-ASR admin password"
    )
    p.add_argument(
        "--config", default=str(_DEFAULT_CFG),
        help=f"Path to admin_password.json (default: {_DEFAULT_CFG})"
    )
    args = p.parse_args()

    app = _App(Path(args.config))
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

