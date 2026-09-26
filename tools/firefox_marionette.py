#!/usr/bin/env python3
"""Minimal Firefox Marionette client: navigate, act, screenshot.

Marionette is Firefox's own automation protocol -- a length-prefixed JSON socket
that needs no driver binary and no geckodriver, which matters here because the
box has no ImageMagick, no ffmpeg and no working `firefox --screenshot` (headless
software rendering has no framebuffer for it to grab).

    ff = Firefox(width=1280, height=860)
    ff.open("http://localhost:8001")
    ff.shot("out.png")
    ff.close()
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time

PROFILE_PREFS = """
user_pref("marionette.port", 0);
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("toolkit.telemetry.enabled", false);
user_pref("app.update.enabled", false);
user_pref("extensions.autoDisableScopes", 15);
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("dom.disable_beforeunload", true);
"""


class MarionetteError(RuntimeError):
    pass


class Firefox:
    def __init__(self, width: int = 1280, height: int = 860, port: int = 2828, binary: str = "firefox"):
        self.width, self.height = width, height
        self.port = port
        self.binary = binary
        self.proc: subprocess.Popen | None = None
        self.sock: socket.socket | None = None
        self.msg_id = 0
        self.tmp = tempfile.mkdtemp(prefix="ffprof-")
        self.screens = 0

    # --- lifecycle ---------------------------------------------------------

    def start(self, timeout: float = 60.0) -> "Firefox":
        prefs = os.path.join(self.tmp, "user.js")
        with open(prefs, "w") as fh:
            fh.write(PROFILE_PREFS)
        cmd = [
            self.binary, "--headless", "--no-remote", "--profile", self.tmp,
            "--marionette", "--window-size", f"{self.width},{self.height}",
            "about:blank",
        ]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env={**os.environ, "MOZ_HEADLESS": "1", "MOZ_HEADLESS_WIDTH": str(self.width),
                 "MOZ_HEADLESS_HEIGHT": str(self.height), "HOME": self.tmp},
        )
        self.sock = self._connect(timeout)
        self._read_packet()  # handshake
        self.call("WebDriver:NewSession", {"capabilities": {"alwaysMatch": {}}})
        self.call("WebDriver:SetWindowRect", {"width": self.width, "height": self.height,
                                              "x": 0, "y": 0})
        self.call("WebDriver:SetTimeouts", {"implicit": 0, "pageLoad": 60000, "script": 30000})
        return self

    def _connect(self, timeout: float) -> socket.socket:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise MarionetteError(f"firefox exited with {self.proc.returncode}")
            try:
                s = socket.create_connection(("127.0.0.1", self.port), timeout=5)
                s.settimeout(180)
                return s
            except OSError:
                time.sleep(0.3)
        raise MarionetteError("marionette never opened its port")

    def close(self) -> None:
        try:
            if self.sock:
                self.sock.close()
        except OSError:
            pass
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def __enter__(self) -> "Firefox":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # --- protocol ----------------------------------------------------------

    def _read_packet(self):
        assert self.sock
        head = b""
        while b":" not in head:
            chunk = self.sock.recv(1)
            if not chunk:
                raise MarionetteError("marionette closed the socket")
            head += chunk
        length = int(head[:-1])
        buf = b""
        while len(buf) < length:
            buf += self.sock.recv(length - len(buf))
        return json.loads(buf.decode("utf-8"))

    def call(self, name: str, params: dict | None = None):
        assert self.sock
        self.msg_id += 1
        payload = json.dumps([0, self.msg_id, name, params or {}]).encode("utf-8")
        self.sock.sendall(str(len(payload)).encode("ascii") + b":" + payload)
        while True:
            msg = self._read_packet()
            if isinstance(msg, list) and len(msg) == 4 and msg[0] == 1 and msg[1] == self.msg_id:
                if msg[2]:
                    raise MarionetteError(f"{name}: {msg[2]}")
                return msg[3]

    # --- page --------------------------------------------------------------

    def open(self, url: str) -> None:
        self.call("WebDriver:Navigate", {"url": url})

    def script(self, body: str, args: list | None = None):
        res = self.call("WebDriver:ExecuteScript",
                        {"script": body, "args": args or [], "newSandbox": False})
        return res.get("value") if isinstance(res, dict) else res

    def async_script(self, body: str, args: list | None = None):
        res = self.call("WebDriver:ExecuteAsyncScript",
                        {"script": body, "args": args or [], "newSandbox": False})
        return res.get("value") if isinstance(res, dict) else res

    def shot_raw(self, full: bool = False) -> bytes:
        res = self.call("WebDriver:TakeScreenshot",
                        {"full": full, "hash": False, "scroll": False,
                         "id": None, "highlights": []})
        data = res.get("value") if isinstance(res, dict) else res
        self.screens += 1
        return base64.b64decode(data)

    def shot(self, path: str, full: bool = False) -> str:
        raw = self.shot_raw(full)
        with open(path, "wb") as fh:
            fh.write(raw)
        return path

    ELEMENT_KEY = "element-6066-11e4-a52e-4f735466cecf"

    def find(self, css: str) -> str:
        """Return the element uuid for a CSS selector.

        Marionette hands back a web-element reference map, but every command that
        takes an `id` wants the bare uuid, so unwrap it once here.
        """
        res = self.call("WebDriver:FindElement", {"using": "css selector", "value": css,
                                                  "id": None})
        ref = res["value"] if isinstance(res, dict) and "value" in res else res
        if isinstance(ref, dict):
            return ref[self.ELEMENT_KEY]
        return ref

    def click(self, css: str) -> None:
        self.call("WebDriver:ElementClick", {"id": self.find(css)})

    def send_keys(self, css: str, text: str) -> None:
        eid = self.find(css)
        self.call("WebDriver:ElementClear", {"id": eid})
        self.call("WebDriver:ElementSendKeys", {"id": eid, "text": text, "value": list(text)})

    def press_enter(self, css: str | None = None) -> None:
        eid = self.find(css) if css else None
        actions = [{
            "type": "key",
            "id": eid,
            "actions": [{"type": "keyDown", "value": "\ue007"}, {"type": "keyUp", "value": "\ue007"}],
        }]
        self.call("WebDriver:PerformActions", {"actions": actions})
        self.call("WebDriver:ReleaseActions", {})

    def size(self) -> tuple[int, int]:
        return self.width, self.height
