# ==== maxarm_pc.py  (runs on your PC) ====

import serial
import json
import time

class MaxArmClient:
    def __init__(self, port, baud=115200, timeout=1):
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        # optional: read the initial "READY" line from the board
        time.sleep(0.5)
        line = self.ser.readline().decode(errors="ignore").strip()
        if line:
            print("MaxArm says:", line)

    def _send(self, payload: dict) -> dict:
        data = json.dumps(payload) + "\n"
        self.ser.write(data.encode("utf-8"))
        reply = self.ser.readline().decode(errors="ignore").strip()
        if not reply:
            return {"ok": False, "err": "no_reply"}
        try:
            return json.loads(reply)
        except Exception:
            return {"ok": False, "err": "bad_reply", "raw": reply}

    def home(self, time_ms=2000):
        return self._send({"cmd": "home", "time": int(time_ms)})

    def move_xyz(self, x, y, z, time_ms=1000):
        return self._send({
            "cmd": "move_xyz",
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "time": int(time_ms),
        })

    def pump(self, on: bool):
        return self._send({"cmd": "pump", "on": bool(on)})

    def close(self):
        self.ser.close()
