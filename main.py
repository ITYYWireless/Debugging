# ==== pc_control.py  (runs ON the MaxArm board) ====

from BusServo import BusServo
from espmax import ESPMax
from SuctionNozzle import SuctionNozzle
import sys, ujson, time

bus = BusServo()
arm = ESPMax(bus)
nozzle = SuctionNozzle()

# Optional: go to a safe home pose at startup
arm.go_home(2000)
time.sleep_ms(2200)

print("READY")  # PC can read this

def handle_command(cmd):
    c = cmd.get("cmd")

    if c == "home":
        t = int(cmd.get("time", 2000))
        arm.go_home(t)
        return {"ok": True}

    if c == "move_xyz":
        x = float(cmd.get("x", 0))
        y = float(cmd.get("y", -150))
        z = float(cmd.get("z", 150))
        t = int(cmd.get("time", 1000))

        # clamp into a safe range: z <= 225, radius >= 50 (from espmax.py)
        # this avoids math errors & collisions
        if z > 225:
            z = 225

        if (x ** 2 + y ** 2) ** 0.5 < 50:
            return {"ok": False, "err": "too_close"}

        ok = arm.set_position((x, y, z), t)
        return {"ok": bool(ok)}

    if c == "pump":
        # control suction, e.g. {"cmd": "pump", "on": true}
        on = bool(cmd.get("on", False))
        if on:
            nozzle.on()
        else:
            nozzle.off()
        return {"ok": True}

    return {"ok": False, "err": "unknown_cmd"}

# Main loop: read JSON lines from USB, execute, print JSON reply
while True:
    line = sys.stdin.readline()
    if not line:
        continue
    line = line.strip()
    if not line:
        continue

    try:
        cmd = ujson.loads(line)
    except Exception as e:
        print(ujson.dumps({"ok": False, "err": "bad_json"}))
        continue

    try:
        result = handle_command(cmd)
    except Exception as e:
        result = {"ok": False, "err": "exception"}

    print(ujson.dumps(result))

