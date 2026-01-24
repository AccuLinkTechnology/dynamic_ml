import json
import socket
import time
from tower_motor_controller import TowerMotor
import inclinometer

PI_PORT = 5005  # must match Jetson

def device_setup():
    ##AZIMUTH and Elevation should be starting from 10, 10.
    starting_pos_x, starting_pos_y = 10, 10
    inc = inclinometer.Inclinometer('/dev/ttyUSB0', 38400)
    x_motor = TowerMotor(id=14, SENS_PIN=18, PUL_PIN=5, DIR_PIN=6, microsteps=1)
    y_motor = TowerMotor(id=15, SENS_PIN=17, PUL_PIN=24, DIR_PIN=25, microsteps=1)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PI_PORT))
    print(f"Listening UDP on :{PI_PORT}")

    last_t = None

    while True:
        data, addr = sock.recvfrom(4096)
        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception as e:
            print("Bad packet:", e)
            continue

        now = time.time()
        dt = 0.0 if last_t is None else (now - last_t)
        last_t = now

        # Accept either key style:
        # Jetson inference sender: "az", "el"
        # Older code: "delta_az", "delta_el"
        delta_az = float(msg.get("az", msg.get("delta_az", 0.0)))
        delta_el = float(msg.get("el", msg.get("delta_el", 0.0)))

        # TEMP: print raw keys once if you want to confirm
        # print("RAW:", msg)

        print(f"From {addr[0]} | ?az={delta_az:+.3f} ?el={delta_el:+.3f} | dt={dt*1000:.1f} ms")

if __name__ == "__main__":
    main()
