#!/usr/bin/env python3
"""Quick motor bus test — pings all motors and enables torque."""

import sys

PORT = "/dev/ttyACM0"
MOTOR_IDS = [1, 2, 3, 4, 5, 6]  # SO-101 follower motors

try:
    import scservo_sdk as scs
except ImportError:
    sys.exit("ERROR: scservo_sdk not found — are you in the rynnvla002 env?")

ph = scs.PacketHandler(0)  # protocol 0 for Feetech SCS
port_handler = scs.PortHandler(PORT)

if not port_handler.openPort():
    sys.exit(f"ERROR: Could not open port {PORT}")
if not port_handler.setBaudRate(1000000):
    sys.exit("ERROR: Could not set baud rate to 1000000")

print(f"Opened {PORT} @ 1000000 baud\n")

# Ping each motor
print("=== Ping ===")
found = []
for mid in MOTOR_IDS:
    model_num, comm, err = ph.ping(port_handler, mid)
    if comm == scs.COMM_SUCCESS:
        print(f"  Motor {mid}: OK  (model={model_num})")
        found.append(mid)
    else:
        print(f"  Motor {mid}: FAIL  ({ph.getTxRxResult(comm)})")

if not found:
    port_handler.closePort()
    sys.exit("\nNo motors found. Check power and cable.")

# Try enabling torque on found motors
print("\n=== Enable Torque ===")
TORQUE_ENABLE_ADDR = 40   # SCS series Torque_Enable register
LOCK_ADDR = 55            # Lock register

for mid in found:
    # Unlock first
    comm, err = ph.write1ByteTxRx(port_handler, mid, LOCK_ADDR, 0)
    # Enable torque
    comm, err = ph.write1ByteTxRx(port_handler, mid, TORQUE_ENABLE_ADDR, 1)
    # Lock
    comm, err = ph.write1ByteTxRx(port_handler, mid, LOCK_ADDR, 1)
    if comm == scs.COMM_SUCCESS:
        print(f"  Motor {mid}: torque ENABLED")
    else:
        print(f"  Motor {mid}: torque FAILED — {ph.getTxRxResult(comm)}")

port_handler.closePort()
print("\nDone.")
