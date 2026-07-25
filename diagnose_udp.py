import socket
import time

PORT = 3000
CONTROL_PORT = 3001
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.bind(("0.0.0.0", PORT))
sock.settimeout(0.1)

print(f"Listening for UDP packets on port {PORT}...", flush=True)
print("Sending broadcast heartbeats to auto-register Laptop IP on ESP32-S3...", flush=True)

last_heartbeat_time = 0
last_frame_time = time.time()

try:
    while True:
        now = time.time()
        # Broadcast discovery packet to ESP32-S3
        if now - last_heartbeat_time > 1.0:
            try:
                sock.sendto(b"HB", ("255.255.255.255", CONTROL_PORT))
                last_heartbeat_time = now
            except Exception:
                pass
            
        try:
            data, addr = sock.recvfrom(65535)
            if data != b"HB" and data != b"HB\n":
                print(f"Received {len(data)} bytes from {addr}", flush=True)
                last_frame_time = now
        except socket.timeout:
            if now - last_frame_time > 5.0:
                print("ERROR: No packets received (Timeout 5s).", flush=True)
                # Reset timer to avoid spamming error messages
                last_frame_time = now
except KeyboardInterrupt:
    print("Stopped.", flush=True)
finally:
    sock.close()
