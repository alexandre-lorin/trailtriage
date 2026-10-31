"""Run a genuine YOLO scan while recording and rejecting outbound connections."""

import socket
from pathlib import Path

from trailtriage.config import Config
from trailtriage.pipeline import scan

attempts = []


def reject_connection(self, address):
    attempts.append(address)
    raise AssertionError(f"Outbound connection forbidden: {address}")


socket.socket.connect = reject_connection
config = Config()
config.detection.model = "models/yolo11n.pt"
config.detection.device = "cpu"
config.detection.batch_size = 4
first = scan(Path("test_photos"), Path("output/offline-verified"), config)
second = scan(Path("test_photos"), Path("output/offline-verified"), config)
assert second["inferred"] == 0
assert not attempts, attempts
print({"first": first, "resume": second, "outbound_attempts": len(attempts)})
