import hashlib
import os
from pathlib import Path

from .config import Detection


def model_identity(config: Detection) -> str:
    path = Path(config.model).expanduser().resolve()
    if not path.is_file() or path.suffix != ".pt":
        raise ValueError("Provide a local .pt model with --model; automatic downloads are disabled")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Detector:
    def __init__(self, config: Detection):
        model_identity(config)
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["YOLO_VERBOSE"] = "false"
        try:
            import torch
            from ultralytics import YOLO, settings
            from ultralytics.engine.results import Results
        except ImportError as exc:
            raise RuntimeError(
                "Install detection dependencies: pip install -e '.[detection]'"
            ) from exc
        settings.update({"sync": False})
        self.result_type = Results
        self.torch = torch
        self.device = (
            ("0" if torch.cuda.is_available() else "cpu")
            if (config.device == "auto")
            else config.device
        )
        if self.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")
        self.batch_size = (
            (8 if self.device != "cpu" else 2)
            if (config.batch_size == "auto")
            else int(config.batch_size)
        )
        self.config = config
        self.model = YOLO(str(Path(config.model).expanduser().resolve()))
        print(f"Device: {self.device}; model: {config.model}; batch: {self.batch_size}")

    def predict(self, images: list) -> list[list[dict]]:
        try:
            results = self.model.predict(
                images,
                device=self.device,
                conf=self.config.confidence_threshold,
                imgsz=640,
                verbose=False,
                save=False,
            )
        except self.torch.cuda.OutOfMemoryError:
            if len(images) == 1:
                raise
            self.torch.cuda.empty_cache()
            middle = len(images) // 2
            return self.predict(images[:middle]) + self.predict(images[middle:])
        detections = []
        for result in results:
            if not isinstance(result, self.result_type) or result.boxes is None:
                raise RuntimeError("Model must be an object-detection model")
            boxes = result.boxes.cpu()
            detections.append(
                [
                    {
                        "class": result.names[int(cls)],
                        "confidence": float(conf),
                        "box": [float(v) for v in box],
                    }
                    for cls, conf, box in zip(boxes.cls, boxes.conf, boxes.xyxy)
                ]
            )
        return detections
