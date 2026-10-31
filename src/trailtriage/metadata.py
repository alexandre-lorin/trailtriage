from datetime import datetime
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageOps


def extract(path: Path) -> dict:
    start = perf_counter()
    stat = path.stat()
    with Image.open(path) as source:
        exif = source.getexif()
        # DateTimeOriginal is commonly inside the EXIF sub-IFD.
        nested = exif.get_ifd(34665) if 34665 in exif else {}
        raw_time = nested.get(36867) or exif.get(36867) or exif.get(306)
        timestamp = stat.st_mtime
        timestamp_source = "mtime"
        if raw_time:
            try:
                timestamp = datetime.strptime(str(raw_time), "%Y:%m:%d %H:%M:%S").timestamp()
                timestamp_source = "exif"
            except ValueError:
                pass
        image = ImageOps.exif_transpose(source)
        image.load()  # Validate decoding, not only the header.
        width, height = image.size
    return {
        "original_path": str(path),
        "filename": path.name,
        "timestamp": timestamp,
        "timestamp_source": timestamp_source,
        "width": width,
        "height": height,
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "processing_time_ms": (perf_counter() - start) * 1000,
    }
