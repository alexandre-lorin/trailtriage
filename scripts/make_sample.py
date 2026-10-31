"""Create 30 artificial fixtures; drawings do not validate detector accuracy."""

from pathlib import Path

from PIL import Image, ImageDraw

root = Path("test_photos")
root.mkdir(exist_ok=True)
for i in range(30):
    image = Image.new("RGB", (640, 480), (70 + i % 5, 100, 60))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 320, 640, 480), fill=(130, 100, 80))
    if 5 <= i < 15:
        draw.ellipse((240, 120, 290, 170), fill="black")
        draw.rectangle((240, 170, 290, 300), fill="black")
    elif 15 <= i < 25:
        draw.rectangle((180, 220, 380, 300), fill="gray")
        draw.ellipse((210, 280, 250, 320), fill="black")
        draw.ellipse((320, 280, 360, 320), fill="black")
    elif i >= 25:
        draw.ellipse((200, 250, 300, 290), fill="brown")
    exif = Image.Exif()
    exif[36867] = f"2026:01:02 12:0{i // 10}:0{i % 10}"
    image.save(root / f"fixture_{i:02}.jpg", exif=exif)
(root / "corrupt.jpg").write_bytes(b"broken JPEG fixture")
print(root.resolve())
