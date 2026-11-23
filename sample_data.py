"""Fetch the official sample and model once and verify them before use."""
import hashlib
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from zipfile import ZipFile

ASSETS = Path(__file__).resolve().parent / "assets"
VIDEO = ASSETS / "vntraffic.mp4"
WEIGHTS = ASSETS / "yolo11n.pt"
VIDEO_HASH = "7b4ee8eb6def9717d107d55b71b63fad265358d191c85051fc323c6fab290d57"


def verified(path, digest):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == digest


def save_verified(stream, target, digest):
    with NamedTemporaryFile(dir=ASSETS, delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        with temp_path.open("wb") as output:
            shutil.copyfileobj(stream, output)
        if not verified(temp_path, digest):
            raise ValueError(f"Downloaded file failed checksum: {target.name}")
        temp_path.replace(target)
    finally:
        temp_path.unlink(missing_ok=True)


def download(url, target, digest):
    if verified(target, digest):
        return
    print(f"Downloading {target.name} ...", flush=True)
    request = Request(url, headers={"User-Agent": "ITS-Vehicle-Demo/1.0"})
    with urlopen(request, timeout=120) as stream:
        save_verified(stream, target, digest)


def prepare():
    ASSETS.mkdir(exist_ok=True)
    if not verified(VIDEO, VIDEO_HASH):
        archive = ASSETS / "Vehicle_Tracking.zip"
        download("https://zenodo.org/records/18195750/files/Vehicle_Tracking.zip?download=1",
                 archive, "7fdfb05a02900402670310e4525575c55dfe39e8961e60e3c4533ca3fe16d89f")
        with ZipFile(archive) as dataset, dataset.open("VNTraffic/VNTraffic_Original-video.mp4") as stream:
            save_verified(stream, VIDEO, VIDEO_HASH)
    download("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
             WEIGHTS, "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1")
