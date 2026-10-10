from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "cmapss"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print("Downloading NASA PCoE C-MAPSS archive...")
    request = urllib.request.Request(URL, headers={"User-Agent": "engine-health-lab/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        nested = [m for m in archive.namelist() if Path(m).name.lower() == "cmapssdata.zip"]
        if nested:
            payload = archive.read(nested[0])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [m for m in archive.namelist() if Path(m).name.startswith(("train_FD", "test_FD", "RUL_FD"))]
        if not members:
            raise RuntimeError(f"Archive did not contain expected C-MAPSS trajectory files: {archive.namelist()[:8]}")
        for member in members:
            target = DEST / Path(member).name
            target.write_bytes(archive.read(member))
    print(f"Extracted {len(members)} trajectory/target files to {DEST}")


if __name__ == "__main__":
    main()
