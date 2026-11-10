from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent

PIPELINE = ROOT / "Code" / "traffic_pipeline.py"
GEOMETRY = ROOT / "Code" / "geometry_bellevue.json"
TRACKS = ROOT / "Data" / "all_vehicle_trajectories.csv"
OUTPUT = ROOT / "Reproduced_Results"

VIDEO_NAME = "Traffic_video (2).mp4"

# Supports video in the main submission folder,
# an InputVideo subfolder, or the parent project folder.
VIDEO_CANDIDATES = [
    ROOT / VIDEO_NAME,
    ROOT / "InputVideo" / VIDEO_NAME,
    ROOT.parent / VIDEO_NAME
]

VIDEO = next(
    (p for p in VIDEO_CANDIDATES if p.is_file()),
    None
)

for path in [PIPELINE, GEOMETRY, TRACKS]:
    if not path.is_file():
        raise SystemExit(f"Required file missing: {path}")

if VIDEO is None:
    raise SystemExit(
        f"Original video not found: {VIDEO_NAME}\n"
        "Place it in the same folder as Code, Data and Results."
    )

command = [
    sys.executable,
    str(PIPELINE),
    "--video", str(VIDEO),
    "--geometry", str(GEOMETRY),
    "--out", str(OUTPUT),
    "--csv", str(TRACKS),
    "--model", "yolo11s.pt",
    "--stride", "2",
    "--imgsz", "1280",
    "--skip-video"
]

print("Original video:", VIDEO)
print("Reproducing analysis using saved trajectories...")

subprocess.run(command, check=True)

print("Analysis complete:", OUTPUT)
