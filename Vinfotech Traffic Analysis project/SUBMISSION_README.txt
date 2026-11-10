VINfotech Traffic Intersection Analytics
=======================================

Model: YOLO11s
Tracker: BoT-SORT
Inference size: 1280
Processing stride: 2
Input video: Traffic_video (2).mp4

Final provisional results:
- Matched vehicle movements: 261
- Unmatched entry candidates: 71
- Gate events: 696
- Tracked vehicle IDs: 1170

Final geometry:
- South IN / OUT: Refined Trial D
- North OUT: Refined Trial N2
- West IN: Refined Trial W1
- West OUT: Original
- North IN: Original
- East IN / OUT: Original

FOLDER STRUCTURE

Code/
    traffic_pipeline.py
    geometry_bellevue.json
    run_with_paths.py
    requirements.txt

Data/
    all_vehicle_trajectories.csv

Results/
    Statistics, turning matrix, queues,
    waiting times, figures and diagnostics

Video/
    annotated_traffic_desktop.mp4
    (provided separately when necessary)

ORIGINAL VIDEO

Keep the original filename:
Traffic_video (2).mp4

Place the original video directly in the main
folder containing Code, Data and Results.

HOW TO RUN

1. Install required Python dependencies:

   python -m pip install -r Code/requirements.txt

2. Reproduce statistics without rerunning YOLO:

   python Code/run_with_paths.py

3. Output files will appear in:

   Reproduced_Results/

NOTE

All counts are provisional. The pipeline
uses saved YOLO/BoT-SORT trajectories.
Waiting times are observed stationary durations,
not full intersection delay measurements.
South queue estimates are unavailable.
