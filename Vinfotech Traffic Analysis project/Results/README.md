# Traffic Intersection Video Analytics — Vinfotech Case Study

    Source: Traffic_video (2).mp4
    Video duration: 608.1 seconds, 1920×1080, 30.0 source FPS.
    Processed approximately 15.0 video FPS with yolo11s.pt and BoT-SORT (appearance ReID disabled).

    ## Outputs
    - annotated_traffic_desktop.mp4: tracked IDs, gates, path trails, and counts (generated when video supplied)
- traffic_pipeline.py: VS Code / terminal runnable Python source
- requirements.txt: Python environment dependencies
    - all_vehicle_trajectories.csv: raw frame, ID, bbox, class, confidence, bottom-center contact point
    - approach_entry_counts.csv: directed crossing candidates per entering approach
    - gate_crossing_events.csv: gate events (raw); gate_crossing_events_deduplicated.csv: conservative candidate dedup
    - turning_matrix_4x4.csv, turning_movements_matched.csv, unmatched_entries.csv
    - stopped_vehicle_episodes.csv, completed_stationary_waits.csv, queue_per_second.csv
    - intersection_summary.csv, results.json, manual_validation_sample.csv, figures and calibration_preview.jpg

    ## Methods and limitations
    The eight geometric gates and queue zones were manually approximated from a reference frame.
    A crossing must be in the specified direction, through the finite gate segment, within a continuous track.
    Turning entries are paired with exits only for the same track ID; movements without confirmed exits remain unmatched, not guessed.
    Queue is the number of tracked near-stationary vehicles in a queue polygon, NOT a measured physical queue length.
    Stationary time is computed only on observed multi-frame slow-motion episodes; censored waits are not treated as zero.
    YOLO does not guarantee perfect detection, and BoT-SORT does not guarantee uninterrupted IDs.
    The South camera view is limited. Validate all four directional measurements manually.

    ## Important disclosure
    This is historical Bellevue recording (2017), not a newly captured live camera, so it does not fully meet the live-source instruction.
    No measured accuracy (e.g., 70%) is claimed absent manually labelled ground truth.
    The camera-side coordinate configuration is in traffic_pipeline.py and should be verified by visual inspection.
    