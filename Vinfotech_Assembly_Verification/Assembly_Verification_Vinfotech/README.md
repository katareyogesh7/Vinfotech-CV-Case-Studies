# Vinfotech — Assembly Verification (Case Study 1)

**Classical OpenCV prototype. Runs locally on a CPU using Python, NumPy, and OpenCV.** No Google Colab, YOLO, trained model, GPU, internet connection, or hard-coded test-image verdicts are required after installation.

> **Scope:** This folder is **Case Study 1 only**. The Vinfotech brief also requests a separate **Case Study 2 (traffic analytics)**; this project does not implement that second case study.

## 1. Project layout

```text
Assembly_Verification_Vinfotech/
  assembly_check.py            <- Run this program; do NOT name it inspect.py
  requirements.txt
  README.md
  APPROACH_200_WORDS.md
  make_submission_zip.py
  run_all.bat
  candidate_pack/
    reference_good/            <- 3 known-good PNGs
    test_images/               <- 6 supplied test PNGs
  output/                      <- GENERATED automatically, replaced every run
```

## 2. Setup using E drive (Windows PowerShell in VS Code)

Extract the ZIP to `E:\Assembly_Verification_Vinfotech`, then open that folder in VS Code.

```powershell
cd E:\Assembly_Verification_Vinfotech
python --version
```

To reduce storage usage on C, place temporary installation files on E **in the current terminal session**:

```powershell
New-Item -ItemType Directory -Force E:\PythonTemp | Out-Null
$env:TEMP = 'E:\PythonTemp'
$env:TMP = 'E:\PythonTemp'
$env:PIP_CACHE_DIR = 'E:\PythonTemp\pip-cache'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
```

Python itself, VS Code, and some operating-system files may still occupy C. The `.venv` packages, source images, saved results, and redirected temporary installation files go to E.

**If `python` is not recognized:** install Python normally first; then try `py -m venv .venv` instead. In VS Code choose **Python: Select Interpreter** and select `.venv\Scripts\python.exe`.

## 3. Run the code

With a PowerShell terminal open in the project folder:

**All 6 test images:**

```powershell
.\.venv\Scripts\python.exe assembly_check.py --folder candidate_pack\test_images
```

**One test image:**

```powershell
.\.venv\Scripts\python.exe assembly_check.py --image candidate_pack\test_images\test_01.png
```

**Run the default test folder:**

```powershell
.\.venv\Scripts\python.exe assembly_check.py
```

**New image in another folder (the live interview):**

```powershell
.\.venv\Scripts\python.exe assembly_check.py --image "E:\new_images\sample.png"
```

Or double-click `run_all.bat` after setup.

The script uses **paths relative to its own folder**, so it can also run from a different drive or inside PyCharm. It supports paths containing non-English Unicode characters.

**Important: Every run replaces the entire dedicated `output` folder.** Copy any output you want to preserve before running the script again. **The `candidate_pack` input folders are never overwritten.**

## 4. Generated outputs

- `output/inspection_report.csv` — one row per image, verdict `OK` / `NOT OK`, issues
- `output/component_details.csv` — a row for **each component check**, measured score and threshold
- `output/thresholds.json` — calibration thresholds derived from the three good references
- `output/test_XX_assembly.png` — nut, gears, circlips on original camera image
- `output/test_XX_hardware.png` — screws and motor area on a straightened view
- `output/test_XX_all_checks.png` — the two images side by side
- `output/summary_grid.png` — contact sheet for the inspected batch

Red = rule flagged a concern; green = rule passed. **Green does not guarantee industrial acceptance**. A skipped circlip check means its carrier gear was already flagged as missing.

## 5. Algorithm

1. **Auto-locate golden gear** with HSV colour threshold, morphology, contours and image moments.
2. **Check output nut** using golden-pixel percentage in a ring at the expected gear centre.
3. **Find idler and compound gears** from their expected geometry relative to the detected gold gear; measure grey coverage and dark circlip regions.
4. **Auto-locate/straighten blue base plate** with HSV segmentation, minimum-area rotated rectangle and perspective transform.
5. **Inspect screws** at one-time calibrated positions on the straightened plate. Use grey coverage for the four corner bolts, and Canny edge counts for four bearing screws and two motor mounting screws.
6. **Check visible pinion, cable connector, and red cable** with HSV colour masks.
7. Calibrate trial limits **only using the three known-good reference images**, inspect all supplied tests, print statuses and save images/reports.

## 6. Assumptions and limitations

**Assumptions:** Same assembly type, similar appearance and resolution, fixed overhead camera with small rotation/translation; blue plate, gold gear, grey metal, dark-blue circlips and red cable keep roughly their supplied colours; the expected relative component geometry is unchanged; all three `reference_good` images are correctly assembled.

**Known limitations:** This is a **rule-based prototype, not a trained ML model**. HSV ranges, position ratios and threshold margins were manually selected and evaluated on these supplied images. Scores can fail under new lighting, colour, view angle, resolution, substantial occlusion, or misplaced components. The checks for supporting **bearing-block bodies and the motor itself** are indirect: visible screws/pinion/connector are inspected, but an invisible or malfunctioning motor is not verified. The system does **not reliably detect every possible missing or misplaced part** as requested in the full brief. A bare plate with unexpected parts may also cause an inspection error rather than a clean component diagnosis.

The reported `OK` means **all implemented image-based checks passed**, not guaranteed complete assembly quality. Test on unseen faulty and correct images before deploying.

**What I would improve:** Acquire labelled good/fault examples for *every* component and misplacement type, track board alignment with physically fixed fiducials, validate small fasteners with shape/template checks, add explicit bearing-block/motor-body inspection, and report confidence/uncertainty rather than a single fixed binary threshold.

## 7. Validation demonstrated

- Tested the supplied reference and test images on a local CPU.
- The provided example run produces: Test 01 `NOT OK` (output nut); 02 `OK`; 03 `OK`; 04 `NOT OK` (idler circlip); 05 `OK`; 06 `NOT OK` (idler gear).
- Artificial missing *corner* and *bearing* screws were detected by their experimental checks. These simulations are not substitutes for real unseen faulty examples.
- The test results were used during the development of these rules; they are **not an independent accuracy benchmark**.

## 8. Submission ZIP

After running the batch command and checking the report:

```powershell
.\.venv\Scripts\python.exe make_submission_zip.py
```

This creates `Assembly_Verification_submission.zip` one directory *above* the project, including code, images, README, and the latest outputs, but excluding `.venv`, caches and temporary folders. **The broader case-study brief also asks for a Git repository link and Case Study 2.**
