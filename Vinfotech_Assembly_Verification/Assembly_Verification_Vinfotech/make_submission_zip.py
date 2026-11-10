"""Create a submission ZIP, excluding the local Python environment and caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

project = Path(__file__).resolve().parent
zip_path = project.parent / 'Assembly_Verification_submission.zip'
excluded_dirs = {'.venv', '__pycache__', '.git', '.idea', '.vscode', 'pip_temp'}
with ZipFile(zip_path, 'w', ZIP_DEFLATED, compresslevel=7) as archive:
    for file in sorted(project.rglob('*')):
        if not file.is_file():
            continue
        relative = file.relative_to(project)
        if any(part in excluded_dirs for part in relative.parts):
            continue
        if file.suffix.lower() in {'.pyc','.pyo'}:
            continue
        archive.write(file, arcname=str(Path(project.name)/relative))
print('Created:', zip_path)
