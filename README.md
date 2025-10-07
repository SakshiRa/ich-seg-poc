# ICH Segmentation POC 🧠

A minimal FastAPI + MONAI proof of concept for intracranial hemorrhage segmentation on brain CT scans.

## Features
- Upload `.nii` or `.nii.gz` CT scans
- Automatic preprocessing and safe parsing
- Placeholder segmentation logic
- Dockerized backend (FastAPI + Uvicorn)
- Ready for model inference & 3D visualization

## Run locally
```bash
docker build -t ich-seg-poc .
docker run -p 8080:8080 ich-seg-poc
```
## 🧩 Example inference call
```bash
curl -X POST "http://localhost:8080/inference" \
  -F "file=@sample_data/TBI_INVZR247VVT.nii"
```
