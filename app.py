from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os
import nibabel as nib
import numpy as np
import gzip
import shutil

app = FastAPI(title="ICH Segmentation POC")

# --- Utility: robust NIfTI loader ---
def safe_load_nifti(file_path):
    if file_path.endswith(".nii.gz"):
        try:
            # Check if it's actually gzipped
            with gzip.open(file_path, "rb") as f:
                f.read(1)
            return nib.load(file_path)
        except OSError:
            # Mislabelled file — rename and load as .nii
            new_path = file_path[:-3]  # remove .gz
            shutil.move(file_path, new_path)
            return nib.load(new_path)
    else:
        return nib.load(file_path)


@app.post("/segment")
async def segment_ct(file: UploadFile = File(...)):
    """
    Accepts a .nii or .nii.gz file, performs dummy segmentation,
    and returns basic stats for now (POC).
    """
    try:
        suffix = ".nii.gz" if file.filename.endswith(".gz") else ".nii"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        img = safe_load_nifti(tmp_path)
        data = img.get_fdata()

        # --- Placeholder segmentation (thresholding as mock inference) ---
        seg = np.zeros_like(data)
        seg[data > np.percentile(data, 99)] = 1  # fake "ICH" region

        # --- Compute mock metrics ---
        volume_voxels = np.sum(seg)
        volume_ml = volume_voxels * np.prod(img.header.get_zooms()) / 1000

        # Clean up temp file
        os.remove(tmp_path)

        return JSONResponse({
            "filename": file.filename,
            "shape": data.shape,
            "ICH_volume_ml": round(volume_ml, 3),
            "status": "Segmentation complete (placeholder model)."
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
def root():
    return {"message": "ICH segmentation POC API is running."}
