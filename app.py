from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
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

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    # Save uploaded file
    suffix = ".nii.gz" if file.filename.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Load NIfTI safely
    img = safe_load_nifti(tmp_path)
    data = img.get_fdata()

    # --- Placeholder segmentation ---
    seg = np.zeros_like(data)
    seg[data > np.percentile(data, 99)] = 1  # fake ICH mask

    # --- Save mask ---
    mask_img = nib.Nifti1Image(seg.astype(np.uint8), affine=img.affine)
    mask_path = tmp_path.replace(suffix, "_mask.nii.gz")
    nib.save(mask_img, mask_path)

    # --- Save mid-slice overlay ---
    mid_slice = data.shape[2] // 2
    plt.figure(figsize=(6,6))
    plt.imshow(data[:, :, mid_slice], cmap='gray')
    plt.imshow(seg[:, :, mid_slice], cmap='Reds', alpha=0.5)
    overlay_path = tmp_path.replace(suffix, "_overlay.png")
    plt.axis('off')
    plt.savefig(overlay_path, bbox_inches='tight')
    plt.close()

    os.remove(tmp_path)  # clean up original

    return {
        "mask_file": mask_path,
        "overlay_file": overlay_path,
        "shape": data.shape,
        "status": "Inference complete (placeholder)"
    }

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
