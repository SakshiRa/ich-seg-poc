from fastapi.testclient import TestClient
from app import app
import os

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "ICH segmentation POC API is running."


def test_inference_upload(tmp_path):
    test_file = "sample_data/TBI_INVZR247VVT.nii"
    if not os.path.exists(test_file):
        # Skip if file not present
        return

    with open(test_file, "rb") as f:
        files = {"file": ("TBI_INVZR247VVT.nii", f, "application/octet-stream")}
        response = client.post("/segment_ct", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "segmentation_result" in data or "message" in data
