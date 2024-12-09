import sys
import os
import joblib
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
import torch

# Add the directory containing the 'save' module to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../archive/demo-streamlit/streamlit')))

# Define model paths
image_model_path = "app/service-api/models/finalized_model_img.pth"  # Image model path
text_model_path = "app/service-api/models/finalized_model_text.sav"  # Text model path

# Load the models
image_model = torch.load(image_model_path)  # Image model
text_model = joblib.load(text_model_path)  # Text model

# Import the app after loading models
from main import app

client = TestClient(app)

@pytest.fixture
def mock_model_loading():
    """
    Mock the model loading processes to avoid dependency on actual model files.
    """
    with patch("torch.load") as mock_torch_load, \
         patch("joblib.load") as mock_joblib_load:
        mock_torch_load.return_value = image_model  # Mock the torch model
        mock_joblib_load.return_value = text_model  # Mock the joblib model
        yield mock_torch_load, mock_joblib_load


def test_predict(mock_model_loading):
    """
    Test the /predict endpoint with mocked model loading.
    """
    image_path = "app/service-api/samples/harry_potter_lego.jpg"  # Path to the test image

    # Ensure the file exists
    assert os.path.exists(image_path), f"Test image file not found at {image_path}"

    # Open the image file and send a POST request with both text and file data
    with open(image_path, "rb") as image_file:
        response = client.post(
            "/predict",
            data={"text": "test"},
            files={"file": image_file}  # Send the file as 'file' in the multipart/form-data
        )

    # Check that the status code is 200 (OK)
    assert response.status_code == 200

    # Optionally, check the response data structure
    response_json = response.json()
    assert "prediction" in response_json  # Adjust according to your actual API response structure
