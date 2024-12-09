from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import torch
from main import app  # Replace with the actual import of your FastAPI app

@patch("main.mlflow.sklearn.load_model")
@patch("main.joblib.load")
@patch("main.torch.load")
def test_predict(mock_torch_load, mock_joblib_load, mock_mlflow_load_model):
    # Mock model loading
    mock_mlflow_load_model.return_value = MagicMock()
    mock_joblib_load.return_value = MagicMock()
    mock_torch_load.return_value = MagicMock()

    # Mock model methods
    mock_mlflow_model = MagicMock()
    mock_mlflow_load_model.return_value = mock_mlflow_model
    mock_mlflow_model.predict.return_value = [0]

    mock_text_model = MagicMock()
    mock_joblib_load.return_value = mock_text_model
    mock_text_model.predict.return_value = [1]  # Mock text model prediction

    mock_img_model = MagicMock()
    mock_torch_load.return_value = mock_img_model
    mock_img_model.predict.return_value = (2, [0.1, 0.9])  # Mock image model prediction

    mock_combined_model = MagicMock()
    mock_torch_load.return_value = mock_combined_model
    mock_combined_model.return_value = torch.tensor([[0.3, 0.7]])  # Mock combined model output

    # Create a mock Iris input to test
    iris_data = {
        'data': [[5.1, 3.5, 1.4, 0.2, 'path/to/image.jpg']]
    }

    # Create TestClient instance
    client = TestClient(app)

    # Send the test data to the prediction endpoint
    response = client.post("/predict", json=iris_data)  # Adjust the endpoint as necessary

    # Validate the response
    assert response.status_code == 200
    assert response.json() == {"prediction": [0]}  # Replace with the expected prediction output
