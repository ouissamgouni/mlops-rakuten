import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
import sys

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the directory to sys.path to import the ViTPipeline class
sys.path.append('C:/Users/Dan/Desktop/misc/code/mlops-rakuten/archive/demo-streamlit/streamlit')
from save import ViTPipeline

app = FastAPI()

class Iris(BaseModel):
    data: list

ml_models = {}

# Image preprocessing function (define this based on your image model)
def preprocess_image(image_path: str):
    from PIL import Image
    import torchvision.transforms as transforms

    # Define a transform to preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.debug("Loading models...")
        try:
            ml_models["text_model"] = joblib.load('app/service-api/models/finalized_model_text.sav')
            logger.debug("Text model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading text model: {e}")
            raise HTTPException(status_code=503, detail="Text model unavailable.")

        img_model_path = Path('app/service-api/models/finalized_model_img.pth')
        if img_model_path.exists():
            try:
                ml_models["img_model"] = torch.load(img_model_path, map_location=torch.device('cpu')) 
                if isinstance(ml_models["img_model"], torch.nn.Module):
                    ml_models["img_model"].eval()
                    logger.debug("Image model loaded and set to evaluation mode.")
                else:
                    logger.warning(f"The img_model is not a PyTorch model, skipping .eval() for {type(ml_models['img_model'])}.")
            except Exception as e:
                logger.error(f"Error loading image model: {e}")
                raise HTTPException(status_code=503, detail="Image model unavailable.")
        else:
            logger.warning("Image model file not found. Skipping image model loading.")

        # Optional combined model
        combined_model_path = Path('app/service-api/models/concatmodel_2024-10-07_08-52-25_epoch10of20.pth')
        if combined_model_path.exists():
            try:
                ml_models["combined_model"] = torch.load(combined_model_path, map_location=torch.device('cpu'))
                if isinstance(ml_models["combined_model"], torch.nn.Module):
                    ml_models["combined_model"].eval()
                    logger.debug("Combined model loaded and set to evaluation mode.")
                else:
                    logger.warning(f"The combined model is not a PyTorch model, skipping .eval() for {type(ml_models['combined_model'])}.")
            except Exception as e:
                logger.error(f"Error loading combined model: {e}")
                raise HTTPException(status_code=503, detail="Combined model unavailable.")
        else:
            ml_models["combined_model"] = None
            logger.debug("No combined model found. Skipping combined model loading.")

        yield
    finally:
        logger.debug("Cleaning up models...")
        ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post('/predict', tags=["predictions"])
def get_prediction(iris: Iris):
    logger.debug("Prediction endpoint hit.")
    try:
        text_data = iris.data[0][0]
        image_path = iris.data[0][1]
        logger.debug(f"Received text data: {text_data}")
        logger.debug(f"Received image path: {image_path}")

        # Check model availability
        text_model = ml_models.get("text_model")
        img_model = ml_models.get("img_model")
        if not text_model or not img_model:
            logger.error("Models not properly loaded.")
            raise HTTPException(status_code=503, detail="Models not properly loaded.")
        
        # Text Prediction
        text_prediction = text_model.predict([text_data])[0]
        logger.debug(f"Text model prediction: {text_prediction}")

        # Image Processing and Prediction
        image_tensor = preprocess_image(image_path)  # Define this function
        img_prediction = None
        if isinstance(img_model, torch.nn.Module):
            img_model.eval()
            img_prediction = img_model(image_tensor)  # Assuming img_model is a nn.Module
            logger.debug(f"Image model prediction: {img_prediction}")
        else:
            logger.error("Invalid image model.")
            raise HTTPException(status_code=500, detail="Invalid image model.")
        
        # Combine predictions if combined model exists
        final_prediction = None
        combined_model = ml_models.get("combined_model")
        if combined_model:
            logger.debug("Using combined model for final prediction.")
            text_features = torch.tensor([text_prediction], dtype=torch.float32)
            img_features = torch.tensor(img_prediction, dtype=torch.float32)
            combined_features = torch.cat((text_features, img_features), dim=1)
            combined_output = combined_model(combined_features)
            _, final_preds = torch.max(combined_output, 1)
            final_prediction = final_preds.item()
            logger.debug(f"Combined model final prediction: {final_prediction}")

        return {
            "text_prediction": text_prediction,
            "img_prediction": img_prediction.tolist() if isinstance(img_prediction, torch.Tensor) else img_prediction,
            "final_prediction": final_prediction,
            "probabilities": img_prediction.tolist() if isinstance(img_prediction, torch.Tensor) else None
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.debug("Starting FastAPI application with uvicorn.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
