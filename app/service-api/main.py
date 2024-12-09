from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
import sys

sys.path.append('C:/Users/Dan/Desktop/misc/code/mlops-rakuten/archive/demo-streamlit/streamlit')
from save import ViTPipeline

app = FastAPI()

class Iris(BaseModel):
    data: list

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_models["text_model"] = joblib.load('app/service-api/models/finalized_model_text.sav')

        img_model_path = Path('app/service-api/models/finalized_model_img.pth')
        if img_model_path.exists():
            ml_models["img_model"] = torch.load(img_model_path, map_location=torch.device('cpu')) 

            if isinstance(ml_models["img_model"], torch.nn.Module):
                ml_models["img_model"].eval()
            else:
                print(f"Warning: The img_model is not a PyTorch model, skipping .eval() for {type(ml_models['img_model'])}.")
        else:
            raise FileNotFoundError("You need the finalized_model_img.pth file.")

        # Optional
        combined_model_path = Path('app/service-api/models/concatmodel_2024-10-07_08-52-25_epoch10of20.pth')
        if combined_model_path.exists():
            ml_models["combined_model"] = torch.load(combined_model_path, map_location=torch.device('cpu'))
            if isinstance(ml_models["combined_model"], torch.nn.Module):
                ml_models["combined_model"].eval()
            else:
                print(f"Warning: The combined model is not a PyTorch model, skipping .eval() for {type(ml_models['combined_model'])}.")
        else:
            ml_models["combined_model"] = None 

        yield
    finally:
        ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post('/predict', tags=["predictions"])
def get_prediction(iris: Iris):
    text_data = iris.data[0][0] 
    image_path = iris.data[0][1]
    
    text_model = ml_models.get("text_model")
    img_model = ml_models.get("img_model")
    combined_model = ml_models.get("combined_model")
    
    if not text_model or not img_model:
        raise HTTPException(status_code=500, detail="Models not properly loaded.")
    
    text_prediction = text_model.predict([text_data])[0]
    
    if isinstance(img_model, torch.nn.Module):
        img_prediction, probabilities = img_model.predict(image_path) 
    
    final_prediction = None
    if combined_model:
        text_features = torch.tensor([text_prediction], dtype=torch.float32)
        img_features = torch.tensor(probabilities, dtype=torch.float32)

        combined_features = torch.cat((text_features, img_features), dim=1)

        combined_output = combined_model(combined_features)
        _, final_preds = torch.max(combined_output, 1)
        final_prediction = final_preds.item()

    return {
        "text_prediction": text_prediction,
        "img_prediction": img_prediction,
        "final_prediction": final_prediction,
        "probabilities": probabilities if 'probabilities' in locals() else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
