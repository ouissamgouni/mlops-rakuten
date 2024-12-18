from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import re
import string
import tempfile
from typing import Annotated
import uuid
import aiofiles
import fastapi
import spacy
from sqlalchemy import select
import torch
import joblib
import logging
from fastapi import Depends, FastAPI, HTTPException,  APIRouter, Request, UploadFile
import os
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from app.routers.metrics import call_evaluate, evaluate_prediction_batch
from ..db import get_async_session, Prediction
from sqlalchemy.ext.asyncio import AsyncSession
from minio import Minio
from ..users import current_active_user

DBSessionDep = Annotated[AsyncSession, Depends(get_async_session)]

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


NUM_CLASSES = 27

prdtypecode_list = ['10', '1140', '1160', '1180', '1280', '1281', '1300', '1301', 
                    '1302', '1320', '1560', '1920', '1940', '2060', '2220', 
                    '2280', '2403', '2462', '2522', '2582', '2583', '2585', 
                    '2705', '2905', '40', '50', '60']

# Tri de la liste par ordre ascendant
sorted_prdtypecode_list = sorted(prdtypecode_list, key = int)

# Dict avec "key":"value" = "prdtypecode":"index" (ou "index" est l'index du "prdtypecode dans la liste triée")
prdtypecode_dict = {code: index for index, code in enumerate(sorted_prdtypecode_list)}


class CombinedModel(nn.Module):
    '''
    - desc : combine les entrées img et text et construit les couches de classification finale
    - params:
    -> text_input_size : taille du vecteur texte
    -> image_input_size : taille du vecteur image
    -> num_classes : nombre de classes en sortie de la couche de classification
    '''
    def __init__(self, text_input_size, image_input_size, num_classes):
        super(CombinedModel, self).__init__()
        # Deux couches fully connected (dense)
        self.fc1 = nn.Linear(text_input_size + image_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, combined_features):
        # Concat des sorties des 2 model
        x = self.relu(self.fc1(combined_features))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x

class EcommerceDataset(Dataset):
    '''
    - desc : cette classe combine les model text, img, effectue une prediction pour chaque et retourne les features
    - params :
    -> df : données des produits
    -> text_model : modèle texte (chargé)
    -> img_model : modèle image (chargé)
    -> img_dir : path racine du dossier image
    - returns :
    -> text_features : sortie du modèle text
    -> img_features : sortie du modèle img
    -> label : label correspondant
    '''
    def __init__(self, df, text_model, img_model, img_dir):
        self.df = df
        self.text_model = text_model
        self.img_model = img_model
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Extraction text / img
        text_description = self.df.iloc[idx]['txt_model_in']
        #prdtypecode = self.df.iloc[idx]['prdtypecode']

        # path img
        imgid = self.df.iloc[idx]['imageid']
        prdid = self.df.iloc[idx]['productid']
        image_name = f'image_{imgid}_product_{prdid}.jpg'
        image_path = os.path.join(self.img_dir, image_name)  

        # Chargement img
        #img = Image.open(image_path)
        
        # Model prediction text - vecteur de probas
        text_features = self.text_model.predict_proba([text_description])[0]
        #print("type(text_features) : ", type(text_features))
        #print("len(text_features) : ", len(text_features))
        # Conversion en tensor si nécessaire
        if isinstance(text_features, np.ndarray):
            text_features = torch.tensor(text_features, dtype = torch.float32)
        #print("type(text_features) after tensor conv : ", type(text_features))
        #print("len(text_features) after tensor conv : ", len(text_features))
        
        # Prédictions du modèle image - vecteur de probas
        _, img_features = self.img_model.predict(image_path)
        #print("type(img_features) : ", type(img_features))
        #print("len(img_features) : ", len(img_features))
        # Conversion en tensor si nécessaire
        if isinstance(img_features, np.ndarray):
            img_features = torch.tensor(img_features, dtype = torch.float32)
        #print("type(img_features) after tensor conv : ", type(img_features))
        #print("len(img_features) after tensor conv : ", len(img_features))

        #label = prdtypecode_dict[str(prdtypecode)]
        #print(f" In EcommerceDataset : Label: {label}, Type: {type(label)}, Size: {label.shape if hasattr(label, 'shape') else 'N/A'}")

        return text_features, img_features#, label



to_ignore=['cm','mm','taille','aaa','aaaa','grand','dimensions','description','hauteur','largeur','couleur','nbsp','comprend','description','import','france','japonais','anglais','blanc','gris','noir']

def camel_split(s):
    if s=='' :return s
    result = [s[0]] 
    for char in s[1:]:
        if char.isupper():
            result.extend([' ', char])
        else:
            result.append(char)
    return ''.join(result)

nlpfr = spacy.load('fr_core_news_sm')
def clean_text_1(c):
    c= re.sub(r'.*Attention !!! Ce produit est un import.*', '', c)
    c= re.sub(re.compile(r'\[Import Allemand\]'), '', c)
    c= re.sub(r'<.*?>', ' ', c)
    c = camel_split(c)
    c = c.lower()
    c= re.sub(re.compile(r'\b(?:{})\b'.format('|'.join(map(re.escape, nlpfr.Defaults.stop_words)))), '', c)
    c= re.sub('[%s]' % re.escape(string.punctuation), '', c)
    c= re.sub(re.compile(r'\b(?:{})\b'.format('|'.join(map(re.escape, to_ignore)))), '', c)
    c= re.sub(r'\d', '', c)
    c=' '.join([word for word in c.split() if len(word) >=3])
    #ignored as takes time consuming
    #c=' '.join([token.lemma_ for token in list(nlpfr(c)) if (token.is_stop==False)])
    return c

def prepare_text_data(data:pd.DataFrame, cleaner=clean_text_1):
    result = data.fillna({'designation':'','description':''})
    txt_data_origin= result["designation"].str.cat(result["description"], sep = " ")
    result['txt_model_in'] = txt_data_origin.apply(lambda x:cleaner(x))
    return result


ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_combined_checkpoint =  torch.load('models/local/final_model.pth')
    ml_models["text"] = joblib.load('models/local/model_text.sav')
    ml_models["img"] = torch.load('models/local/model_img.pth')
    model_c = CombinedModel(text_input_size = 27, image_input_size = 27, num_classes = NUM_CLASSES)
    model_c.load_state_dict(model_combined_checkpoint)
    model_c.eval()
    ml_models["final"]=model_c
    yield
    ml_models.clear()

router = APIRouter(lifespan=lifespan)#, dependencies=[Depends(current_active_user)])

@router.post('/predict', tags=["predictions"])
async def get_prediction(file: UploadFile, db_session: DBSessionDep, request: Request):

    try:
        contents = await file.read()
        async with aiofiles.open(file.filename, 'wb') as f:
            await f.write(contents)
    except Exception:
        raise HTTPException(
            status_code= fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='There was an error uploading the file',
        )
    finally:
        await file.close()

    print(f'Successfuly uploaded {file.filename}')

    df_raw = pd.read_csv(file.filename, index_col=0)

    df=prepare_text_data(df_raw)

    prdtypecode_sorted = ['10', '40', '50', '60', '1140', '1160', '1180', '1280', '1281', '1300', '1301', '1302', '1320', 
                        '1560', '1920', '1940', '2060', '2220', '2280', '2403', '2462', '2522', '2582', '2583', '2585', 
                        '2705', '2905']
    
    with torch.no_grad():

        model_text = ml_models['text']
        text_prediction = model_text.predict(df['txt_model_in'])

        logger.info("Text model :%s",text_prediction)

        model_img = ml_models['img']

        image_names= df.apply(lambda r: 'image_'+ str(r['imageid']) + '_product_' + str(r['productid']) +'.jpg', axis=1)  
        #img_prediction, _ = image_path.apply(lambda x:model_img.predict(x))

        minio_ep=os.environ['MINIO_ENDPOINT']
        minio_ak=os.environ['MINIO_ACCESS_KEY']
        minio_sk=os.environ['MINIO_SECRET_KEY']
        minio_infer_bucket=os.environ['MINIO_INFER_BUCKET']


        minio_client = Minio(minio_ep, secure=False, access_key=minio_ak,secret_key=minio_sk)
        bucket_name = minio_infer_bucket 
        '''
        def store_s3(imgaename):
            imgaepath = img_dir + '/'+ imgaename
            destination_file = Path(imgaepath).name

            # Make the bucket if it doesn't exist.
            found = client.bucket_exists(bucket_name)
            if not found:
                client.make_bucket(bucket_name)
                print("Created bucket", bucket_name)
            else:
                print("Bucket", bucket_name, "already exists")

            # Upload the file, renaming it in the process
            client.fput_object(
                bucket_name, destination_file, imgaepath,
            )
            print(
                imgaepath, "successfully uploaded as object",
                destination_file, "to bucket", bucket_name,
            )

        for _, value in image_path.items():
            store_s3(value)
    '''
        

        #logger.info("Image model :" + str(img_prediction))

        pred_batch_id= uuid.uuid4()

        with tempfile.TemporaryDirectory() as tmpdirname:
            for _, image_name in image_names.items() :
                minio_client.fget_object(bucket_name, image_name,   Path(tmpdirname)/image_name )
    
        
            predictions = []
            
            dataset = EcommerceDataset(df, model_text, model_img, tmpdirname)
            demo_loader = DataLoader(dataset, batch_size = 1, shuffle = False)
        
            model_c = ml_models['final']
        
            for text_features, img_features in demo_loader:
                combined_features = torch.cat((text_features, img_features), dim = 1)
                outputs = model_c(combined_features)
                _, preds = torch.max(outputs, 1)
                predictions.append(prdtypecode_sorted[preds.item()])
            logger.info("Final model : %s", predictions)
            
            dfresult = df.copy()
            dfresult['prediction'] = text_prediction #predictions
            dfresult['pred_batch_id'] = str(pred_batch_id)

            def dfRowToOrm(row):
                pred= Prediction() 
                pred.prediction_batch_id=row['pred_batch_id']
                pred.provided_index=row.name
                pred.prediction=row['prediction']
                pred.designation = row['designation']
                pred.description = row['description']
                pred.product_id = row['productid']
                pred.image_id = row['imageid']
                pred.txt_model_input = row['txt_model_in']
                pred.app_version = request.app.version
                return pred

            list_pred= [dfRowToOrm(row) for _, row in dfresult.iterrows() ] 
            db_session.add_all(list_pred)
            await db_session.commit()
            return pred_batch_id



@router.post('/ground-truth/{prediction_batch_id}', tags=["predictions"])
async def set_ground_truth(prediction_batch_id:str, file: UploadFile, db_session: DBSessionDep):
    try:
        contents = await file.read()
        async with aiofiles.open(file.filename, 'wb') as f:
            await f.write(contents)
    except Exception:
        raise HTTPException(
            status_code= fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='There was an error uploading the file',
        )
    finally:
        await file.close()

    print(f'Successfuly uploaded {file.filename}')
    df_raw = pd.read_csv(file.filename, index_col=0)

    payload=df_raw.to_dict(orient='dict')['prdtypecode']

    predictions = (await db_session.scalars(select(Prediction)\
                                            .filter(Prediction.prediction_batch_id == prediction_batch_id, Prediction.provided_index.in_(payload.keys()))))\
                                                .fetchall()

    if not predictions:
        raise HTTPException(status_code=404, detail="Prediction not found")

    for pred in predictions:
        pred.ground_truth=payload[pred.provided_index]
        pred.ground_truth_at=datetime.now()
    await db_session.commit()
    metrics = await evaluate_prediction_batch(prediction_batch_id, db_session)

    # trigger overall metrics evaluation
    call_evaluate()

    return {'prediction_batch_id':prediction_batch_id, 
            'metrics': metrics}
