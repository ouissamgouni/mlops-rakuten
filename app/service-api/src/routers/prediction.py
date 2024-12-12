from contextlib import asynccontextmanager
import pickle
import re
import string
import aiofiles
import fastapi
import spacy
import torch
import joblib
from pathlib import Path
import logging
from fastapi import FastAPI, HTTPException, APIRouter, UploadFile
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg

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
        text_description = self.df.iloc[idx]['desi_desc_c']
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

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    model_combined_checkpoint =  torch.load('models/local/final_model.pth') #torch.load('models/local/final_model.pth')
    ml_models["text"] = joblib.load('models/local/model_text.sav')
    ml_models["img"] = torch.load('models/local/model_img.pth')
    model_c = CombinedModel(text_input_size = 27, image_input_size = 27, num_classes = NUM_CLASSES)
    model_c.load_state_dict(model_combined_checkpoint)
    model_c.eval()
    ml_models["final"]=model_c
    yield
    ml_models.clear()

router = APIRouter(lifespan=lifespan)

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
    result['desi_desc_c'] = txt_data_origin.apply(lambda x:cleaner(x))
    return result

@router.post('/predict', tags=["predictions"])
async def get_prediction(file: UploadFile| None = None):

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

    #file = open('data/X_text_target.pkl', 'rb')
    #df = pickle.load(file)

    df_raw = pd.read_csv(file.filename, index_col=0)
    img_dir="/Users/ouissamgouni/Documents/it-workspace/bootcamp-mle-24/project-rakuten/code-v1/data/images/image_train"
    
    df=prepare_text_data(df_raw)

    prdtypecode_sorted = ['10', '40', '50', '60', '1140', '1160', '1180', '1280', '1281', '1300', '1301', '1302', '1320', 
                        '1560', '1920', '1940', '2060', '2220', '2280', '2403', '2462', '2522', '2582', '2583', '2585', 
                        '2705', '2905']
    
    with torch.no_grad():
        demo_df= df.sample(n=1)
        designation = demo_df.iloc[0]['designation']
        logger.info('Designation: ' + str(designation))
        description = demo_df.iloc[0]['description']
        logger.info('Description: ' + str(description))

        #prdtypecode = demo_df.iloc[0]['prdtypecode']

        imgid = demo_df.iloc[0]['imageid']
        prdid = demo_df.iloc[0]['productid']
        image_name = f'image_{imgid}_product_{prdid}.jpg'
        image_path = os.path.join(img_dir, image_name)   
        
        try:
            mpimg.imread(image_path)
        except FileNotFoundError as error:  
            print('Image not found', error)


        #logger.info('Expected product type:', prdtypecode,' ',demo_df.iloc[0]['désignation textuelle'])

        model_text = ml_models['text']
        text_prediction = model_text.predict([demo_df.iloc[0]['desi_desc_c']])[0]

        logger.info("Text model :" + str(text_prediction))

        model_img = ml_models['img']
        print(type(model_img))
        img_prediction, _ = model_img.predict(image_path)
        logger.info("Image model :" + str(img_prediction))

        demo_dataset = EcommerceDataset(demo_df, model_text, model_img, img_dir)
        demo_loader = DataLoader(demo_dataset, batch_size = 1, shuffle = False)
    
        model_c = ml_models['final']
        for text_features, img_features in demo_loader:
            combined_features = torch.cat((text_features, img_features), dim = 1)
            outputs = model_c(combined_features)
            _, preds = torch.max(outputs, 1)
            predicted_class = prdtypecode_sorted[preds.item()]
            logger.info("Final model :" + predicted_class)
            print('-'*20)