import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf


MODELS_ROOT = r"../models/"
conv1D_filename = "conv1D_model.h5"
lstm_with_glove = "model_text_with_glove.h5"



def get_class_code(val):
    dict_class =       {0:10, 1:40, 2:50, 3:60, 4:1140,
                        5:1160, 6:1180, 7:1280, 8:1281,
                        9:1300, 10:1301, 11:1302, 12:1320,
                        13:1560, 14:1920, 15:1940, 16:2060,
                        17:2220, 18:2280, 19:2403, 20:2462,
                        21:2522, 22:2582, 23:2583, 24:2585,
                        25:2705, 26:2905,
                        }   
    return dict_class[val]



def get_label(code):
    dict_code_label =  { 10 : 'Livres anciens / occasion',
            40 : 'Jeux vidéos anciens, équipement',
            50 : 'Accessoires & produits dérivés gaming',
            60 : 'Consoles de jeu',
            1140 : 'Figurines',
            1160 : 'Cartes de jeu',
            1180 : 'Figurines & Jeux de Société',
            1280 : 'Jeux & jouets pour enfants',
            1281 : 'Jeux de société',
            1300 : 'Modélisme',
            1301 : 'Vêtements bébé et jeux pour la maison',
            1302 : 'Jeux & jouets d\'extérieur pour enfants',
            1320 : 'Jouets & accessoires pour bébé',
            1560 : 'Meubles d\'intérieur',
            1920 : 'Linge de maison',
            1940 : 'Alimentation & vaisselle',
            2060 : 'Objets décoration maison',
            2220 : 'Equipement pour animaux',
            2280 : 'Journaux, revues, magazines anciens',
            2403 : 'Livres, BD, magazines anciens',
            2462 : 'Consoles, jeux et équipement occasion',
            2522 : 'Papeterie',
            2582 : 'Meubles d\'extérieur',
            2583 : 'Equipement pour piscine',
            2585 : 'Outillage intérieur / extérieur, tâches ménagères',
            2705 : 'Livres neufs',
            2905 : 'Jeux PC'}
   
    return dict_code_label[code]

def get_real_target(val):
    
    dict_labels = {'0': 0,
                 '1': 1,
                 '10': 2,
                 '11': 3,
                 '12': 4,
                 '13': 5,
                 '14': 6,
                 '15': 7,
                 '16': 8,
                 '17': 9,
                 '18': 10,
                 '19': 11,
                 '2': 12,
                 '20': 13,
                 '21': 14,
                 '22': 15,
                 '23': 16,
                 '24': 17,
                 '25': 18,
                 '26': 19,
                 '3': 20,
                 '4': 21,
                 '5': 22,
                 '6': 23,
                 '7': 24,
                 '8': 25,
                 '9': 26}
        
    
    for real_cls, gen_label in dict_labels.items():
         if val == gen_label:
            return int(real_cls)

    return "class doesn't exist"


# Load Pretrained Models - Conv1D 
#@st.cache(allow_output_mutation=True) depreacetd

@st.cache_resource
def load_my_model(model):
    st.write("Loading model...")
    model = load_model(MODELS_ROOT + model )     #,  compile = True
    st.write("Loading model FINISHED")
    return model


def tokenize_text(input_text):
    st.write("tokenize_text...")
    maxlen = 300 
    tokenizer = Tokenizer(num_words = maxlen)
    tokenizer.fit_on_texts(input_text)
    # Get our text data word index
    text_word_index = tokenizer.word_index
    # get textt tokenization
    text_seq = tokenizer.texts_to_sequences(input_text)
    text = tf.keras.utils.pad_sequences(text_seq)
    text = tf.keras.preprocessing.sequence.pad_sequences(text,
                                                        maxlen = maxlen,
                                                        padding='post')    
    st.write("tokenize_text FINISHED") 
    st.write(text)
    st.write("token matrix dim = " + str(text.shape))
    return text



def predict_with_conv1d(input_text):    
    st.write("Predict class...")
    text_tokenized = tokenize_text(input_text)
    
    model = load_my_model(conv1D_filename)
    
    #model = load_my_model(lstm_with_glove)
    
    #model.summary()
    y_pred_proba = model.predict(text_tokenized)
    st.write("Probas:")
    st.write(y_pred_proba)
    st.write("dim probas matrix=" + str(y_pred_proba.shape))
    y_pred_class = np.argmax(y_pred_proba, axis = 1).astype(int)    
    st.write("predicetd classes:")
    st.write(y_pred_class)
    st.write('length:' + str(len(y_pred_class)))
    
    # get corresponding cats
    #cats = []
    #for y in y_pred_class:
    #    st.text(y)
    #    cats.append([str(y), get_label(y)])
    #
    #st.write(cats)    
    #df_classes = pd.DataFrame(cats)
    #st.dataframe(df_classes)
    
    # get prediction
    counts = np.bincount(y_pred_class)
    st.write("counts =" + str(counts))
    y_pred = np.argmax(counts)  #y_pred_class[0]
    st.write(f"y_pred= {y_pred}")
    pred_class = get_class_code(y_pred)
    st.write(f"pred_class= {pred_class}")
    pred_label = get_label(pred_class)
    st.write(f"pred_label= {pred_label}")
    st.write("Predict class FINISHED")
    
    return str(pred_class) , pred_label , y_pred_proba
    
    
user_desig_input = st.text_area('*Designation du produit', )
user_descrip_input = st.text_area('Description', )

if st.button('Classifier'): 
    if user_desig_input == "":
        st.write('Merci de saisir un le champ "Designation du produit"' ) 
    else : 
        if len(user_descrip_input) > 0:
            pred_class , pred_label ,y_pred_proba = predict_with_conv1d(' '.join(user_desig_input).join(' ').join(user_descrip_input) )                    
        else:
            pred_class , pred_label ,y_pred_proba = predict_with_conv1d(user_desig_input)
            
        precision = np.amax(y_pred_proba)
        precision = precision * 100
        precision = np.round(precision,2)

        msg1 = '<span style="color:green">La classe prédite: '  + str(pred_class) + '</span>'
        msg2 = '<span style="color:green">La catégorie prédite : ' + pred_label + '</span>'
        msg3 = '<span style="color:green">Certitude : ' + str(precision) +'%'+ '</span>'

        st.markdown(msg1, unsafe_allow_html=True)
        st.markdown(msg2, unsafe_allow_html=True)
        st.markdown(msg3, unsafe_allow_html=True) 

