import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("data/titanic/train.csv")
st.title("Projet de classification binaire Titanic")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
    st.write("### Introduction")
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())
    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())

if page == pages[1] : 
    st.write("### DataVizualization")
    fig = plt.figure()
    sns.countplot(x = 'Survived', data = df)
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x = 'Sex', data = df)
    plt.title("Répartition du genre des passagers")
    st.pyplot(fig)
    fig = plt.figure()
    sns.countplot(x = 'Pclass', data = df)
    plt.title("Répartition des classes des passagers")
    st.pyplot(fig)
    fig = sns.displot(x = 'Age', data = df)
    plt.title("Distribution de l'âge des passagers")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x = 'Survived', hue='Sex', data = df)
    st.pyplot(fig)
    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig)
    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.heatmap(df[['Age', 'Fare', 'SibSp', 'Parch','Survived']].corr(), ax=ax)
    st.write(fig)