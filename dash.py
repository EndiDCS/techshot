import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import nltk
import pickle
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
from main import *




st.set_page_config(
    page_title="Avaliando Filmes",
    page_icon=":)"
)

titulos = [
    "### 1: Orgulho e preconceito",
    "### 2: O jogo da imitação",
    "### 3: Whisper of the Heart"
]

imagens = [
    "orgulho.jpg",
    "imitacao.jpg",
    "whisper.jpg"
]
st.markdown("# Análise de sentimentos: Avaliando filmes ")
for i in range(0,len(titulos)):



    st.markdown(titulos[i])

    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(imagens[i])
        st.image(image, caption=None, clamp=False, channels="RGB", output_format="auto")
    with col2:
        review = st.text_area('Escreva aqui o seu review:', '',key=i)
        if len(review) <= 1 :
            hide=True
        else:
            hide=False

        review = aplica_transformacoes([review])

        d = {
            'review': review
        }

        with open("model.pickle", "rb") as input_file:
            model = pickle.load(input_file)

        resultado = model.predict(review)

        if hide==False:
            if resultado == 'positive':
                st.success("Review positivo")
            else:
                st.error("Review negativo")













