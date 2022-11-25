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

st.markdown("# Avaliando filmes ")



review = st.text_input('Review', 'I love this movie the drama and all his story is so good.')
st.write('The current movie title is', review)

review = aplica_transformacoes([review])




d = {
    'review':review
}

with open("model.pickle","rb") as input_file:
    model = pickle.load(input_file)

resultado = model.predict(review)

if resultado == 'positive':
    st.success("Review positivo")
else:
    st.error("Review negativo")
st.write(model)
