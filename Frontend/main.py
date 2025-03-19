import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model import classifySentiment,groupClassifier


st.title("🔍 Análisis de Sentimiento para Empresas")

option = st.sidebar.selectbox("Elige una opción", ["Analizar un comentario", "Subir archivo CSV"])


if option == "Analizar un comentario":
    user_input = st.text_area("Escribe un comentario:")
    if st.button("Analizar"):
        original_text ,sentiment = classifySentiment(user_input)
        st.write(f"📊 Sentimiento: {sentiment}")

elif option == "Subir archivo CSV":
    file = st.file_uploader("Sube un archivo con comentarios", type=["csv"])
    if file:
        df = pd.read_csv(file,sep=None, engine="python")
        clasification = groupClassifier(df)
        clasified_data = pd.DataFrame(clasification)
        plt.figure(figsize=(6,4))
        sns.countplot(x=clasified_data["label"], hue=clasified_data["label"], palette="pastel", legend=False)

        # Etiquetas y título
        plt.xlabel("Sentiment category")
        plt.ylabel("number of texts")
        plt.title("Data distribution")

        st.pyplot(plt)
        
        # Mostrar estadísticas
        #sentiment_counts = df["sentiment"].value_counts()
        #st.bar_chart("Aqui va la visualizacion de archivo")
        
        # WordCloud
       # words = " ".join(df["texto"])
        #wordcloud = WordCloud(width=800, height=400).generate(words)
        #st.image(wordcloud.to_array())"""
