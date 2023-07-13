import streamlit as st
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    # Tokenização
    tokens = word_tokenize(text.lower())

    # Remoção de stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Reconstituir o texto pré-processado
    preprocessed_text = ' '.join(stemmed_tokens)
    
    return preprocessed_text

# Carregar o modelo e vetorizador
model_filename = 'modelo_naive_bayes.pkl'
vectorizer_filename = 'vectorizer.pkl'

with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

with open(vectorizer_filename, 'rb') as file:
    vectorizer = pickle.load(file)

# Criar a interface do Streamlit
# Configurar a interface do usuário com Streamlit
st.title("Machine learning modelo Naive bayes")

st.header("Classificador SMS SPAM")

# Texto
st.markdown("**Só texto em inglês o modelo só consegue interpretar idioma em inglês**")

# Carregar a imagem a partir de uma URL
image_url = 'https://img.freepik.com/free-vector/stalker-with-laptop-controls-intimidates-victim-with-messages-cyberstalking-pursuit-social-identity-online-false-accusations-concept-pinkish-coral-bluevector-isolated-illustration_335657-1324.jpg?w=1380&t=st=1689130863~exp=1689131463~hmac=b8ea6ec7a4807ad0d1c94db4333b628cdeda9c6e84616ba96522be24de432d28'
st.image(image_url, caption='Vetor')

# Texto 1
sms_text1 = st.text_area("Texto 1", "")

# Texto 2
sms_text2 = st.text_area("Texto 2", "")

# Texto 3
sms_text3 = st.text_area("Texto 3", "")

# Realizar as previsões
if st.button("Classificar"):
    if sms_text1 and sms_text2 and sms_text3:
        processed_sms1 = preprocess_text(sms_text1)  # Pré-processamento do texto 1
        processed_sms2 = preprocess_text(sms_text2)  # Pré-processamento do texto 2
        processed_sms3 = preprocess_text(sms_text3)  # Pré-processamento do texto 3

        # Vetorização dos textos de teste usando o vetorizador ajustado
        vectorized_sms1 = vectorizer.transform([processed_sms1])  # Vetorização do texto 1
        vectorized_sms2 = vectorizer.transform([processed_sms2])  # Vetorização do texto 2
        vectorized_sms3 = vectorizer.transform([processed_sms3])  # Vetorização do texto 3

        # Classificação dos textos
        prediction1 = loaded_model.predict(vectorized_sms1)  # Classificação do texto 1
        prediction2 = loaded_model.predict(vectorized_sms2)  # Classificação do texto 2
        prediction3 = loaded_model.predict(vectorized_sms3)  # Classificação do texto 3

        # Exibição das classificações
        st.write("Classificação do Texto 1: ", prediction1[0])
        st.write("Classificação do Texto 2: ", prediction2[0])
        st.write("Classificação do Texto 3: ", prediction3[0])
    else:
        st.warning("Por favor, insira todos os textos do SMS.")
