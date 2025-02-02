import streamlit as st
from dotenv import dotenv_values
import openai
from PyPDF2 import PdfReader
import docx
import io

# Wczytanie zmiennych środowiskowych
env = dotenv_values(".env")

# Funkcja do generowania tekstu za pomocą OpenAI
def generate_text(prompt, max_tokens):
    """Generuje tekst na podstawie podanego promptu."""
    try:
        res = openai.Completion.create(
            model="gpt-4",  # Upewnij się, że używasz poprawnego modelu, np. gpt-4
            temperature=0.7,
            max_tokens=max_tokens,
            prompt=prompt
        )
        return res.choices[0].text.strip()
    except Exception as e:
        return f"Błąd: {str(e)}"

# Funkcja do odczytu tekstu z plików
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

# Konfiguracja strony w Streamlit
st.set_page_config(page_title="Fabryka Opowieści: Asystent Twórczy", layout="centered")

# OpenAI API key protection
if "openai_api_key" not in st.session_state:
    if env.get("OPENAI_API_KEY"):
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
    else:
        st.info("Dodaj swój klucz API OpenAI, aby móc korzystać z tej aplikacji.")
        st.session_state["openai_api_key"] = st.sidebar.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            openai.api_key = st.session_state["openai_api_key"]

if not st.session_state.get("openai_api_key"):
    st.stop()

# Inicjalizacja klienta OpenAI
openai.api_key = st.session_state["openai_api_key"]

# Tytuł aplikacji
st.title("Linia produkcyjna opowieści niedokończonych")

# Zakładki
add_tab, search_tab = st.tabs([ 
    "Dodaj opowieści, które mają być dokończone", 
    "Dodaj opowieści, które Ciebie inspirują"
])

# Inicjalizacja pamięci w sesji
if "saved_stories" not in st.session_state:
    st.session_state["saved_stories"] = ["", "", ""]

with add_tab:
    st.header("Dodaj swoje opowiadanie")
    
    # Opcja wklejania tekstu
    st.session_state["saved_stories"][0] = st.text_area("Wprowadź treść opowiadania 1", value=st.session_state["saved_stor
