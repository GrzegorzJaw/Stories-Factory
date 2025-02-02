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
            st.session_state["openai_api_key"] = openai.api_key
            st.rerun()

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
    st.session_state["saved_stories"][0] = st.text_area("Wprowadź treść opowiadania 1", value=st.session_state["saved_stories"][0], height=100)
    st.session_state["saved_stories"][1] = st.text_area("Wprowadź treść opowiadania 2", value=st.session_state["saved_stories"][1], height=100)
    st.session_state["saved_stories"][2] = st.text_area("Wprowadź treść opowiadania 3", value=st.session_state["saved_stories"][2], height=100)
    
    # Opcja wrzucania plików
    uploaded_files = st.file_uploader("Lub dodaj pliki z tekstami (max 3)", type=["txt", "pdf", "doc", "docx"], accept_multiple_files=True)
    file_texts = []
    if uploaded_files:
        for file in uploaded_files[:3]:
            extracted_text = extract_text_from_file(file)
            if extracted_text:
                file_texts.append(extracted_text)
    
    # Wybór długości generowanego tekstu
    st.subheader("Wybierz budżet na generowanie tekstu")
    if st.button("Do 5 PLN"):
        max_tokens = 500
    elif st.button("Do 10 PLN"):
        max_tokens = 1000
    elif st.button("Do 20 PLN"):
        max_tokens = 2000
    else:
        max_tokens = 500
    
    # Generowanie kontynuacji
    if st.button("Generuj kontynuację") and (any(st.session_state["saved_stories"]) or file_texts):
        all_texts = [t for t in st.session_state["saved_stories"] if t] + file_texts
        for i, story in enumerate(all_texts[:3]):
            result = generate_text(f"Kontynuuj to opowiadanie: {story}", max_tokens)
            st.subheader(f"Wygenerowana kontynuacja {i+1}:")
            st.write(result)

with search_tab:
    st.header("Przeglądaj inspirujące historie")
    st.write("(Funkcjonalność w trakcie budowy)")
