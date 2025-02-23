import streamlit as st
from dotenv import dotenv_values
import openai
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.cluster import KMeans
import random
import json
import os

# Instalacja torch w wersji CPU (bo Streamlit Cloud nie obsługuje GPU)
os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")


# Nazwa modelu
model_name = 'Lajonbot/LaMini-GPT-774M-19000-steps-polish'

# Pobierz model
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Załaduj zmienne środowiskowe
config = dotenv_values(".env")

# Streamlit
st.set_page_config(page_title="Fabryka Opowieści: Asystent Twórczy", layout="wide")

# Sprawdź i ustaw klucz OpenAI API
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = config.get("OPENAI_API_KEY", "")
    
openai.api_key = st.session_state.get("openai_api_key", "")

if not openai.api_key:
    openai.api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if not openai.api_key:
        st.warning("You need to add your OpenAI API key to use this app.")
        st.stop()

nlp = spacy.load("en_core_web_sm")


ANALYSIS_FILE = "analysis.json"  # Ścieżka do pliku zapisu analizy

def save_analysis_to_file():
    """Zapisuje analizę do pliku JSON."""
    with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "ner_results": st.session_state.get("ner_results", []),
            "topic_results": st.session_state.get("topic_results", {})
        }, f, ensure_ascii=False, indent=4)

def load_analysis_from_file():
    """Wczytuje analizę z pliku JSON, jeśli istnieje."""
    if os.path.exists(ANALYSIS_FILE):
        with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            st.session_state["ner_results"] = data.get("ner_results", [])
            st.session_state["topic_results"] = data.get("topic_results", {})

def delete_analysis_file():
    """Kasuje zapisany plik analizy."""
    if os.path.exists(ANALYSIS_FILE):
        os.remove(ANALYSIS_FILE)
    st.session_state.pop("ner_results", None)
    st.session_state.pop("topic_results", None)
    st.sidebar.success("✅ Analiza została skasowana.")

def analyze_text_with_ner(input_text):
    st.sidebar.write("🔍 **Rozpoczęto analizę tekstu**")
    max_chunk_size = 500
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    tokens = inputs['input_ids'].shape[1]
    
    if tokens > max_chunk_size:
        chunks = [input_text[i:i+max_chunk_size] for i in range(0, len(input_text), max_chunk_size)]
    else:
        chunks = [input_text]
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        st.sidebar.write(f"📌 Analizowanie części {i+1} z {len(chunks)}")
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            model_output = model(**inputs)
        embeddings.append(model_output.logits.mean(dim=1).numpy())
        time.sleep(1)
    
    st.session_state['text_embeddings'] = embeddings
    st.sidebar.write("✅ **Analiza zakończona!**")

    save_analysis_to_file()  # Zapisujemy analizę do pliku

    max_chunk_size = 500  # Limit długości fragmentu tekstu
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    tokens = inputs['input_ids'].shape[1]
    
    if tokens > max_chunk_size:
        chunks = [input_text[i:i+max_chunk_size] for i in range(0, len(input_text), max_chunk_size)]
    else:
        chunks = [input_text]
    
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            model_output = model(**inputs)
        embeddings.append(model_output.logits.mean(dim=1).numpy())
    
    st.session_state['text_embeddings'] = embeddings
    doc = nlp(input_text)
    entities_info = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
    st.session_state['ner_results'] = entities_info
    st.sidebar.write("🔍 Rozpoznane encje:", entities_info)  # Debug

def analyze_text_with_topic_modeling(input_text, num_topics=3):
    vectorizer = CountVectorizer(stop_words='english')
    text_data = vectorizer.fit_transform([input_text])
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(text_data)

    topics = {}
    for index, topic in enumerate(lda.components_):
        topics[f"Topic {index+1}"] = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]
    
    st.session_state['topic_results'] = topics
    st.sidebar.write("📌 Wykryte tematy:", topics)  # Debug

    save_analysis_to_file()  # Zapisujemy analizę do pliku

def generate_text_based_on_user_input(generation_choice, additional_input, length_option, previous_story=""):
    # Wczytanie analizy przed generowaniem
    load_analysis_from_file()

    if "ner_results" not in st.session_state or "topic_results" not in st.session_state:
        st.sidebar.error("❌ Analiza nie jest dostępna. Najpierw przeanalizuj tekst!")
        return "Error: Analysis data not available."

    entities = ', '.join(ent['text'] for ent in st.session_state['ner_results'])
    topics = '; '.join(f"{key}: {', '.join(val)}" for key, val in st.session_state['topic_results'].items())

    prompt = f"""
    Kontynuuj poniższą historię, zachowując jej logiczną spójność, styl i rozwijając istniejące wątki.
    
    Historia:
    {previous_story if previous_story else additional_input}
    
    Ważne postacie i elementy fabularne: {entities}
    Główne tematy i motywy: {topics}
    
    Zadbaj o płynność narracji i konsekwencję wydarzeń.
    """

    length_options = {
        "Tekst krótki (maks. 5 PLN)": 700,
        "Tekst dłuższy (maks. 10 PLN)": 1800,
        "Tekst długi (maks. 20 PLN)": 3500
    }
    max_tokens = length_options.get(length_option, 300)

    time.sleep(2)  # Opóźnienie zapobiegające przekroczeniu limitów
    response = openai.ChatCompletion.create(
        stop=None,
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "Jesteś asystentem literackim pomagającym tworzyć spójne opowieści."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.9
    )

    generated_text = response['choices'][0]['message']['content'].strip()

    if not generated_text:
        st.warning("Błąd: Wygenerowany tekst jest pusty.")

    return generated_text

def generation():
    st.subheader("Jaki jest główny cel opowieści?")
    goal_options = [
        "IDEE I REFLEKSJE", "ROZRYWKA", "EDUKACJA", "EMOCJONALNE DOŚWIADCZENIE", "EKSPERYMENTY Z WYMYŚLONYM ŚWIATEM"
    ]
    
    if 'selected_goal' not in st.session_state:
        st.session_state['selected_goal'] = None
    
    col1, col2, col3 = st.columns(3)
    for i, goal in enumerate(goal_options):
        with (col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3):
            if st.button(goal, key=f'goal_{i}'):
                st.session_state['selected_goal'] = goal
    
    if st.session_state['selected_goal']:
        if st.session_state['selected_goal'] == "IDEE I REFLEKSJE":
            st.subheader("Wybierz dodatkowe opcje dla IDEI I REFLEKSJI:")
            option1 = st.checkbox("KONTYNUACJA - Analiza istniejących idei i refleksji oraz rozwijanie ich w nowym tekście")
            option2 = st.checkbox("WIELOWARSTWOWE ZAKOŃCZENIA - Tworzenie otwartych zakończeń z możliwością interpretacji")
            option3 = st.checkbox("ZADAWANIE PYTAŃ I DYLEMATY - Inspiracja do refleksji nad trudnymi pytaniami moralnymi")
            
            st.session_state['idea_options'] = {
                "Kontynuacja": option1,
                "Wielowarstwowe zakończenia": option2,
                "Zadawanie pytań i dylematy": option3
            }
        st.write(f"**Wybrany cel:** {st.session_state['selected_goal']}")
    if 'generated_story' not in st.session_state:
        st.session_state['generated_story'] = ""
    
    generation_choice = st.selectbox("Wybierz opcję", ["Kontynuuj istniejącą opowieść", "Stwórz nową opowieść"])
    additional_input = st.text_area("Podaj tekst jako punkt wyjścia:", key='generation_input')
    length_option = st.selectbox("Wybierz długość", ["Tekst krótki (maks. 5 PLN)", "Tekst dłuższy (maks. 10 PLN)", "Tekst długi (maks. 20 PLN)"])
    
    if st.button("Generuj opowieść"):
        if not additional_input and not st.session_state['generated_story']:
            st.warning("Podaj tekst jako punkt wyjścia.")
        else:
            new_story = generate_text_based_on_user_input(
                generation_choice, additional_input, length_option, st.session_state['generated_story']
            )
            time.sleep(10)
            st.session_state['generated_story'] += "\n\n" + new_story
            st.rerun()
    
    st.text_area("Wygenerowana opowieść:", st.session_state['generated_story'], height=300)
    
    if st.session_state['generated_story']:
        if st.button("Kontynuuj opowieść"):
            new_story = generate_text_based_on_user_input(
                "Kontynuuj istniejącą opowieść", "", length_option, st.session_state['generated_story']
            )
            time.sleep(10)
            st.session_state['generated_story'] += "\n\n" + new_story
            st.rerun()

def home():
    st.subheader("Witaj w Fabryce Opowieści: Asystent Twórczy!")
    st.write("Wybierz funkcję z menu po lewej stronie.")

def analysis():
    user_text = st.text_area("Podaj tekst do analizy:", key='analysis_text')
    if st.button("Analizuj tekst"):
        if not user_text:
            st.warning("Podaj tekst do analizy.")
        else:
            analyze_text_with_ner(user_text)
            analyze_text_with_topic_modeling(user_text)

def main():
    page_choice = st.sidebar.selectbox("Wybierz stronę", ["Home", "Analysis", "Generation"], index=0)
    
    if page_choice == "Home":
        home()
    elif page_choice == "Analysis":
        analysis()
    elif page_choice == "Generation":
        generation()

if __name__ == "__main__":
    main()
