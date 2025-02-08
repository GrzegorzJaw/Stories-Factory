import streamlit as st
from dotenv import dotenv_values
import openai
import tiktoken
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
import time

# Załaduj zmienne środowiskowe
config = dotenv_values(".env")

# Streamlit page configuration
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

encoding = tiktoken.encoding_for_model("gpt-4")
nlp = spacy.load("en_core_web_sm")

#przerabianie tekstu na embedingsy w celu analizy
def analyze_and_store_embeddings(input_text):
    segment_length = 3000
    segments = [input_text[i:i + segment_length] for i in range(0, len(input_text), segment_length)]
    
    all_embeddings = []
    
    for segment in segments:
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=segment
            )
            embeddings = response['data'][0]['embedding']
            all_embeddings.append(embeddings)
        except Exception as e:
            st.error(f"Error in embedding: {e}")
            return None

    st.session_state['embeddings'] = all_embeddings

#Funkcja ner  umożliwia rozpoznanie kluczowych elementów w tekście,
# takich jak imiona, nazwy miejsc czy organizacji, co może być używane do dalszej analizy lub generowania treści.
def analyze_text_with_ner(input_text):
    doc = nlp(input_text)
    entities_info = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
    st.session_state['ner_results'] = entities_info

#Funkcja topic_modeling  pozwala określić główne tematy i ich charakterystyczne słowa w analizowanym tekście
def analyze_text_with_topic_modeling(input_text, num_topics=3):
    vectorizer = CountVectorizer(stop_words='english')
    text_data = vectorizer.fit_transform([input_text])
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(text_data)

    topics = {}
    for index, topic in enumerate(lda.components_):
        topics[f"Topic {index+1}"] = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]
    
    st.session_state['topic_results'] = topics

#Funkcja concept map tworzy strukturę graficzną reprezentującą relacje między różnymi elementami tekstu,
#  co może być użyteczne do dalszej analizy semantycznej lub wizualizacji zależności w tekście.
def create_concept_map(input_text):
    doc = nlp(input_text)
    graph = nx.Graph()

    for chunk in doc.noun_chunks:
        graph.add_node(chunk.text)
        for entity in doc.ents:
            graph.add_edge(chunk.text, entity.text)
    
    st.session_state['concept_map'] = graph

def generate_text_based_on_user_input(generation_choice, additional_input, length_option):
    if 'embeddings' not in st.session_state or 'ner_results' not in st.session_state or 'topic_results' not in st.session_state:
        return "Error: Analysis data not available."

    # Prepare context from the analysis
    entities = ', '.join(ent['text'] for ent in st.session_state['ner_results'])
    topics = '; '.join(f"{key}: {', '.join(val)}" for key, val in st.session_state['topic_results'].items())

    # Formulate the base prompt based on user choice
    if generation_choice == "Dokończenie opowieści":
        prompt = f"Using the provided analysis, continue the given story with a coherent ending. Consider: {additional_input}.\nEntities: {entities}.\nTopics: {topics}."
    else:
        prompt = f"Using the provided analysis, create a new story set in the same world with the existing characters. Consider: {additional_input}.\nEntities: {entities}.\nTopics: {topics}."

    length_options = {
        "Krótki, do 2 stron": 500,
        "Średni, do 3 stron": 1000,
        "Długi, do 5 stron": 1500
    }
    max_tokens = length_options.get(length_option, 500)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a creative assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            n=1,
            stop=None
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred while generating text: {e}"

# Interfejs użytkownika
st.title("Fabryka Opowieści: Asystent Twórczy")

st.header("Krok 1: Analiza Twojego Tekstu")
story_contents = st.text_area("Wprowadź tekst do analizy", height=300)

if st.button("Analizuj Tekst") and story_contents:
    start_time = time.time()

    # Initial message
    st.info("Analizuję tekst, który mi przekazałeś...")

    # Perform analyses
    analyze_and_store_embeddings(story_contents)
    analyze_text_with_ner(story_contents)
    analyze_text_with_topic_modeling(story_contents)
    create_concept_map(story_contents)

    elapsed_time = time.time() - start_time

    # Conditional message based on time elapsed
    if elapsed_time > 20:
        st.info("Dużo tego! Poczekaj jeszcze chwilkę...")

    st.success("Analiza wątków: zrobione")

st.header("Krok 2: Podaj szczegóły dla nowego tekstu")

# Option to choose between continuing the story or creating a new one
generation_choice = st.radio("Wybierz typ opowieści:", [
    "Dokończenie opowieści",
    "Stworzenie nowej opowieści na podstawie podanego tekstu"
])

# Allow the user to provide a description of their preferences
additional_input = st.text_area("Podaj dodatkowe preferencje dla tekstu:", height=100)

# Options for text length
length_option = st.radio("Wybierz długość nowego tekstu:", [
    "Krótki, do 2 stron",
    "Średni, do 3 stron",
    "Długi, do 5 stron"
])

if st.button("Generuj Tekst"):
    # Generate text based on the user's choices and input
    generated_text = generate_text_based_on_user_input(generation_choice, additional_input, length_option)

    st.subheader("Wygenerowany Tekst")
    st.text_area("Rezultat wygenerowanego tekstu", value=generated_text, height=300)

    if generated_text:
        st.download_button(
            label="Pobierz wygenerowany tekst",
            data=generated_text.encode(),
            file_name="generated_text.txt",
            mime='text/plain'
        )