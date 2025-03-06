import streamlit as st
from dotenv import dotenv_values
import openai
from io import BytesIO
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os


# Wczytanie zmiennych środowiskowych
env = dotenv_values(".env")

# Sprawdzenie, czy klucz API istnieje w sesji
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = env.get("OPENAI_API_KEY", "")

# Jeśli nie ma klucza API, poproś użytkownika o jego podanie
if not st.session_state["openai_api_key"]:
    st.session_state["openai_api_key"] = st.sidebar.text_input("Podaj swój OpenAI API Key", type="password")

# Jeśli nadal nie ma klucza API, zatrzymaj aplikację
if not st.session_state["openai_api_key"]:
    st.warning("Musisz dodać swój OpenAI API Key, aby korzystać z aplikacji.")
    st.stop()

# Inicjalizacja klienta OpenAI w `st.session_state` (jeśli jeszcze nie istnieje)
if "openai_client" not in st.session_state:
    st.session_state["openai_client"] = openai.OpenAI(api_key=st.session_state["openai_api_key"])

st.title("Fabryka Niedokończonych Opowieści")

token_cost_per_token = 0.0001
budget_options = {
    "Tekst do 5 PLN": 25000,
    "Tekst do 10 PLN": 50000,
    "Tekst do 15 PLN": 75000,
}

# ✅ Poprawiona funkcja analizy NER
def analyze_text_with_ner(input_text):
    try:
        client = st.session_state["openai_client"]  # Pobranie klienta z sesji

        response = client.chat.completions.create(
            model="o3-mini",  # Zmiana modelu na o3-mini
            messages=[
                {"role": "system", "content": "Identify named entities in the following text."},
                {"role": "user", "content": input_text}
            ],
            max_tokens=500
        )

        entities_info = response.choices[0].message.content.strip().split(', ')
        st.session_state['ner_results'] = entities_info

    except Exception as e:
        st.error(f"Failed to analyze text for entities: {e}")
        st.session_state['ner_results'] = []


# ✅ Poprawiona funkcja analizy tematów
def analyze_text_with_topic_modeling(input_text, num_topics=3):
    vectorizer = CountVectorizer(stop_words='english')
    text_data = vectorizer.fit_transform([input_text])

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(text_data)

    topics = {}
    for index, topic in enumerate(lda.components_):
        topics[f"Topic {index+1}"] = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]

    st.session_state['topic_results'] = topics

# ✅ Poprawiona funkcja tworzenia mapy koncepcyjnej
def create_concept_map(input_text):
    try:
        client = st.session_state["openai_client"]  # Pobranie klienta z sesji

        response = client.chat.completions.create(
            model="o3-mini",
        response = openai.ChatCompletion.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "Analyze relationships and conceptual connections in the text."},
                {"role": "user", "content": input_text}
            ],
            max_tokens=500
            max_tokens=500,
            temperature=0.8,
        )

        concept_relations = response.choices[0].message.content.strip().split('. ')
        concept_relations = response.choices[0].message['content'].strip().split('. ')
        st.session_state['concept_relations'] = concept_relations

    except Exception as e:
        st.error(f"Failed to create concept map: {e}")
        st.session_state['concept_relations'] = []



# Precyzyjne usunięcie marginesów bocznych, ale z 1 cm marginesem
st.markdown("""
    <style>
        /* Ustawienie szerokości kontenera i dodanie marginesów bocznych */
        .block-container {
            padding-left: 1cm !important;
            padding-right: 1cm !important;
            max-width: 100% !important;
        }

        /* Usunięcie domyślnego paddingu dla kolumn */
        div[data-testid="column"] {
            padding-left: 0px !important;
            padding-right: 0px !important;
        }
    </style>
    """, unsafe_allow_html=True)


# Set up wider columns
col1, col2, col3 = st.columns([3, 4, 3], gap="medium")

# Step 1: Text analysis and information extraction

col1, col2, col3 = st.columns([2, 2, 3])

with col1:
    st.header("1. Treść początkowa - wprowadzenie oraz analiza")
    story_contents = st.text_area("Wprowadź tekst do analizy", height=300)


    if "story_contents" not in st.session_state:
        st.session_state["story_contents"] = ""

    if st.button("Analizuj Tekst") and story_contents:
        st.session_state["story_contents"] = story_contents
        st.info("Analizuję tekst, który mi przekazałeś...")
        analyze_text_with_ner(story_contents)
        analyze_text_with_topic_modeling(story_contents)
        create_concept_map(story_contents)
        st.success("Analiza tekstu wykonana poprawnie.")

# Selecting additional options
with col2:
    st.header("2. Tworzenie koncepcji")

    st.subheader("IDEE I REFLEKSJE")
    option1_1 = st.checkbox("KONTYNUACJA")
    option1_2 = st.checkbox("WIELOWARSTWOWE ZAKOŃCZENIA")
    option1_3 = st.checkbox("ZADAWANIE PYTAŃ I DYLEMATY")

    st.subheader("ROZRYWKA")
    option2_1 = st.checkbox("KLASYCZNA FABUŁA PRZYGODOWA")
    option2_2 = st.checkbox("HUMOR I SATYRA")
    option2_3 = st.checkbox("THRILLER I NAPIĘCIE")
    option2_4 = st.checkbox("FANTASTYKA I ŚWIATY WYOBRAŹNI")
    option2_5 = st.checkbox("HORROR I GROZA")

    st.subheader("EDUKACJA")
    option3_1 = st.checkbox("PRZEKAZ WIEDZY")
    option3_2 = st.checkbox("ROZWIJANIE UMIEJĘTNOŚCI KRYTYCZNEGO MYŚLENIA")
    option3_3 = st.checkbox("HISTORIA Z PRZESŁANIEM MORALNYM")
    option3_4 = st.checkbox("JĘZYK I STYL")
    option3_5 = st.checkbox("NAUKA PRZEZ PRZYGODĘ")

    st.subheader("EMOCJONALNE DOŚWIADCZENIE")
    option4_1 = st.checkbox("WZRUSZENIE I NOSTALGIA")
    option4_2 = st.checkbox("BUDOWANIE NAPIĘCIA I NIEPEWNOŚCI")
    option4_3 = st.checkbox("GŁĘBOKIE PRZEŻYCIA PSYCHOLOGICZNE")
    option4_4 = st.checkbox("RADOŚĆ I HUMOR")
    option4_5 = st.checkbox("MOCNE DOZNANIA I EKSTREMALNE EMOCJE")

    st.subheader("EKSPERYMENTY Z WYMYŚLONYM ŚWIATEM")
    option5_1 = st.checkbox("TWORZENIE NOWEGO ŚWIATA")
    option5_2 = st.checkbox("EKSPERYMENTOWANIE ZE STRUKTURĄ OPOWIEŚCI")
    option5_3 = st.checkbox("INNOWACYJNE POSTACIE")
    option5_4 = st.checkbox("ŁĄCZENIE GATUNKÓW")
    option5_5 = st.checkbox("ŚWIAT OPARTY NA NOWEJ TECHNOLOGII LUB MAGII")

with col3:
    st.header("Tworzenie planu kontynuacji opowieści")

    if st.button("Generuj Plan"):
        st.info("Generowanie planu...")

        st.session_state["story_outline"] = []
        ner_summary = ", ".join(st.session_state.get("ner_results", [])[:5])
        topics_summary = ", ".join([", ".join(words) for words in list(st.session_state.get("topic_results", {}).values())[:2]])
        concept_summary = ", ".join(st.session_state.get("concept_relations", [])[:5])

        additional_options = ", ".join(filter(None, [
            "Kontynuacja istniejących idei" if option1_1 else "",
            "Wielowarstwowe zakończenia" if option1_2 else "",
            "Zadawanie pytań i moralne dylematy" if option1_3 else "",
            "Klasyczna fabuła przygodowa" if option2_1 else "",
            "Humor i satyra" if option2_2 else "",
            "Thriller i napięcie" if option2_3 else "",
            "Fantastyka i światy wyobraźni" if option2_4 else "",
            "Horror i groza" if option2_5 else "",
            "Przekaz wiedzy" if option3_1 else "",
            "Rozwijanie umiejętności krytycznego myślenia" if option3_2 else "",
            "Historia z przesłaniem moralnym" if option3_3 else "",
            "Język i styl" if option3_4 else "",
            "Nauka przez przygodę" if option3_5 else "",
            "Wzruszenie i nostalgia" if option4_1 else "",
            "Budowanie napięcia i niepewności" if option4_2 else "",
            "Głębokie przeżycia psychologiczne" if option4_3 else "",
            "Radość i humor" if option4_4 else "",
            "Mocne doznania i ekstremalne emocje" if option4_5 else "",
            "Tworzenie nowego świata" if option5_1 else "",
            "Eksperymentowanie ze strukturą opowieści" if option5_2 else "",
            "Innowacyjne postacie" if option5_3 else "",
            "Łączenie gatunków" if option5_4 else "",
            "Świat oparty na nowej technologii lub magii" if option5_5 else ""
        ]))

        initial_prompt = f"""
            Oto początek opowieści: {st.session_state["story_contents"]}

        Na podstawie poniższego tekstu stwórz 9-punktowy plan kontynuacji opowieści. Wszystko w języku tekstu przekazanego do analizy.

        Uwzględnione opcje: {additional_options}

        Generuj **kolejno każdy punkt**.
        """
        for i in range(9):
            part = "Wprowadzenie" if i < 3 else "Rozwinięcie historii" if i < 6 else "Zakończenie"
            point_prompt = initial_prompt + f"\nGenerate point for: {part}. Całość musi być logiczna, musi być zachowany ciąg przyczynowo skutkowy opowieści."

            # ✅ Sprawdzenie, czy `openai_client` istnieje
            if "openai_client" not in st.session_state:
                st.error("Błąd: Klient OpenAI nie został poprawnie zainicjalizowany.")
                break  # Przerwij pętlę, jeśli klient nie istnieje

            client = st.session_state["openai_client"]

            try:
                response = client.chat.completions.create(  # ✅ Poprawione API OpenAI v1.0+
                    model="o3-mini",
                    messages=[{"role": "user", "content": point_prompt}]
                response = openai.ChatCompletion.create(
                    model="o3-mini",
                    messages=[{"role": "user", "content": point_prompt}],
                    max_tokens=150,
                    temperature=0.8,
                )

                point_content = response.choices[0].message.content.strip()  # ✅ Poprawiony dostęp do odpowiedzi
                point_content = response.choices[0].message['content'].strip()
                st.session_state["story_outline"].append(point_content)

            except Exception as e:
                st.error(f"Błąd podczas generowania punktu {i+1}: {e}")
                break  # Przerwij pętlę w razie błędu, by uniknąć dalszych problemów

                break
            time.sleep(3)

        st.subheader("Plan Kontynuacji opowieści")
        for i, point in enumerate(st.session_state["story_outline"]):
            st.text_area(f"Punkt {i+1}", value=point, height=80)

    st.header("4. Wybierz Budżet Generowania Tekstu")
    selected_budget = st.radio("Wybierz poziom inwestycji", list(budget_options.keys()))
    max_tokens = budget_options[selected_budget]
    st.write(f"Wybrano budżet: {selected_budget}, maksymalna liczba tokenów: {max_tokens}")

    if st.button("Zatwierdź i Generuj Opowieść"):
        st.info("Generowanie historii... Proszę czekać.")

        story_parts = []

        for j in range(0, len(st.session_state["story_outline"]), 3):
            current_plan_points = "\n".join(st.session_state["story_outline"][j:j + 3])

            story_prompt = f"""
            Oto początek historii, który użytkownik podał do analizy:
            {st.session_state["story_contents"]}

            Na podstawie 9-punktowego planu kontynuuj opowieść w formie podobnej do formy tekstu początkowego, uwzględniając styl pisarski.
            PLAN SEGMENT:
            """ + current_plan_points

            # ✅ Pobranie klienta z sesji (upewniamy się, że istnieje)
            if "openai_client" not in st.session_state:
                st.error("Błąd: Klient OpenAI nie został poprawnie zainicjalizowany.")
                break  # Przerwij pętlę, jeśli klient nie istnieje

            client = st.session_state["openai_client"]

            try:
                response = client.chat.completions.create(  # ✅ Poprawione API OpenAI v1.0+
                    model="o3-mini",
                response = openai.ChatCompletion.create(
                    model="o3-mini",
                    messages=[{"role": "user", "content": story_prompt}],
                    max_tokens=1500
                    max_tokens=1500,
                    temperature=0.8,
                )

                segment_content = response.choices[0].message.content.strip()  # ✅ Poprawiona metoda dostępu do odpowiedzi
                segment_content = response.choices[0].message['content'].strip()
                story_parts.append(segment_content)

            except Exception as e:
                st.error(f"Błąd podczas generowania segmentu historii: {e}")
                break  # Przerywamy pętlę w razie błędu, by uniknąć kolejnych błędnych wywołań

                break
            time.sleep(3)

        story = "\n".join(story_parts)

        st.subheader("Wygenerowana Opowieść:")
        st.write(story)

        buffer = BytesIO()
        buffer.write(story.encode("utf-8"))
        buffer.seek(0)
        st.download_button("Pobierz opowieść", data=buffer, file_name="historia.txt", mime="text/plain", key="download_story")

                break
        st.write(story)

        buffer = BytesIO()

        buffer.seek(0)
        st.download_button("Pobierz opowieść", data=buffer, file_name="historia.txt", mime="text/plain", key="download_story")
