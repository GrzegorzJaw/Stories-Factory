import streamlit as st
from dotenv import dotenv_values
import openai
from io import BytesIO
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation



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
            max_completion_tokens=500,
            reasoning_effort="high"  # Możliwe wartości: "low", "medium", "high"
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
        client = st.session_state["openai_client"]  # Pobranie klienta OpenAI z sesji

        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "Analyze relationships and conceptual connections in the text."},
                {"role": "user", "content": input_text}
            ],
            max_completion_tokens=500,
            reasoning_effort="high"  # Możliwe wartości: "low", "medium", "high"
        )

        # ✅ Poprawny sposób dostępu do treści odpowiedzi
        concept_map_text = response.choices[0].message.content.strip()

        # Podział tekstu na punkty (np. lista zdań)
        concept_relations = concept_map_text.split('. ')

        # Zapisanie w stanie sesji Streamlit
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

        # ✅ Jeden request do API zamiast pętli!
        full_prompt = f"""
            Oto początek opowieści: {st.session_state["story_contents"]}

            Analiza:
            - Rozpoznane byty NER: {ner_summary}
            - Tematy tekstu: {topics_summary}
            - Kluczowe pojęcia i relacje: {concept_summary}
            - Uwzględnione opcje narracyjne: {additional_options}

            Na tej podstawie **wygeneruj szczegółowy 9-punktowy plan kontynuacji opowieści**.
            Każdy punkt powinien być rozwinięty, uwzględniać spójność fabularną i logiczny ciąg przyczynowo-skutkowy.
            Struktura planu:
            1. Wprowadzenie - przedstawienie dalszej części świata i postaci
            2. Wprowadzenie - rozwinięcie tła i ekspozycja fabularna
            3. Wprowadzenie - zawiązanie akcji i pierwszy konflikt
            4. Rozwinięcie - eskalacja problemów i decyzji bohaterów
            5. Rozwinięcie - pojawienie się nieoczekiwanych zwrotów akcji
            6. Rozwinięcie - kulminacja głównego wątku
            7. Zakończenie - początek rozwiązania fabularnego
            8. Zakończenie - finalne wybory bohaterów i konsekwencje
            9. Zakończenie - ostateczne domknięcie historii lub otwarte zakończenie
        """

        try:
            client = st.session_state["openai_client"]  # Pobranie klienta z sesji
            
            response = client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": full_prompt}],
                max_completion_tokens=1500,
                reasoning_effort="medium"  # ✅ Medium = szybsze generowanie
            )

            plan_text = response.choices[0].message.content.strip()
            st.session_state["story_outline"] = plan_text.split('\n')[:9]

            st.subheader("Plan Kontynuacji Opowieści")
            for i, point in enumerate(st.session_state["story_outline"]):
                st.text_area(f"Punkt {i+1}", value=point, height=80)

        except Exception as e:
            st.error(f"Błąd podczas generowania planu: {e}")

    st.header("4. Wybierz Budżet Generowania Tekstu")
    selected_budget = st.radio("Wybierz poziom inwestycji", list(budget_options.keys()))
    max_completion_tokens = budget_options[selected_budget]
    st.write(f"Wybrano budżet: {selected_budget}, maksymalna liczba tokenów: {max_completion_tokens}")

    if st.button("Zatwierdź i Generuj Opowieść"):
        st.info("Generowanie historii... Proszę czekać.")
        
        complete_story_prompt = f"""
        Oto początek historii, który użytkownik podał do analizy:
        {st.session_state["story_contents"]}

        Na bazie poniższego planu, wygeneruj pełną opowieść w formie narracyjnej z dialogami:
        {' '.join(st.session_state['story_outline'])}
        
        Uwzględnij opcje narracyjne oraz wyniki analizy, tworząc logiczną i spójną kontynuację w odpowiadającym stylu.
        """

        try:

            client = st.session_state["openai_client"]  # Pobranie klienta z sesji

            response = client.chat.completions.create(  # ✅ Poprawione API OpenAI v1.0+
                    model="o3-mini",
                    messages=[{"role": "user", "content": complete_story_prompt}],
                    max_completion_tokens=3000,
                    reasoning_effort="high"  # Możliwe wartości: "low", "medium", "high"
                )

            if response and hasattr(response, "choices") and len(response.choices) > 0:
                story = response.choices[0].message.content.strip()
                if not story:
                    st.error("Otrzymano pustą odpowiedź od OpenAI.")
            else:
                st.error("Nie udało się wygenerować odpowiedzi od OpenAI.")
                story = "Błąd generowania opowieści."

            st.subheader("Wygenerowana Opowieść:")
            st.write(story)

            buffer = BytesIO()
            buffer.write(story.encode("utf-8"))
            buffer.seek(0)
            st.download_button("Pobierz opowieść", data=buffer, file_name="historia.txt", mime="text/plain", key="download_story")

        except Exception as e:
            st.error(f"Błąd podczas generowania historii: {e}")