import streamlit as st
from dotenv import dotenv_values
import openai
from PyPDF2 import PdfReader
import docx
import time

# Load environment variables from .env file
env = dotenv_values(".env")

# Function to generate text using OpenAI with delay
def generate_text(prompt, max_tokens):
    try:
        processing_message = st.empty()  # Create an empty slot for the message
        with st.spinner("Processing..."):
            processing_message.markdown(
                "<h1 style='text-align: center; color: red;'>ATTENTION HUMANKIND! CREATING!</h1>",
                unsafe_allow_html=True
            )
            time.sleep(10)  # 10-second delay

        truncated_prompt = prompt[:4000]  # Ensure the prompt is within a safe character limit
        print(f"Sending prompt: {truncated_prompt[:100]}...")  # Debug: print first 100 characters for inspection

        res = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0,
            messages=[{
                "role": "user",
                "content": truncated_prompt,
            }],
            max_tokens=max_tokens,
        )
        
        print(res)  # Debug: print the entire response to inspect structure and possible errors
        
        if res and "choices" in res and len(res.choices) > 0:
            return res.choices[0].message.content
        else:
            st.error("No valid response received.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
    finally:
        processing_message.empty()

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        return "\n".join([
            page.extract_text() 
            for page in pdf_reader.pages if page.extract_text()
        ])
    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

# Streamlit page configuration
st.set_page_config(page_title="Fabryka Opowieści: Asystent Twórczy", layout="centered")

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = env.get("OPENAI_API_KEY", "")

if not st.session_state.get("openai_api_key"):
    st.session_state["openai_api_key"] = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    if not st.session_state.get("openai_api_key"):
        st.warning("You need to add your OpenAI API key to use this app.")
        st.stop()

openai.api_key = st.session_state["openai_api_key"]

st.title("Unfinished Story Production Line")

add_tab, search_tab = st.tabs([
    "Add Stories to be Completed",
    "Add Stories That Inspire You"
])

if "saved_stories" not in st.session_state:
    st.session_state["saved_stories"] = ["", "", ""]

with add_tab:
    st.header("Add Your Story")

    st.session_state["saved_stories"][0] = st.text_area("Enter Story Content 1", value=st.session_state["saved_stories"][0], height=100)
    st.session_state["saved_stories"][1] = st.text_area("Enter Story Content 2", value=st.session_state["saved_stories"][1], height=100)
    st.session_state["saved_stories"][2] = st.text_area("Enter Story Content 3", value=st.session_state["saved_stories"][2], height=100)

    user_description = st.text_input("Please write what kind of text you would like to create.", placeholder="e.g., I want a thrilling detective story")

    uploaded_files = st.file_uploader("Or Add Files with Texts (max 3)", type=["txt", "pdf", "doc", "docx"], accept_multiple_files=True)
    file_texts = []
    if uploaded_files:
        for file in uploaded_files[:3]:
            extracted_text = extract_text_from_file(file)
            if extracted_text:
                file_texts.append(extracted_text)
            else:
                st.warning(f"File {file.name} contains no text or cannot be read.")

    continuation_type = st.radio("Choose a continuation option:", [
        "Continuation in the form of a short story",
        "Continuation in the form of a novel chapter",
        "A new piece, not a continuation, based on the provided content."
    ])

    st.text("GOAL: Describe the aims of your story.")

    # Setting initial goal percentages
    if "goals" not in st.session_state:
        st.session_state.goals = {
            "Ideas and Reflections": 20,
            "Entertainment": 20,
            "Education": 20,
            "Building Emotional Experience": 20
        }

    def adjust_slider(new_value, changed_key):
        total = new_value + sum(v for k, v in st.session_state.goals.items() if k != changed_key)
        diff = 100 - total
        if diff != 0:
            for k in st.session_state.goals.keys():
                if k != changed_key:
                    st.session_state.goals[k] += diff // (len(st.session_state.goals) - 1)
        return new_value

    # Sliders for goals
    st.session_state.goals["Ideas and Reflections"] = st.slider(
        "Ideas and Reflections", 0, 100, st.session_state.goals["Ideas and Reflections"],
        on_change=adjust_slider, args=(st.session_state.goals["Ideas and Reflections"], "Ideas and Reflections")
    )
    st.session_state.goals["Entertainment"] = st.slider(
        "Entertainment", 0, 100, st.session_state.goals["Entertainment"],
        on_change=adjust_slider, args=(st.session_state.goals["Entertainment"], "Entertainment")
    )
    st.session_state.goals["Education"] = st.slider(
        "Education", 0, 100, st.session_state.goals["Education"],
        on_change=adjust_slider, args=(st.session_state.goals["Education"], "Education")
    )
    st.session_state.goals["Building Emotional Experience"] = st.slider(
        "Building Emotional Experience", 0, 100, st.session_state.goals["Building Emotional Experience"],
        on_change=adjust_slider, args=(st.session_state.goals["Building Emotional Experience"], "Building Emotional Experience")
    )

    if sum(st.session_state.goals.values()) != 100:
        st.warning("The total percentage should be exactly 100%.")

    st.subheader("Select Token Budget for Generation")
    max_tokens = 5000  # Default token budget
    if st.button("Up to 5 PLN"):
        max_tokens = 5000
    elif st.button("Up to 10 PLN"):
        max_tokens = 25000
    elif st.button("Up to 20 PLN"):
        max_tokens = 10000

    if st.button("Generate") and (any(st.session_state["saved_stories"]) or file_texts):
        all_texts = [t for t in st.session_state["saved_stories"] if t] + file_texts
        for i, story in enumerate(all_texts[:3]):
            if continuation_type == "Continuation in the form of a short story":
                prompt = f"Write a short story continuation for the following content: {story}"
            elif continuation_type == "Continuation in the form of a novel chapter":
                prompt = f"Write the next chapter as part of a novel for the following content: {story}"
            elif continuation_type == "A new piece, not a continuation, based on the provided content.":
                prompt = f"Create a new story based on the world and facts in the following content: {story}"

            prompt += f"\n\nUser description: {user_description}"

            goals_text = "\n".join([f"{key}: {value}%" for key, value in st.session_state.goals.items()])
            prompt += f"\n\nStory Goals:\n{goals_text}"

            result = generate_text(prompt, max_tokens)
            st.subheader(f"Generated {continuation_type} {i+1}:")
            st.write(result)

with search_tab:
    st.header("Browse Inspiring Stories")
    st.write("(Feature under construction)")