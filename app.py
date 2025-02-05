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
        # Display message during the delay with markdown for styling
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
        
        # Safely accessing the response in case of variations in the API response
        if res and "choices" in res and len(res.choices) > 0:
            return res.choices[0].message.content
        else:
            st.error("No valid response received.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
    finally:
        # Clear the processing message after completion
        processing_message.empty()




# Function to read text from files
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

# Check if API key is available
if "openai_api_key" not in st.session_state:
    # First, try to get the key from .env file
    st.session_state["openai_api_key"] = env.get("OPENAI_API_KEY", "")

# Safeguard if no API key present
if not st.session_state.get("openai_api_key"):
    # Component for entering OpenAI API key
    st.session_state["openai_api_key"] = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    if not st.session_state.get("openai_api_key"):
        st.warning("You need to add your OpenAI API key to use this app.")
        st.stop()

# Initialize OpenAI client
openai.api_key = st.session_state["openai_api_key"]

# Application title
st.title("Unfinished Story Production Line")

# Tabs
add_tab, search_tab = st.tabs([
    "Add Stories to be Completed",
    "Add Stories That Inspire You"
])

# Initialize session state storage
if "saved_stories" not in st.session_state:
    st.session_state["saved_stories"] = ["", "", ""]

with add_tab:
    st.header("Add Your Story")

    # Text entry option
    st.session_state["saved_stories"][0] = st.text_area("Enter Story Content 1", value=st.session_state["saved_stories"][0], height=100)
    st.session_state["saved_stories"][1] = st.text_area("Enter Story Content 2", value=st.session_state["saved_stories"][1], height=100)
    st.session_state["saved_stories"][2] = st.text_area("Enter Story Content 3", value=st.session_state["saved_stories"][2], height=100)

    # File upload option
    uploaded_files = st.file_uploader("Or Add Files with Texts (max 3)", type=["txt", "pdf", "doc", "docx"], accept_multiple_files=True)
    file_texts = []
    if uploaded_files:
        for file in uploaded_files[:3]:
            extracted_text = extract_text_from_file(file)
            if extracted_text:
                file_texts.append(extracted_text)
            else:
                st.warning(f"File {file.name} contains no text or cannot be read.")
    
    # Token budget selection
    st.subheader("Select Token Budget for Generation")
    max_tokens = 500  # Default token budget
    if st.button("Up to 5 PLN"):
        max_tokens = 2500
    elif st.button("Up to 10 PLN"):
        max_tokens = 5000
    elif st.button("Up to 20 PLN"):
        max_tokens = 10000

    # Generate story continuation
    if st.button("Generate Continuation") and (any(st.session_state["saved_stories"]) or file_texts):
        all_texts = [t for t in st.session_state["saved_stories"] if t] + file_texts
        for i, story in enumerate(all_texts[:3]):
            result = generate_text(f"Continue this story: {story}", max_tokens)
            st.subheader(f"Generated Continuation {i+1}:")
            st.write(result)

with search_tab:
    st.header("Browse Inspiring Stories")
    st.write("(Feature under construction)")