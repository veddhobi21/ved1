import hashlib
from dotenv import load_dotenv
import os
import sqlite3
import streamlit as st
# Load environment variables
load_dotenv()

# Initialize session states
if "login_status" not in st.session_state:
    st.session_state["login_status"] = False
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None


# Database Setup
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()
def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        conn.close()
        return "Registration successful!"
    except sqlite3.IntegrityError:
        return "Username already exists. Please choose a different one."

def login_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password_hash))
    user = c.fetchone()
    conn.close()
    return user is not None

# --- Session State Initialization ---
if "login_status" not in st.session_state:
    st.session_state["login_status"] = False

if "current_user" not in st.session_state:
    st.session_state["current_user"] = None

# --- Registration Page ---
def registration_page():
    st.title("🔐 Register")
    with st.form("register_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Register")
        if submitted:
            result = register_user(username, password)
            if "successful" in result:
                st.success(result)
            else:
                st.error(result)

# --- Login Page ---
def login_page():
    st.title("🔓 Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if login_user(username, password):
                st.session_state["login_status"] = True
                st.session_state["current_user"] = username
                st.success(f"Welcome, {username}!")
            else:
                st.error("Invalid username or password.")
def chatbot_page3():
    st.markdown("""
            <div class='header-container'>
                <h1>🤖Advance PDF Q-A Chatbot</h1>
                <p>Chat with AI, extract PDF insights, and generate reports.</p>
            </div>
        """, unsafe_allow_html=True)
    # import pdfplumber
    # def extract_text_from_pdf(pdf_file):
    #     pdf_text = ""
    #     with pdfplumber.open(pdf_file) as pdf:
    #         for page in pdf.pages:
    #             pdf_text += page.extract_text()
    #     return pdf_text
    #
    # def answer_question(text, question):
    #     # Simple keyword-based search
    #     sentences = text.split('. ')
    #     sentences = [s.strip() for s in sentences if s]
    #     best_sentence = max(sentences, key=lambda s: s.lower().count(question.lower()),
    #                         default="No relevant answer found.")
    #     return best_sentence
    #
    # st.title("PDF Chatbot")
    #
    # # File uploader
    # pdf_file = st.file_uploader("Upload a PDF", type="pdf")
    #
    # if pdf_file:
    #     # Extract text from PDF
    #     pdf_text = extract_text_from_pdf(pdf_file)
    #     st.write("PDF text extracted successfully.")
    #
    #     # Display extracted text (for debugging or confirmation)
    #     st.text_area("Extracted PDF Text", pdf_text, height=300)
    #
    #     # Question input
    #     question = st.text_input("Ask a question about the PDF content:")
    #
    #     if question:
    #         # Get answer
    #         answer = answer_question(pdf_text, question)
    #         st.write("Answer:", answer)
    from PyPDF2 import PdfReader
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    from nltk import sent_tokenize
    import nltk
    nltk.download('punkt_tab')
    import nltk
    print(nltk.data.path)


    # Load Models
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

    # Initialize FAISS Index
    dimension = 384  # Embedding size for the model
    index = faiss.IndexFlatL2(dimension)
    text_data = []  # Store text chunks for reference

    # Function to Extract Text from PDF
    def extract_text_from_pdf(pdf_file):
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    # Function to Split Text into Logical Chunks
    def split_text_to_chunks(text, max_length=500):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    # Streamlit App
    # st.title("Advanced PDF Q&A Chatbot")

    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_pdf:
        with st.spinner("Processing the PDF..."):
            raw_text = extract_text_from_pdf(uploaded_pdf)
            chunks = split_text_to_chunks(raw_text)

            # Embed chunks and build FAISS index
            embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
            embeddings = np.array(embeddings)
            index.add(embeddings)
            text_data.extend(chunks)

        st.success("PDF uploaded and indexed!")

    query = st.text_input("Ask a question:")
    if query:
        # Embed the query and find relevant chunks
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)
        distances, indices = index.search(np.array(query_embedding), k=3)  # Retrieve top 3 chunks

        # Refine the answer with extractive Q&A
        candidate_answers = []
        for idx in indices[0]:
            chunk = text_data[idx]
            result = qa_model(question=query, context=chunk)
            candidate_answers.append((result['answer'], result['score'], chunk))

        # Sort by confidence and display the best answer
        best_answer = max(candidate_answers, key=lambda x: x[1])
        st.write(f"**Answer:** {best_answer[0]}")
        st.write(f"**Confidence:** {best_answer[1]:.2f}")
        st.write(f"**Relevant Context:** {best_answer[2]}")

        # Log the top 3 answers for transparency
        with st.expander("See other possible answers"):
            for ans, score, context in candidate_answers:
                st.write(f"- **Answer:** {ans}\n  **Confidence:** {score:.2f}\n  **Context:** {context[:200]}...")

    # Summarization for Quick Overview
    if st.checkbox("Summarize PDF Content"):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(raw_text, max_length=150, min_length=30, do_sample=False)
        st.write("**Summary:**")
        st.write(summary[0]['summary_text'])


    
def chatbot_page2():
    import streamlit as st

    st.markdown("""
                <div class='header-container'>
                    <h1>🤖Advanced Text Summarization</h1>
                    <p>Chat with AI, extract PDF insights, and generate reports.</p>
                </div>
            """, unsafe_allow_html=True)
    import streamlit as st
    from transformers import pipeline

    # Title and description
    # st.title("Advanced Text Summarization with Streamlit")
    st.write("Enter a long text, and this application will summarize it for you using advanced NLP models.")

    # Sidebar for model selection
    st.sidebar.title("Summarization Options")
    model_name = st.sidebar.selectbox(
        "Choose a pre-trained model:",
        ("facebook/bart-large-cnn", "t5-small", "t5-base", "t5-large")
    )

    # Load summarization pipeline
    @st.cache_resource
    def load_summarization_model(model_name):
        return pipeline("summarization", model=model_name)

    summarizer = load_summarization_model(model_name)

    # Input text
    st.subheader("Input Text")
    text = st.text_area("Paste your text here for summarization:", height=200)

    # Summarization parameters
    st.sidebar.subheader("Parameters")
    max_length = st.sidebar.slider("Maximum length of summary:", 30, 300, 130)
    min_length = st.sidebar.slider("Minimum length of summary:", 10, 100, 30)
    do_sample = st.sidebar.checkbox("Use sampling for summarization", value=False)

    # Generate summary
    if st.button("Summarize"):
        if text.strip():
            st.write("Summarizing...")
            try:
                summary = summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=do_sample,
                )[0]['summary_text']
                st.subheader("Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to summarize.")




import streamlit as st
from transformers import BlipForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def chatbot_page4():

    # Load the BLIP model and processor
    @st.cache_resource
    def load_model():
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model

    processor, model = load_model()

    # Caption generation function
    def generate_caption(image):
        inputs = processor(images=image, return_tensors="pt").to("cpu")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    # Streamlit App
    st.title("🖼️ Text-to-Image Caption Generator")
    st.write("Upload an image, and the app will generate a caption for it.")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate caption
        with st.spinner("Generating caption..."):
            caption = generate_caption(image)

        st.subheader("Generated Caption")
        st.write(f"**{caption}**")

        # Option to refine the caption
        st.subheader("Refine Caption")
        refined_caption = st.text_input("Modify the generated caption:", value=caption)

        if refined_caption:
            st.success("Caption updated successfully!")
            st.write(f"**Refined Caption:** {refined_caption}")

# Main Application Logic
def main():
    init_db()
    st.sidebar.title("Navigation")
    if not st.session_state["login_status"]:
        page = st.sidebar.selectbox("Choose a page", ["Login", "Register"])
        if page == "Login":
            login_page()
        elif page == "Register":
            registration_page()
    else:
        menu = st.sidebar.radio("Navigate to", ["text summ","PDF Q-A","Text-to-Image caption generator"])
        if menu == "text summ":
            chatbot_page2()
        elif menu == "PDF Q-A":
            chatbot_page3()
        elif menu == "Text-to-Image caption generator":
            chatbot_page4()

if __name__ == "__main__":
    main()
