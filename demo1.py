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

def chatbot_page1():
    from PIL import Image, ImageOps, ImageFilter
    st.markdown("""
                        <div class='header-container'>
                            <h2>Image Processing & OCR.</h2>
                        </div>
                    """, unsafe_allow_html=True)

    
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
        menu = st.sidebar.radio("Navigate to", ["text summ","Image Enhancement","Text-to-Image caption generator"])
        if menu == "text summ":
            chatbot_page2()
        elif menu == "Text-to-Image caption generator":
            chatbot_page4()

if __name__ == "__main__":
    main()
