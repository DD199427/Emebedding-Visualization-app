import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/visualize/"

st.title("🔍 Word Embedding 3D Visualization")

# Upload file
uploaded_file = st.file_uploader("Upload Corpus (TXT)", type=["txt"])
query = st.text_input("Enter your query:")

corpus = []
if uploaded_file is not None:
    corpus = uploaded_file.read().decode("utf-8").split("\n")

if st.button("Generate Visualization"):
    if query and corpus:
        payload = {"query": query, "corpus": corpus}
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            st.success("✅ Visualization generated!")
            st.components.v1.html(response.json()["html"], height=600)  # Render HTML output
        else:
            st.error(f"❌ API Error: {response.status_code} - {response.text}")
    else:
        st.warning("⚠️ Upload corpus and enter query!")
