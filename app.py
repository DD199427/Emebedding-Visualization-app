from fastapi import FastAPI, UploadFile, File
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import os

# Initialize FastAPI app
app = FastAPI()


@app.get("/")
def home():
    return {"message": "FastAPI is running! Use /docs to test the API."}




GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DIR = "glove.6B"
GLOVE_FILE = "glove.6B.50d.txt"

# Ensure embeddings exist
if not os.path.exists(GLOVE_FILE):
    print("🔽 Downloading GloVe embeddings...")
    os.system(f"wget {GLOVE_URL} -O glove.zip && unzip glove.zip && rm glove.zip")

def load_glove_embeddings():
    embeddings = {}
    with open(GLOVE_DIR + "/" + GLOVE_FILE, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = list(map(float, values[1:]))
            embeddings[word] = vector
    return embeddings

# Convert sentence to embedding (average of word embeddings)
def embed_sentence(sentence, embeddings):
    words = sentence.split()
    word_vectors = [embeddings[word] for word in words if word in embeddings]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(50)

# Compare query to corpus using cosine similarity
def compare_query_to_corpus(query, corpus, embeddings):
    query_vector = embed_sentence(query, embeddings)
    similarity_scores = []
    
    for sentence in corpus:
        sentence_vector = embed_sentence(sentence.strip(), embeddings)
        similarity = np.dot(query_vector, sentence_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(sentence_vector))
        similarity_scores.append((sentence, similarity))
    
    return sorted(similarity_scores, key=lambda x: x[1], reverse=True)

# Reduce vectors to 3D using PCA
def reduce_to_3d(vectors):
    pca = PCA(n_components=3)
    return pca.fit_transform(vectors)

# Generate interactive 3D visualization
def generate_3d_plot(vectors_3d, labels):
    x, y, z = vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2]

    fig = go.Figure()
    for i, label in enumerate(labels):
        color = 'red' if i == 0 else 'blue'  # Highlight query
        size = 15 if i == 0 else 10  
        fig.add_trace(
            go.Scatter3d(
                x=[x[i]], y=[y[i]], z=[z[i]],
                mode='markers+text',
                marker=dict(size=size, color=color),
                text=[label],
                textposition="top center"
            )
        )

    fig.update_layout(
        title="Word Embedding 3D Visualization",
        scene=dict(
            xaxis_title="PCA Dimension 1",
            yaxis_title="PCA Dimension 2",
            zaxis_title="PCA Dimension 3"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig.to_html(full_html=False)  # Convert Plotly figure to HTML snippet

# Define request model
class VisualizationRequest(BaseModel):
    query: str
    corpus: List[str]

@app.post("/visualize/")
async def visualize_embeddings(request: VisualizationRequest):
    query = request.query
    corpus = request.corpus
    
    glove_path = "glove.6B/glove.6B.50d.txt"  # Path to GloVe embeddings
    embeddings = load_glove_embeddings(glove_path)

    # Compute similarity
    top5_similar_sentences = compare_query_to_corpus(query, corpus, embeddings)[:5]

    # Collect vectors and labels
    query_vector = embed_sentence(query, embeddings)
    labels = ["Query"] + [sentence for sentence, _ in top5_similar_sentences]
    vectors = [query_vector] + [embed_sentence(sentence, embeddings) for sentence, _ in top5_similar_sentences]

    # Reduce dimensions to 3D and plot
    vectors_3d = reduce_to_3d(vectors)
    plot_html = generate_3d_plot(vectors_3d, labels)

    return {"html": plot_html}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
