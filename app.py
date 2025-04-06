from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import uvicorn
import os
import requests

app = FastAPI()

class VisualizationRequest(BaseModel):
    query: str
    corpus: List[str]

@app.get("/")
def home():
    return {"message": "FastAPI is running. Use POST /visualize/ to generate visualizations."}

def load_glove_embeddings():
    pkl_path = "glove_embeddings.pkl"

    if not os.path.exists(pkl_path):
        print("🔽 Downloading GloVe embeddings from cloud...")
        url = "https://drive.google.com/file/d/1wbLSdPm5DDIspj_Ojuq4nlwZXV_aiWQJ/view?usp=drive_link/glove_embeddings.pkl"
        r = requests.get(url)
        with open(pkl_path, "wb") as f:
            f.write(r.content)

    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def embed_sentence(sentence, embeddings, dim=50):
    words = sentence.split()
    word_vectors = [embeddings[word] for word in words if word in embeddings]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(dim)

def compare_query_to_corpus(query, corpus, embeddings):
    query_vector = embed_sentence(query, embeddings)
    scores = []
    for sentence in corpus:
        vec = embed_sentence(sentence.strip(), embeddings)
        sim = np.dot(query_vector, vec) / (np.linalg.norm(query_vector) * np.linalg.norm(vec) + 1e-8)
        scores.append((sentence, sim))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def reduce_to_3d(vectors):
    pca = PCA(n_components=3)
    return pca.fit_transform(vectors)

def generate_3d_plot(vectors_3d, labels):
    x, y, z = vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2]
    fig = go.Figure()
    for i, label in enumerate(labels):
        color = 'red' if i == 0 else 'blue'
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
        title="3D Word Embedding Visualization",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig.to_html(full_html=False)

@app.post("/visualize/")
async def visualize_embeddings(request: VisualizationRequest):
    try:
        embeddings = load_glove_embeddings()
        top_sentences = compare_query_to_corpus(request.query, request.corpus, embeddings)[:5]

        vectors = [embed_sentence(request.query, embeddings)] + [embed_sentence(s, embeddings) for s, _ in top_sentences]
        labels = ["Query"] + [s for s, _ in top_sentences]
        vectors_3d = reduce_to_3d(np.array(vectors))
        plot_html = generate_3d_plot(vectors_3d, labels)

        return {"html": plot_html}
    except Exception as e:
        print("🔥 ERROR:", e)
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
