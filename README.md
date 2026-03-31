# SuisenSha

SuisenSha is a movie recommendation engine pipeline combining embedding similarity, sequential transformer models, and RAG-augmented generation to provide personalized recommendations. It operates on the MovieLens-100K dataset.

## Features
- **Data Ingestion**: Automatically downloads and processes the MovieLens-100K dataset.
- **Embedding Recommender (FAISS)**: Uses SentenceTransformers to embed movie titles and genres, forming user profiles from interaction histories.
- **SASRec Neural Recommender**: Implements self-attentive sequential recommendations (Kang & McAuley, 2018) in PyTorch.
- **LLM Concierge (RAG)**: Generates human-readable rationales using DistilGPT-2 based on candidates retrieved from FAISS.
- **FastAPI Backend**: Serves realtime recommendations.
- **Streamlit UI**: An interactive dashboard to inspect user histories, compare recommendation models, and request RAG rationales.

## Setup and Installation

### Prerequisites
- Python 3.9+
- See `requirements.txt` for dependencies.

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## Running the Pipeline

You can run individual phases or the entire end-to-end pipeline through the CLI:

```bash
# Run the entire training, embedding, and evaluation sequence
python -m pipeline.cli --phase 0

# Run specific phases (1-8):
python -m pipeline.cli --phase 1  # Download and parse data
python -m pipeline.cli --phase 2  # Build embeddings and FAISS index
python -m pipeline.cli --phase 3  # Train SASRec
python -m pipeline.cli --phase 4  # Test RAG generation logically
python -m pipeline.cli --phase 5  # Evaluate Phase 2 & 3 Recommenders
python -m pipeline.cli --phase 6  # Start the FastAPI Backend (Port 8000)
python -m pipeline.cli --phase 7  # Start Streamlit UI
python -m pipeline.cli --phase 8  # Batch Pre-compute RAG via SQLite Cache
```

## Documentation

For further information, please refer to the 📖 [Documentation](docs/GUIDE.md).
