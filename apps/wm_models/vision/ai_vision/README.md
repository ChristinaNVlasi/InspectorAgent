# AI Vision System for Alternator Component Analysis

## 📁 Project Structure

```
ai_vision/
├── requirements.txt          # Dependencies
├── config.py                # Configuration settings
├── main.py                  # Main application entry point
├── data/                    # Data processing and management
│   ├── dataset_manager.py   # Dataset loading and splitting
│   └── data_validator.py    # Image quality validation
├── preprocessing/           # Image preprocessing pipeline
│   ├── image_processor.py   # Core image processing
│   ├── augmentation.py      # Data augmentation
│   └── quality_check.py     # Image quality assessment
├── embeddings/             # Feature extraction and embeddings
│   ├── clip_embedder.py    # CLIP-based embeddings
│   ├── vit_embedder.py     # Vision Transformer embeddings
│   └── embedding_utils.py  # Utility functions
├── models/                 # ML models and training
│   ├── classifier.py       # Component condition classifier
│   ├── trainer.py          # Model training pipeline
│   └── model_utils.py      # Model utilities
├── rag/                    # RAG system components
│   ├── vector_store.py     # ChromaDB vector database
│   ├── retriever.py        # Similarity search
│   ├── llm_interface.py    # LLM integration
│   └── prompt_templates.py # Prompt engineering
├── api/                    # FastAPI backend
│   ├── app.py              # FastAPI application
│   ├── routes.py           # API endpoints
│   └── schemas.py          # Pydantic models
├── web_ui/                 # Frontend interface
│   ├── streamlit_app.py    # Streamlit interface
│   └── components/         # UI components
└── utils/                  # Utilities and helpers
    ├── logger.py           # Logging configuration
    ├── metrics.py          # Evaluation metrics
    └── visualization.py    # Result visualization
```

## 🚀 Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Configure settings in `config.py`
3. Prepare dataset: `python data/dataset_manager.py`
4. Build embeddings: `python embeddings/clip_embedder.py`
5. Setup vector store: `python rag/vector_store.py`
6. Run web interface: `streamlit run web_ui/streamlit_app.py`

## 🔄 Workflow

1. **Data Preparation**: Process and validate image dataset
2. **Embedding Generation**: Create vector representations
3. **Vector Store Setup**: Index embeddings in ChromaDB
4. **Model Training**: Fine-tune classification model
5. **RAG Pipeline**: Implement retrieval and generation
6. **Web Interface**: Deploy user-friendly interface



### How to run 

cd ai_vision

python main.py build

python main.py web 