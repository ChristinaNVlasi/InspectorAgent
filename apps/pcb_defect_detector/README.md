# PCB Defect Detection System

## 🚀 NOW WITH GRID-BASED LOCALIZATION! 

A RAG-based (Retrieval-Augmented Generation) visual inspection system for detecting defects in PCB boards using CLIP embeddings and similarity search.

**✨ NEW: Grid-Based Inspection** - Precise defect localization at grid-cell level (A1-D4, A5-D8)!

## 🎯 Overview

This system uses state-of-the-art computer vision techniques to detect defects in PCB (Printed Circuit Board) images. It leverages:

- **CLIP Embeddings**: OpenAI's CLIP model for generating image embeddings
- **ChromaDB**: Vector database for efficient similarity search
- **RAG Architecture**: Retrieval-Augmented Generation for intelligent defect detection
- **✨ Grid-Based Processing**: Precise defect localization with 4×4 grid segmentation
- **Dual-Mode Support**: Choose between grid-based (precise) or whole-image (fast) processing
- **FastAPI Backend**: Production-ready REST API with comprehensive endpoints
- **Streamlit UI**: User-friendly web interface for testing

## 🌟 What's New in v2.0

### Grid-Based Localization
- **Precise defect locations**: Know exactly where defects are (e.g., "burned at B1")
- **Per-cell analysis**: Each of the 16 grid cells analyzed independently
- **Multiple defect detection**: Find all defects in a single image
- **Enhanced visualization**: Annotated images with marked defect cells
- **Location-specific RAG**: Search only relevant grid locations in database

### Quick Access
- **Quick Start**: See [GRID_BASED_QUICKSTART.md](GRID_BASED_QUICKSTART.md)
- **Technical Details**: See [GRID_ARCHITECTURE.md](GRID_ARCHITECTURE.md)
- **Summary**: See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## 🏗️ Architecture

```
PCB Image Upload
    ↓
CLIP Encoder (generates 512-dim embedding)
    ↓
Vector Database Search
    ├── Compare with OK samples
    └── Compare with defect samples
    ↓
Similarity Analysis
    ↓
Result: OK / NOT_OK + Defect Info
```

## 📁 Project Structure

```
pcb_defect_detector/
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── build_database.py         # Script to build vector database
├── embeddings/
│   └── clip_embedder.py     # CLIP-based image encoder
├── rag/
│   └── vector_store.py      # ChromaDB vector store
├── models/
│   └── pcb_inspector.py     # Main inspection logic
└── web_ui/
    └── streamlit_app.py     # Streamlit interface
```

## 🚀 Quick Start

### ⚡ Grid-Based Mode (Recommended)

```bash
cd app/pcb_defect_detector

# One-stop setup and start
./setup_grid_system.sh

# Or individual commands:
./setup_grid_system.sh build-db      # Build grid-based database
./setup_grid_system.sh test          # Run tests
./setup_grid_system.sh start-api     # Start API server
```

**API will be available at**: `http://localhost:8000`  
**Documentation**: `http://localhost:8000/docs`

### 📘 Traditional Setup

### 1. Install Dependencies

```bash
cd app/pcb_defect_detector
pip install -r requirements.txt
```

### 2. Build Vector Database

Index all PCB images (OK and defect samples) into the vector database:

```bash
python build_database.py --reset
```

This will:
- Scan all images in `../data/samples/EBS7_TCM/`
- Generate CLIP embeddings for each image
- Store embeddings in ChromaDB with metadata
- Create a searchable vector database

### 3. Run Streamlit Interface

Launch the web UI:

```bash
streamlit run web_ui/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## 📊 Dataset Structure

Expected folder structure for PCB images:

```
app/data/samples/EBS7_TCM/
├── OK/
│   ├── K100377_1.jpg         # Good PCB (front)
│   ├── K100377_2.jpg         # Good PCB (back)
│   └── ...
└── NOT_OK/
    ├── burned_b1/            # Burned defect in area B1
    │   ├── BURNED_B1.png
    │   └── ...
    └── CORROSION_D8/         # Corrosion defect in area D8
        ├── CORROSION_D8.png
        └── ...
```

The system automatically parses:
- **Status**: OK or NOT_OK based on folder structure
- **Defect Type**: From folder name (e.g., "burned", "corrosion")
- **Location**: From folder name (e.g., "B1", "D8")

## 🔍 How It Works

1. **Upload PCB Image**: User uploads an image via Streamlit interface

2. **Generate Embedding**: CLIP model encodes the image into a 512-dimensional vector

3. **Similarity Search**: 
   - Search vector database for similar OK images
   - Search vector database for similar defect images

4. **Analysis**:
   - Compare similarity scores
   - If OK similarity > defect similarity AND > threshold → Status: OK
   - Otherwise → Status: NOT_OK with defect details

5. **Results Display**:
   - Overall status (OK/NOT_OK)
   - Confidence score
   - Defect information (type, location, description)
   - Most similar reference images

## ⚙️ Configuration

Key settings in `config.py`:

```python
# Model
MODEL_CONFIG = {
    "clip_model": "openai/clip-vit-base-patch32",
    "device": "auto"  # Uses CUDA/MPS if available
}

# Detection
DEFECT_CONFIG = {
    "similarity_threshold": 0.75,  # Threshold for OK classification
    "top_k_results": 5              # Number of similar images to show
}
```

## 🎨 Features

- ✅ **Automated Defect Detection**: Classify PCBs as OK or NOT_OK
- 📍 **Defect Localization**: Identify defect type and board location
- 🔎 **Visual Similarity Search**: Show most similar reference images
- 📊 **Confidence Scoring**: Transparency in classification decisions
- 🚀 **Fast Performance**: Efficient vector similarity search
- 🎯 **Extensible**: Easy to add new defect types and PCB types

## 🧪 Testing

1. Build the database with your PCB samples
2. Launch the Streamlit app
3. Upload test images
4. Verify detection accuracy

## 📝 Adding New Defect Types

To add a new defect type:

1. Create a folder in `NOT_OK/` with format: `<defect_type>_<location>`
   - Example: `short_circuit_A5/`

2. Add defect images to the folder

3. Rebuild the database:
   ```bash
   python build_database.py --reset
   ```

4. (Optional) Update `DEFECT_TYPES` in `config.py` for better descriptions

## 🔧 Troubleshooting

**Issue**: "No images found in dataset"
- **Solution**: Check that your PCB images are in the correct folder structure

**Issue**: Low accuracy
- **Solution**: 
  - Add more reference images (both OK and defect)
  - Adjust `similarity_threshold` in config
  - Ensure images are good quality and properly labeled

**Issue**: Slow performance
- **Solution**: 
  - Use GPU if available (CUDA or Apple Silicon MPS)
  - Reduce `top_k_results` in config

## 📚 Technologies Used

- **CLIP**: OpenAI's Contrastive Language-Image Pre-training
- **ChromaDB**: Open-source embedding database
- **PyTorch**: Deep learning framework
- **Streamlit**: Web app framework
- **Transformers**: Hugging Face model library

## 🎓 How to Extend

This system can be enhanced with:
- Multiple PCB board types
- Severity classification (minor/major defects)
- Defect segmentation masks
- Integration with production line cameras
- Automated reporting and alerts
- Multi-language support

---

**Built with ❤️ for PCB Quality Inspection**
