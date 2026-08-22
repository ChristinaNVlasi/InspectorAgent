# 🚀 Quick Start Guide - PCB Defect Detection System

## ✅ System Successfully Created!

Your PCB defect detection system is now ready to use! Here's what was built:

### 📦 What You Got:

1. **RAG-based Detection System** - Uses CLIP embeddings and similarity search
2. **Vector Database** - ChromaDB with 11 PCB images indexed (4 OK, 7 defects)
3. **Streamlit Web UI** - User-friendly interface for testing
4. **Complete Architecture** - Modular, extensible, production-ready code

---

## 🎯 How to Use

### Option 1: Web Interface (Recommended)

The Streamlit app is **already running** at:
- **Local URL:** http://localhost:8503
- **Network URL:** http://192.168.1.143:8503

**How to test:**
1. Open the URL in your browser
2. Upload a PCB image (drag & drop or browse)
3. Click "🔍 Inspect PCB"
4. View results: Status, confidence, defect details, similar images

### Option 2: Python Script

Test programmatically:

```bash
cd /Users/christinavlasi/Documents/GitHub/rEUman--Knorr-ML/app/pcb_defect_detector
python test_inspector.py
```

### Option 3: Python Code

```python
from models.pcb_inspector import PCBInspector

inspector = PCBInspector()
result = inspector.inspect_pcb("path/to/pcb_image.jpg")

print(f"Status: {result['status']}")
print(f"Confidence: {result['confidence']:.2%}")
if result['defect_info']:
    print(f"Defect: {result['defect_info']['defect_type']}")
    print(f"Location: {result['defect_info']['location']}")
```

---

## 📊 Current Database Status

- **Total Images:** 11
- **OK Samples:** 4 (front and back views)
- **Defect Samples:** 7
  - Burned B1: 4 images
  - Corrosion D8: 3 images

---

## 🔧 Adding More PCB Images

To improve accuracy, add more samples:

### 1. Add Images to Dataset

```
app/data/samples/EBS7_TCM/
├── OK/
│   └── [Add more OK PCB images here]
└── NOT_OK/
    ├── burned_b1/
    ├── CORROSION_D8/
    └── [Create new defect folders]
```

**Naming Convention for Defects:**
- Format: `<defect_type>_<location>/`
- Examples:
  - `crack_C3/` - Crack in area C3
  - `missing_A1/` - Missing component in area A1
  - `solder_E5/` - Soldering defect in area E5

### 2. Rebuild Database

```bash
cd /Users/christinavlasi/Documents/GitHub/rEUman--Knorr-ML/app/pcb_defect_detector
python build_database.py --reset
```

### 3. Restart Streamlit

The app will automatically reload with the new database.

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Adjust detection sensitivity
DEFECT_CONFIG = {
    "similarity_threshold": 0.75,  # Lower = more sensitive
    "top_k_results": 5              # Number of similar images
}

# Add new defect types
DEFECT_TYPES = {
    "your_new_defect": {
        "description": "Description of defect",
        "severity": "high",  # or "medium", "low"
        "keywords": ["keyword1", "keyword2"]
    }
}
```

---

## 🎨 Features Demonstrated

✅ **Image Embeddings** - CLIP model converts images to 512-D vectors
✅ **Vector Search** - ChromaDB finds similar PCBs in milliseconds
✅ **Defect Classification** - OK vs NOT_OK with confidence scores
✅ **Defect Localization** - Identifies defect type and board location
✅ **Visual Reference** - Shows most similar images from database
✅ **Web Interface** - Professional Streamlit UI for testing
✅ **Modular Architecture** - Easy to extend and customize

---

## 📈 Next Steps to Improve

1. **Add More Samples**: 50-100 images per defect type for best results
2. **Balance Dataset**: Equal number of OK and defect samples
3. **Multiple Views**: Front and back of same PCB for better matching
4. **Fine-tune Threshold**: Adjust based on your accuracy requirements
5. **Add Segmentation**: Highlight exact defect location on image
6. **Multiple PCB Types**: Support different board models

---

## 🔍 System Architecture

```
Query PCB Image
    ↓
CLIP Encoder (512-D embedding)
    ↓
Vector Database (ChromaDB)
    ├─→ Search OK samples
    └─→ Search defect samples
    ↓
Similarity Comparison
    ↓
Classification + Defect Info
    ↓
Streamlit Display
```

---

## 📝 Project Structure

```
pcb_defect_detector/
├── config.py                    # All configuration
├── requirements.txt             # Dependencies
├── build_database.py            # Index images
├── test_inspector.py            # Test script
├── embeddings/
│   └── clip_embedder.py        # CLIP image encoder
├── rag/
│   └── vector_store.py         # ChromaDB wrapper
├── models/
│   └── pcb_inspector.py        # Main detection logic
└── web_ui/
    └── streamlit_app.py        # Web interface
```

---

## 🐛 Troubleshooting

**App not loading?**
```bash
# Restart Streamlit
cd /Users/christinavlasi/Documents/GitHub/rEUman--Knorr-ML/app/pcb_defect_detector
streamlit run /Users/christinavlasi/Documents/GitHub/rEUman--Knorr-ML/app/pcb_defect_detector/web_ui/streamlit_app.py --server.port 8503
```

**Database empty?**
```bash
python build_database.py --reset
```

**Import errors?**
```bash
pip install -r requirements.txt
```

---

## 🎓 How It Works

1. **Training Phase** (build_database.py):
   - Scans all PCB images in dataset
   - Generates CLIP embeddings (512-D vectors)
   - Stores in ChromaDB with metadata (status, defect type, location)

2. **Inference Phase** (pcb_inspector.py):
   - User uploads PCB image
   - Generate embedding for uploaded image
   - Search vector DB for similar OK and defect images
   - Compare similarity scores
   - Return classification + most similar references

3. **Decision Logic**:
   - If OK_similarity > defect_similarity AND OK_similarity > threshold → OK
   - Otherwise → NOT_OK with defect details from best match

---

## 📚 Technologies Used

- **CLIP** - OpenAI's vision-language model
- **ChromaDB** - Vector database for embeddings
- **PyTorch** - Deep learning backend
- **Streamlit** - Web interface
- **Transformers** - Hugging Face models

---

## ✨ Success!

Your PCB defect detection system is fully operational! 🎉

**Try it now:** http://localhost:8503

For questions or improvements, check the code comments or README.md.

---

**Built with ❤️ for Quality Inspection**
