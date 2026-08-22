# Beko RGS - Noise Diagnosis System

Production-ready AI system for diagnosing washing machine problems based on noise recordings.

## 🎯 Overview

This system uses **audio signal processing** with **similarity-based classification** to identify washing machine issues from noise samples. It's optimized for few-shot learning with limited training data.

## 🏗️ Architecture

- **Audio Processing**: Mel spectrograms, MFCCs, and spectral features
- **Data Augmentation**: Pitch shift, time stretch, noise injection
- **Classification**: Cosine similarity matching with reference embeddings
- **API**: Flask REST API with CORS support
- **UI**: Beautiful HTML/CSS interface with audio recording

## 📋 Detected Issues

1. Bearing worn out
2. Counterweight loose
3. Foot adjustment wrong
4. Motor noise
5. Shock absorber fault
6. Springs loose
7. Water pump faulty

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/christinavlasi/Documents/GitHub/rEUman-Arcelik-ML/app/RGS_v1/noise
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python noise_classifier.py
```

This will:
- Load audio samples from `data/` folder
- Apply data augmentation (10x per sample)
- Extract audio features
- Build reference embeddings
- Save model to `noise_classifier_model.pkl`

### 3. Start the API Server

```bash
python api.py
```

Server will start on `http://localhost:5001`

### 4. Open the Web Interface

Open `../../frontend/index.html` in your browser.

## 🔧 API Usage

### Health Check
```bash
curl http://localhost:5001/health
```

### Diagnose Noise
```bash
curl -X POST http://localhost:5001/diagnose \
  -F "audio=@recording.wav" \
  -F "model_id=WM-2024-001"
```

**Response:**
```json
{
  "success": true,
  "model_id": "WM-2024-001",
  "diagnosis": "bearing worn out",
  "confidence": 87.5,
  "confidence_scores": {
    "bearing worn out": 87.5,
    "motor noise": 45.2,
    "shock absorber fault": 32.1,
    ...
  },
  "recommendation": "The bearing is worn out. Recommend replacing the drum bearing assembly."
}
```

## 📁 Project Structure

```
noise/
├── audio_processor.py      # Audio processing & augmentation
├── noise_classifier.py     # Similarity-based classifier
├── api.py                  # Flask API server
├── requirements.txt        # Python dependencies
├── data/                   # Training audio samples
│   ├── bearing worn out.m4a
│   ├── motor noise.m4a
│   └── ...
└── noise_classifier_model.pkl  # Trained model (after training)
```

## 🎨 Web Interface Features

- **Model ID Input**: Enter washing machine model
- **Audio Recording**: Record noise directly from browser
- **File Upload**: Upload existing audio files
- **Real-time Diagnosis**: Get instant AI predictions
- **Confidence Scores**: See confidence for all classes
- **Recommendations**: Get actionable repair suggestions

## 🔬 Technical Details

### Feature Extraction
- **Mel Spectrogram**: 128 mel bands
- **MFCCs**: 40 coefficients
- **Spectral Features**: Centroid, rolloff, zero-crossing rate

### Data Augmentation
- Pitch shifting (±2 semitones)
- Time stretching (0.9x, 1.1x)
- White noise injection
- Percussive component extraction

### Classification
- Cosine similarity between input and reference embeddings
- Maximum similarity across augmented references
- Confidence = similarity score × 100

## 📊 Model Performance

With data augmentation, the system generates 10+ reference embeddings per class, enabling robust classification even with single training samples per category.

## 🌐 Production Deployment

### Using Docker (Recommended)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5001
CMD ["python", "api.py"]
```

### Environment Variables
- `PORT`: API server port (default: 5001)

## 🛠️ Troubleshooting

**API not responding:**
- Ensure server is running: `python api.py`
- Check port 5001 is not in use

**Low confidence scores:**
- Add more training samples
- Increase augmentation diversity
- Adjust feature extraction parameters

**Browser can't record audio:**
- Use HTTPS or localhost
- Grant microphone permissions

## 📝 License

© 2026 Beko RGS. All rights reserved.

## 🤝 Support

For technical support, contact the Beko RGS development team.
