# AuroraSAR-Change

**Explainable Change Detection for Synthetic Aperture Radar (SAR) Imagery**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

A production-ready pipeline for detecting and visualizing changes between SAR images, built with modern AI engineering practices.

## 🎯 Overview

AuroraSAR-Change provides pixel-level change detection between "before" and "after" SAR images, producing interpretable heatmaps and metrics for analysis. The system is designed for real-world deployment with FastAPI endpoints, Docker support, and ONNX export capabilities.

**Key Features:**
- 🔍 Pixel-level change probability heatmaps
- 🚀 FastAPI service with REST endpoints
- 🎨 Interactive Gradio UI for demonstrations
- 📦 Docker containerization
- ⚡ ONNX export for optimized inference
- 📊 Built-in evaluation metrics

## 🛰️ Why SAR Change Detection?

Synthetic Aperture Radar (SAR) provides unique advantages for Earth observation:

- **All-weather capability**: Works through clouds, smoke, and darkness
- **Consistent imaging**: Independent of sunlight conditions
- **Surface sensitivity**: Detects physical changes in terrain and infrastructure

### Applications

| Domain | Use Cases |
|--------|-----------|
| **Disaster Response** | Flood monitoring, landslide detection, damage assessment |
| **Infrastructure** | Construction monitoring, urban growth tracking |
| **Maritime** | Port activity, ship detection, coastal changes |
| **Environmental** | Deforestation, illegal mining detection |
| **Defense** | Strategic site monitoring, activity detection |

## 🏗️ Architecture

```
aurora-sar-change/
├── aurora/                 # Core model implementation
│   ├── model.py           # Siamese U-Net architecture
│   └── export_onnx.py     # ONNX conversion utilities
├── app/                   # Application layer
│   ├── server.py          # FastAPI service
│   └── space/            
│       └── app.py         # Gradio UI
├── training/              # Training and evaluation
│   └── eval_ir.py         # Evaluation metrics
├── data/                  # Sample data
│   └── pairs/             # Demo image pairs
└── Dockerfile             # Container configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)
- Docker (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/rikulauttia/aurora-sar-change.git
cd aurora-sar-change

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Gradio UI (Interactive Demo)

```bash
PYTHONPATH=$(pwd) python app/space/app.py
```
Open http://127.0.0.1:7860 in your browser

#### Option 2: FastAPI Service (Production API)

```bash
uvicorn app.server:app --reload --port 8000
```

Test the endpoint:
```bash
# Health check
curl -X POST http://127.0.0.1:8000/health

# Inference
curl -X POST http://127.0.0.1:8000/infer \
  -F "before=@data/pairs/demo_002_before.png" \
  -F "after=@data/pairs/demo_002_after.png" \
  -o overlay.png
```

#### Option 3: Docker Container

```bash
# Build the image
docker build -t aurora-sar-change .

# Run the container
docker run -p 8000:8000 aurora-sar-change
```

## 🧠 Technical Details

### Model Architecture

**Siamese U-Net** with shared encoders for feature extraction:
- Dual-stream processing for before/after images
- Feature fusion in decoder stage
- Pixel-wise change probability output
- Lightweight design optimized for CPU inference

### Processing Pipeline

1. **Input Processing**: Co-registered SAR image pairs
2. **Feature Extraction**: Siamese encoder with shared weights
3. **Change Detection**: U-Net decoder with skip connections
4. **Post-processing**: Probability thresholding and overlay generation
5. **Visualization**: Heatmap overlay on reference image

### Fusion Strategy

The system optionally combines neural network predictions with classical change detection (absolute difference) for improved robustness in low-data scenarios.

## 📊 Evaluation

Run the evaluation script to assess model performance:

```bash
python -m training.eval_ir
```

**Metrics computed:**
- Mean change score
- Pixel-wise precision/recall
- Threshold analysis
- Area under ROC curve (AUROC)

## 🔧 Production Deployment

### ONNX Export

Convert the PyTorch model for optimized inference:

```bash
python -m aurora.export_onnx
```

Use with ONNX Runtime:
```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("aurora_siamese.onnx")
# Prepare (N,1,H,W) float32 arrays for inference
```

### Scaling Considerations

For production deployment, consider:

| Component | Recommendation |
|-----------|---------------|
| **Tiling** | Process large scenes in 256×256 tiles with overlap |
| **Batch Processing** | Use async task queues (Celery/Ray) |
| **GPU Acceleration** | Deploy on CUDA-enabled instances |
| **Load Balancing** | Horizontal scaling with Kubernetes |
| **Monitoring** | Prometheus metrics + Grafana dashboards |

## 🔬 Data Requirements

### Demo Data

The repository includes sample 64×64 image pairs for testing:
- `demo_001_*`: No-change scenario
- `demo_002_*`: New bright feature appears

### Real SAR Data

For production use, ensure:
- **Co-registration**: Sub-pixel geometric alignment
- **Radiometric calibration**: Normalized backscatter values
- **Speckle filtering**: Lee/Frost filters or multi-look processing
- **Consistent acquisition**: Similar incidence angles and orbits

### Recommended Datasets

- [Sentinel-1 (ESA)](https://scihub.copernicus.eu/)
- [SpaceNet-6 SAR](https://spacenet.ai/)
- [ICEYE Archive](https://www.iceye.com/)

## 🚧 Roadmap

### Current Limitations
- Demo uses simplified toy data
- Basic speckle handling
- Limited to single-date pairs

### Planned Enhancements
- [ ] Multi-temporal change detection
- [ ] Class-specific change categories
- [ ] Uncertainty quantification
- [ ] Active learning interface
- [ ] Cloud-optimized GeoTIFF support
- [ ] Integration with SAR processing chains (SNAP, GAMMA)

## 🛠️ Troubleshooting

<details>
<summary><b>ModuleNotFoundError: No module named 'aurora'</b></summary>

Set the Python path:
```bash
export PYTHONPATH=$(pwd)
# Or run with: PYTHONPATH=$(pwd) python script.py
```
</details>

<details>
<summary><b>Docker OpenCV libGL error</b></summary>

The Dockerfile includes necessary graphics libraries. Ensure you're using the provided Dockerfile.
</details>

<details>
<summary><b>Port already in use</b></summary>

Use an alternative port:
```bash
uvicorn app.server:app --port 8010
```
</details>

## ⚖️ Ethics & Compliance

### Responsible Use

This technology has dual-use implications. Users must:
- Comply with all applicable laws and regulations
- Respect privacy and data protection requirements
- Consider ethical implications of surveillance applications
- Maintain human oversight in decision-making processes

### Best Practices

- **Transparency**: Clearly communicate confidence levels and limitations
- **Human-in-the-loop**: Maintain analyst review for critical decisions
- **Bias mitigation**: Test across diverse geographic and environmental conditions
- **Data governance**: Implement proper access controls and audit trails

## 📚 Technical Background

### Understanding SAR

**Synthetic Aperture Radar (SAR)** is an active imaging system that:
- Transmits microwave pulses toward Earth
- Measures the amplitude and phase of returned signals
- Produces grayscale intensity images representing surface backscatter

**Key characteristics:**
- **Bright pixels**: Strong backscatter (metal, rough surfaces)
- **Dark pixels**: Weak backscatter (calm water, smooth surfaces)
- **Speckle noise**: Multiplicative noise inherent to coherent imaging

### Change Detection Challenges

1. **Geometric**: Co-registration errors, terrain effects
2. **Radiometric**: Calibration differences, incidence angle variations
3. **Environmental**: Moisture changes, seasonal vegetation
4. **Temporal**: Different acquisition times, orbital patterns

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ICEYE for pioneering commercial SAR capabilities
- The SAR community for open datasets and tools
- Contributors to PyTorch, FastAPI, and Gradio

## 📞 Contact

**Project Lead**: Riku Lauttia  
**Repository**: [github.com/rikulauttia/aurora-sar-change](https://github.com/rikulauttia/aurora-sar-change)

---

<p align="center">
Built with ❤️ for the Earth observation community
</p>