# 🧠 Brain Tumor MRI Image Classification

Deep Learning project for multi-class classification of brain MRI images into **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.

This project implements:
- Custom CNN from scratch
- Residual Network (ResNet-inspired) from scratch
- Transfer Learning ready architecture (with torchvision)
- Full training pipeline with data augmentation, evaluation, and comparison
- Interactive **Streamlit** web app for real-time predictions

## 📌 Problem Statement

Develop a deep learning solution to classify brain MRI scans by tumor type. The system assists radiologists by providing fast, accurate preliminary classification, reducing diagnostic time and supporting early intervention.

## 🗂️ Dataset

- **Source**: Labeled MRI Brain Tumor Dataset (Roboflow / provided zips)
- **Classes**: `glioma`, `meningioma`, `no_tumor`, `pituitary`
- **Split**: Train (~1695), Validation (~502), Test (~246)

Place the extracted folders as:
```
data/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
├── valid/
│   └── ...
└── test/
    └── ...
```

Extract the provided `train-*.zip`, `valid-*.zip` and `test-*.zip` into the `data/` directory.

## 🚀 Quick Start

### 1. Setup
```bash
git clone <your-repo-url>
cd Brain_Tumor_MRI_Project
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
mkdir -p data
# Extract the three zip files so that data/train, data/valid, data/test exist
# with the four class sub-folders inside each.
```

### 3. Train Models
```bash
python train.py --epochs 15 --batch-size 16 --img-size 224
```
This trains:
- CustomCNN
- ResNetFromScratch
- (Optionally) Transfer learning models if torchvision is available

Best models are saved to `models/`.

### 4. Run Streamlit App
```bash
streamlit run app.py
```
Upload an MRI image and get instant prediction with confidence scores.

## 🏗️ Project Structure
```
├── app.py                  # Streamlit web application
├── train.py                # Training script (all models)
├── requirements.txt
├── README.md
├── .gitignore
├── models/                 # Saved .pth weights + class mapping
├── data/                   # train / valid / test folders (not included in zip)
├── src/
│   ├── dataset.py          # Dataset & transforms
│   ├── models.py           # Model definitions
│   └── utils.py            # Metrics, plots, helpers
└── notebooks/
    └── (optional exploratory notebook)
```

## 📊 Models Implemented

| Model                  | Type                  | Description                                      |
|------------------------|-----------------------|--------------------------------------------------|
| CustomCNN              | From Scratch          | 5-block CNN + BatchNorm + Dropout                |
| ResNetFromScratch      | From Scratch          | Residual blocks inspired by ResNet18             |
| MobileNetV2 / ResNet50 | Transfer Learning     | Pretrained on ImageNet (requires torchvision)    |

## 📈 Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (macro)
- Confusion Matrix
- Training / Validation loss & accuracy curves

## 🌐 Streamlit Deployment
The app (`app.py`) is ready for Streamlit Cloud or local hosting.
1. Push this repo to GitHub
2. Connect at https://share.streamlit.io
3. Set main file to `app.py`

## 📝 Notes
- On GPU the full dataset trains comfortably in < 30 min.
- On CPU use smaller `--epochs` or a subset.
- For production, fine-tune the best transfer-learning model.

## License
CC BY 4.0 (dataset) + MIT (code)
