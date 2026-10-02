# How to Run This Project

## 1. Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Prepare the dataset
Extract the three zip files you received (`train-*.zip`, `valid-*.zip`, `test-*.zip`) so that the folder structure becomes:

```
data/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
├── valid/
│   └── (same 4 folders)
└── test/
    └── (same 4 folders)
```

## 3. Train the models
```bash
python train.py --data-dir data --output-dir models --epochs 15 --batch-size 16 --img-size 224
```

This creates:
- `models/custom_cnn_best.pth`
- `models/resnet_scratch_best.pth`
- metrics JSON, history plots and confusion matrices

## 4. Launch the Streamlit app
```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501), upload an MRI image and get the prediction.

## 5. Deploy to Streamlit Cloud / GitHub
1. Create a new GitHub repository
2. Push this entire folder (do **not** commit the large `data/` folder)
3. Go to https://share.streamlit.io → New app → select your repo → main file = `app.py`
4. After the first successful train you can also upload the `.pth` files so the app works without re-training on the cloud.

## Notes
- The current `models/` folder only contains `class_names.json`. You must run `train.py` first (or copy your trained `.pth` files into `models/`).
- For better accuracy use a GPU and the full dataset.
- Transfer-learning models (`mobilenet_v2`, `resnet50`, etc.) require `torchvision`.
