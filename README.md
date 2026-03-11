# Mini Project 8: Flood Area Image Segmentation

**BCIT — Network Security | Mini Project 8**

| Name | Role |
|---|---|
| Yansong Jia | Result analysis, report writing |
| Sepehr Mansouri | Code implementation, model training |

---

## Problem Description

This project applies semantic segmentation to aerial flood imagery. The goal is to classify every pixel as **flooded** or **not flooded** — producing a pixel-level flood map that emergency responders can use to prioritize rescue operations.

**Dataset:** [Flood Area Segmentation](https://kaggle.com/datasets/faizalkarim/flood-area-segmentation) — 290 aerial/UAV images with binary masks.

---

## Results Summary

| Model | mIoU | mDice | Flood IoU |
|---|---|---|---|
| U-Net BCE+Dice @ 640×640 | 0.7348 | 0.8318 | 0.6859 |
| U-Net Dice-only @ 640×640 | 0.7103 | 0.8114 | 0.6468 |
| **MobileNetV2 U-Net @ 640×640** | **0.7958** | **0.8780** | **0.7599** |
| U-Net BCE+Dice @ 256×256 | 0.3607 | 0.5126 | 0.3297 |

**Best model:** MobileNetV2-Encoder U-Net @ 640×640.

Key findings:
- Resolution matters enormously — 256×256 is essentially broken (flood IoU 0.33 vs 0.76 at 640×640)
- Boundary errors are 3.2× higher than interior errors (36.14% vs 11.46%)
- BCE+Dice combined loss outperforms pure Dice loss (mIoU 0.7348 vs 0.7103)

---

## Setup

### 1. Clone and create environment

```bash
git clone <your-repo-url>
cd mini-project-8
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install dependencies

```bash
pip install tensorflow kagglehub numpy matplotlib scipy scikit-learn
```

Or from the requirements file:

```bash
pip install -r requirements.txt
```

### 3. Download dataset

The notebook downloads the dataset automatically via `kagglehub`. You need a Kaggle account with an API key configured:

```bash
# Put your kaggle.json in ~/.kaggle/
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Then the notebook cell runs:

```python
import kagglehub
dataset_path = kagglehub.dataset_download("faizalkarim/flood-area-segmentation")
```

Dataset will be cached at `~/.cache/kagglehub/datasets/faizalkarim/flood-area-segmentation/`.

---

## How to Run

### Training

Open and run `notebook.ipynb` from top to bottom. Sections:

1. **Setup & Imports** — installs packages, checks GPU
2. **Dataset Loading** — downloads via kagglehub, locates Image/ and Mask/ directories
3. **Train/Val/Test Split** — 70/15/15 split (203/43/44 images)
4. **Preprocessing** — resize to 640×640, normalize to [0, 1]
5. **EDA** — visualizes samples and class distribution
6. **Data Augmentation** — flips, rotation, brightness/contrast jitter
7. **Model Architecture** — builds custom U-Net and MobileNetV2-Encoder U-Net
8. **Loss Functions** — defines Dice loss, BCE+Dice combined loss
9. **Training** — trains all three models with callbacks
10. **Resolution Comparison** — trains 256×256 model for comparison
11. **Evaluation** — computes IoU, Dice, confusion matrices on test set
12. **Visualization** — shows predictions with error maps
13. **Error Analysis** — boundary vs. interior error rates

### Evaluation only (load saved checkpoints)

If you have the trained `.keras` files:

```python
import tensorflow as tf
model = tf.keras.models.load_model(
    'best_flood_unet_bce_dice.keras',
    custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_coeff_metric': dice_coeff_metric}
)
# Then run the evaluation cells in Section 11
```

---

## File Structure

```
mini-project-8/
├── notebook.ipynb                   # Full pipeline
├── requirements.txt
├── README.md
├── .gitignore
└── saved_models/                    # Excluded from git (see .gitignore)
    ├── best_flood_unet_bce_dice.keras
    ├── best_flood_unet_dice.keras
    └── best_mobilenet_unet.keras
```

---

## Dependencies

```
tensorflow>=2.15.0
kagglehub
numpy
matplotlib
scipy
scikit-learn
```

---

## .gitignore

The following are excluded from the repository:

```
*.keras
*.h5
*.pkl
__pycache__/
.ipynb_checkpoints/
*.cache/
data/
dataset/
```

---

## Sample Predictions

Good predictions (mIoU ~0.92–0.94): Clear flood regions with strong color contrast. Errors limited to thin boundary zones.

Poor predictions (mIoU ~0.40–0.43): Complex scenes with narrow flood corridors, partially submerged vegetation, or muddy water that blends visually with dry terrain.

See `notebook.ipynb` Section 12 for full visualizations with ground truth and error maps.
