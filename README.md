# Japanese Conifer Wood Classification by CNN and ViT models

This repository details the deep learning classification of conifer wood species using cross-sectional micrographs. We compare the performance of Convolutional Neural Networks (CNNs)—specifically Conv_4, Conv_6, and ResNet—with Vision Transformers (ViTs) at various patch sizes (30, 15, and 10). This work forms the basis for a manuscript currently under review at the Journal of Wood Science, titled: 'Comparison of convolutional neural network and vision transformer for coniferous Japanese wood classification: predictive performance and interpretation of classification strategy.'

## Repository structure

```
├── common/            # shared modules
│   ├── cnn.py         # CNN models
│   ├── vit.py         # ViT builder (patch_size 10 / 20 / 30)
│   ├── layers.py      # ViT blocks (ClassToken, MSA, TransformerBlock)
│   ├── pretrained.py  # optional pretrained ViT weight loading
│   ├── utils.py       # data loading, random-specimen split
│   │                  # confusion-matrix plotting, AdamW helper
│   └── visualize.py   # ViT attention-rollout maps
├── notebooks/
│   ├── train_resnet18.ipynb    # ResNet-18 training
│   └── train_vit_patch10.ipynb # ViT patch-10 training
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Models

| Model  | Details                                                                                                    |
| ------ | ---------------------------------------------------------------------------------------------------------- |
| Conv_4 | 4× [Conv2D 3×3 (8/16/32/64) + MaxPool 2×2 + Dropout 0.3] → GAP → 3× Dense(64, relu) → Dense(classes)       |
| Conv_6 | 3× [2× Conv2D 3×3 (16/32/64) + MaxPool 4×4 + Dropout 0.3] → GAP → 3× Dense(64, relu) → Dense(classes)      |
| ResNet | 7×7 conv s2 + 4 stages [2,2,2,2] basic blocks (16/32/64/128 filters) → GAP → 3× Dense(64) → Dense(classes) |
| ViT    | 12 transformer layers, hidden 144, 12 heads, MLP dim 1024, patch_size ∈ {10, 15, 30}                       |

All models are compiled with AdamW (lr = 1e-3, weight decay = 1e-4), batch size
32, 100 epochs, sparse categorical cross-entropy, and class weighting for
class imbalance. The best checkpoint by validation accuracy is restored before
test evaluation.

## Usage

```bash
jupyter lab notebooks/train_resnet18.ipynb     # ResNet training
jupyter lab notebooks/train_vit_patch10.ipynb  # ViT (patch size 10) training
```

Each notebook:

1. Loads the HDF5 database and crops 90×720 patches (z-score standardized).
2. For each of the 5 seeds: builds the random-specimen train/test split,
   trains the model, restores the best checkpoint, and evaluates.
3. Saves a normalized confusion matrix and a classification report per run
   under `models_<model>/`.
4. Prints a mean ± std summary of accuracy and weighted F1.

To train the other models, swap the builder call, e.g.:

```python
from common.cnn import build_cnn_1_block, build_cnn_3_block
model = build_cnn_1_block(input_shape=X_train.shape[1:], num_classes=num_classes)
```

and for ViT with a different patch size:

```python
from common.vit import build_model as build_vit
model = build_vit(image_size=X_train.shape[:2], patch_size=20, ...)
```

## ## License

MIT — see [LICENSE](LICENSE).
