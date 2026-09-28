"""Shared data loading, splitting, and evaluation utilities for conifer wood
classification.

Pipeline (mirrors the original training scripts):
- HDF5 database with datasets keyed as "family/species"
- images cropped into 90x720 horizontal patches, z-score standardized
- random-specimen ("rs") split: one whole specimen per species held out for test
- class weights for class imbalance
- confusion-matrix plotting and run summaries
"""

import itertools

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

SPECIES = [
    "Abies firma",
    "Abies sachalinensis",
    "Pseudotsuga japonica",
    "Tsuga sielbodii",
    "Larix kaempferi",
    "Picea jezoensis",
    "Pinus densiflora",
    "Pinus thunbergii",
    "Podocarpus macrophyllus",
    "Sciadoptys verticullata",
    "Cephalotaxus harringtonia",
    "Taxus cuspidata",
    "Torreya nucifera",
    "Chamaecyparis obtusa",
    "Chamaecyparis pisifera",
    "Cryptomeria japonica",
    "Thuja standishii",
    "Thujopsis dolabrata",
]

FAMILIES = [
    "Cephalotaxaceae",
    "Cupressaceae",
    "Pinaceae",
    "Podocarpaceae",
    "Sciadopityaceae",
    "Taxaceae",
]

SEED_LIST = [0, 5, 10, 20, 56]


class HdfReader:
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_objs = []
        self.all_groups = []
        self.all_datasets = []

    def list_contents(self, f):
        all_objs = []
        f.visit(all_objs.append)
        all_groups = [obj for obj in all_objs if isinstance(f[obj], h5py.Group)]
        all_datasets = [obj for obj in all_objs if isinstance(f[obj], h5py.Dataset)]
        return all_groups, all_datasets


def standardize(x):
    return (x - np.mean(x)) / np.std(x)


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def load_database(data_path):
    """Open the HDF5 database and return (file, dataset_keys)."""
    dataset = h5py.File(data_path, "r")
    _, keys = HdfReader().list_contents(dataset)
    return dataset, keys


def key_labels(keys, level="species"):
    """Return the label (family or species name) of each dataset key."""
    idx = 0 if level == "family" else 1
    return [k.split("/")[idx] for k in keys]


def make_label_map(level="species"):
    """Map label names to integer ids for the requested classification level."""
    classes = FAMILIES if level == "family" else SPECIES
    return {name: i for i, name in enumerate(classes)}


def get_rs_split_indices(spec_n_list, species, seed):
    """Random-specimen split: pick one specimen per species as test set.

    Returns (train_index, test_index) as arrays of dataset indices.
    """
    np.random.seed(seed)
    test_index = []
    for spec in species:
        index_all = np.where(np.asarray(spec_n_list) == spec)[0]
        index_s = np.random.choice(index_all)
        test_index.append(index_s)

    test_index = np.sort(test_index)
    train_index = np.asarray(
        [value for value in np.arange(0, len(spec_n_list)) if value not in test_index]
    )
    return train_index, test_index


def load_patches(dataset, keys, indices, level="species", crop_height=90):
    """Crop the images of the given dataset keys into 90x720 patches.

    Returns (X, labels): standardized patches (N, crop_height, 720) and the
    family- or species-level label of each patch (as strings).
    """
    idx = 0 if level == "family" else 1
    imgs_all, labels = [], []
    for i in tqdm(indices):
        imgs = dataset[keys[i]][()]
        for img in imgs:
            for k in range(np.divmod(img.shape[0], crop_height)[0]):
                crop_img = img[k * crop_height : k * crop_height + crop_height, :]
                imgs_all.append(standardize(crop_img))
                labels.append(keys[i].split("/")[idx])
    return np.asarray(imgs_all), labels


def compute_class_weight(y):
    """Balanced class weights: n_samples / (n_classes * count_per_class)."""
    y = np.asarray(y)
    classes = np.unique(y)
    n_samples = len(y)
    weight = n_samples / (len(classes) * np.sum(y == classes[:, None], axis=1))
    return {int(c): float(w) for c, w in zip(classes, weight)}


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix",
                         cmap=plt.cm.Blues):
    """Print and plot the (optionally normalized) confusion matrix."""
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def make_adamw(learning_rate=0.001, weight_decay=0.0001):
    """AdamW optimizer: uses tensorflow-addons when available, else keras."""
    from tensorflow import keras

    try:
        import tensorflow_addons as tfa

        return tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
    except ImportError:
        return keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )


def summarize_runs(reports):
    """Mean +/- std of accuracy and weighted F1 over a list of
    classification_report dicts (output_dict=True)."""
    accs = [r["accuracy"] for r in reports]
    f1s = [r["weighted avg"]["f1-score"] for r in reports]
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "accuracies": accs,
        "f1s": f1s,
    }
