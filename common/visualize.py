import cv2
import numpy as np
import tensorflow as tf
from . import layers, vit


def attention_map(model, image):
    """Get an attention map for an image and model using the technique
    described in Appendix D.7 in the paper (unofficial).

    Supports non-square patch grids (e.g. 90x720 inputs with patch 10 give a
    9x72 grid), unlike the original vit-keras implementation which assumes a
    square grid.

    Args:
        model: A ViT model built with common.vit.build_model
        image: An image for which we will compute the attention map.
    """
    img_height, img_width = model.input_shape[1], model.input_shape[2]
    embedding = model.get_layer("embedding")
    patch_size = embedding.strides[0]
    grid_h = img_height // patch_size
    grid_w = img_width // patch_size
    num_patches = grid_h * grid_w

    # Prepare the input
    X = vit.preprocess_inputs(cv2.resize(image, (img_height, img_width)))[np.newaxis, :]  # type: ignore

    # Get the attention weights from each transformer.
    outputs = [
        l.output[1] for l in model.layers if isinstance(l, layers.TransformerBlock)
    ]
    weights = np.array(
        tf.keras.models.Model(inputs=model.inputs, outputs=outputs).predict(X)
    )
    num_layers = weights.shape[0]
    num_heads = weights.shape[2]
    reshaped = weights.reshape(
        (num_layers, num_heads, num_patches + 1, num_patches + 1)
    )

    # From Appendix D.6 in the paper ...
    # Average the attention weights across all heads.
    reshaped = reshaped.mean(axis=1)

    # From Section 3 in https://arxiv.org/pdf/2005.00928.pdf ...
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    reshaped = reshaped + np.eye(reshaped.shape[1])
    reshaped = reshaped / reshaped.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Recursively multiply the weight matrices
    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])

    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_h, grid_w)
    mask = cv2.resize(mask / mask.max(), (image.shape[1], image.shape[0]))[
        ..., np.newaxis
    ]
    return (mask * image).astype("uint8")
