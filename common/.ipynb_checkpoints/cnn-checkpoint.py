"""CNN model builders for conifer wood classification.

Architectures ported from the original training scripts:
- build_cnn_1_block: 4 conv blocks (8/16/32/64 filters), test_script_16 style
- build_cnn_3_block: 3 double-conv blocks (16/32/64 filters), test_script_12 style
- build_resnet18:    reduced ResNet-18 (16/32/64/128 filters), test_script_15 style
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Model

kaiming_normal = keras.initializers.VarianceScaling(
    scale=2.0, mode="fan_out", distribution="untruncated_normal"
)


def build_cnn_1_block(input_shape=(90, 720, 1), num_classes=18):
    """Small CNN: single conv per block, 4 blocks (8/16/32/64 filters)."""
    image_input = Input(shape=input_shape)

    conv1 = Conv2D(8, (3, 3), padding="same", activation="relu")(image_input)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = Dropout(0.3)(conv1)

    conv2 = Conv2D(16, (3, 3), padding="same", activation="relu")(conv1)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = Dropout(0.3)(conv2)

    conv3 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv2)
    conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3 = Dropout(0.3)(conv3)

    conv4 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv3)
    conv4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4 = Dropout(0.3)(conv4)

    output = GlobalAveragePooling2D()(conv4)
    dense1 = Dense(64, activation="relu")(output)
    dense2 = Dense(64, activation="relu")(dense1)
    dense3 = Dense(64, activation="relu")(dense2)
    dense3 = Dropout(0.3)(dense3)
    predictions = Dense(num_classes)(dense3)

    return Model(inputs=image_input, outputs=predictions, name="cnn_1_block")


def build_cnn_3_block(input_shape=(90, 720, 1), num_classes=18):
    """CNN with 3 double-conv blocks (16/32/64 filters) and 4x4 max pooling."""
    image_input = Input(shape=input_shape)

    conv1 = Conv2D(16, (3, 3), padding="same", activation="relu")(image_input)
    conv2 = Conv2D(16, (3, 3), padding="same", activation="relu")(conv1)
    conv2 = MaxPooling2D(pool_size=(4, 4))(conv2)
    conv2 = Dropout(0.3)(conv2)

    conv3 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv2)
    conv4 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv3)
    conv4 = MaxPooling2D(pool_size=(4, 4))(conv4)
    conv4 = Dropout(0.3)(conv4)

    conv5 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv4)
    conv6 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv5)
    conv6 = MaxPooling2D(pool_size=(4, 4))(conv6)
    conv6 = Dropout(0.3)(conv6)

    output = GlobalAveragePooling2D()(conv6)
    dense1 = Dense(64, activation="relu")(output)
    dense2 = Dense(64, activation="relu")(dense1)
    dense3 = Dense(64, activation="relu")(dense2)
    dense3 = Dropout(0.3)(dense3)
    predictions = Dense(num_classes)(dense3)

    return Model(inputs=image_input, outputs=predictions, name="cnn_3_block")


def _conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f"{name}_pad")(x)
    return layers.Conv2D(
        filters=out_planes,
        kernel_size=3,
        strides=stride,
        use_bias=False,
        kernel_initializer=kaiming_normal,
        name=name,
    )(x)


def _basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = _conv3x3(x, planes, stride=stride, name=f"{name}.conv1")
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}.bn1")(out)
    out = layers.ReLU(name=f"{name}.relu1")(out)

    out = _conv3x3(out, planes, name=f"{name}.conv2")
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}.bn2")(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f"{name}.add")([identity, out])
    out = layers.ReLU(name=f"{name}.relu2")(out)

    return out


def _make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            layers.Conv2D(
                filters=planes,
                kernel_size=1,
                strides=stride,
                use_bias=False,
                kernel_initializer=kaiming_normal,
                name=f"{name}.0.downsample.0",
            ),
            layers.BatchNormalization(
                momentum=0.9, epsilon=1e-5, name=f"{name}.0.downsample.1"
            ),
        ]

    x = _basic_block(x, planes, stride, downsample, name=f"{name}.0")
    for i in range(1, blocks):
        x = _basic_block(x, planes, name=f"{name}.{i}")

    return x


def build_resnet18(
    input_shape=(90, 720, 1), blocks_per_layer=(2, 2, 2, 2), num_classes=18
):
    """Reduced ResNet-18 (filters 16/32/64/128) with a small dense head."""
    image_input = Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=3, name="conv1_pad")(image_input)
    x = layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        use_bias=False,
        kernel_initializer=kaiming_normal,
        name="conv1",
    )(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.ZeroPadding2D(padding=1, name="maxpool_pad")(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, name="maxpool")(x)

    x = _make_layer(x, 16, blocks_per_layer[0], name="layer1")
    x = _make_layer(x, 32, blocks_per_layer[1], stride=2, name="layer2")
    x = _make_layer(x, 64, blocks_per_layer[2], stride=2, name="layer3")
    x = _make_layer(x, 128, blocks_per_layer[3], stride=2, name="layer4")

    x = layers.GlobalAveragePooling2D(name="avgpool")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = layers.Dense(units=num_classes)(x)

    return tf.keras.Model(inputs=image_input, outputs=x, name="resnet18")
