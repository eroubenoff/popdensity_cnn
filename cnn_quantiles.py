"""
Performs transfer learning on the satellite image data.

Assumes all data are stored in 'data/data' and can be accessed using
    load_data.


This script differs from cnn.py by splitting the data into quantiles,
and predicting as logits instead of linearly. This is in response
to relatively poor performance from the linear activation layer.

"""

import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.data import Dataset
from load_data import load_data
from skimage import exposure
from sklearn.preprocessing import OneHotEncoder


def create_cnn(inputShape=(224, 224, 3), net="EfficientNet", n_classes=10):
    """ Creates a CNN

    Assumes input is in channel-last ordering, like an image, not a raster.

    Parameters
    ----------
    inputShape : tuple
        width, heigh, depth : Dimensions of the input
    net : string
        Which net to use, either "EfficientNet" or "Resnet"

    Returns
    -------
    tf.model


    """

    # chanDim = -1

    # Either use EfficientNet or ResNet
    if net == "EfficientNet":
        base_model = EfficientNetB7(weights='imagenet',
                                    include_top=False,
                                    input_shape=inputShape)
    else:
        raise ValueError("Need either EfficientNet or ResNet")
    """
    elif net == "ResNet":
        base_model = ResNet101(weights='imagenet',
                               include_top=False,
                               input_shape=inputShape)
    """

    # Set the base model to be untrainable
    base_model.trainable = False

    # Set input layer
    inputs = Input(shape=inputShape)
    # inputs = layers.Rescaling(1./225)(inputs)

    # Set base model on inputs
    x = base_model(inputs, training=False)

    # Add a pooling layer
    x = GlobalAveragePooling2D()(x)
    # x=AveragePooling2D()(x)
    # x=Flatten()(x)

    # Add dense layers
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # x = Dense(32, activation='relu')(x)
    # x = Dropout(0.2)(x)

    """
    # Add final linear activation layer
    preds = Dense(1, activation="linear")(x)
    """

    # Add final activation layer
    preds = Dense(11, activation='softmax')(x)

    # construct the CNN
    model = Model(inputs, preds)

    # return the CNN
    return model


if __name__ == "__main__":
    # Data come in as mmapped arrays
    X_train, y_train, X_test, y_test = load_data(shape=(600, 600), cache=True,
                                                 overwrite_cache=False)

    # For testing: only use 1000 train and 100 test images
    """
    X_train = X_train[0:999]
    y_train = y_train[0:999]
    X_test = X_test[0:99]
    y_test = y_test[0:99]
    """

    # Rescale the exposure of the images
    X_train = [exposure.rescale_intensity(
        x, in_range=(0, np.percentile(x, 98))) for x in X_train]
    X_test = [exposure.rescale_intensity(
        x, in_range=(0, np.percentile(x, 98))) for x in X_test]

    # Bin the data into quantiles (start with 10 and put 0 in its own category,
    # for 11 total)
    # Changing to be quantiles, for 6 total
    n_quantiles = 5
    bins = np.concatenate((y_test, y_train))
    bins = bins[bins != 0.0]
    bins = np.quantile(a=bins,
                       q=np.arange(0, 1, 1/n_quantiles))
    print("Bins:", bins)
    y_test = np.digitize(y_test, bins=bins)
    y_train = np.digitize(y_train, bins=bins)
    print(y_train[1:10])

    # One-hot encoding
    enc = OneHotEncoder()
    enc.fit(np.concatenate([y_test, y_train]).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # Build datasets
    AUTOTUNE = tf.data.AUTOTUNE
    train = Dataset.from_generator(
        lambda: ((x, y) for x, y in zip(X_train, y_train)),
        # lambda: zip(X_train, y_train),
        output_signature=(
            tf.TensorSpec(shape=(600, 600, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(n_quantiles+1,), dtype=tf.float32)
        )
    ).batch(10, drop_remainder=True).prefetch(
                buffer_size=AUTOTUNE)

    test = Dataset.from_generator(
        lambda: ((x, y) for x, y in zip(X_test, y_test)),
        # lambda: zip(X_train, y_train),
        output_signature=(
            tf.TensorSpec(shape=(600, 600, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(n_quantiles+1,), dtype=tf.float32)
        )
    ).batch(10, drop_remainder=True)  # This was the trick!

    model = create_cnn(inputShape=(600, 600, 3), n_classes=n_quantiles+1)
    opt = Adam(learning_rate=1e-3)
    # opt = Adam(learning_rate=0.01)
    model.compile(loss="mse", optimizer=opt, metrics=['mae', 'acc'])

    history = model.fit(train,
                        # validation_data=test,
                        epochs=10
                        # batch_size=1028
                        )

    # model.save("efficientnet.h5")

    model.summary()

    # Test predictions
    preds = model.predict(test)
    preds = tf.keras.applications.efficientnet.decode_predictions(preds, 3)

    """
    diff = preds.flatten() - y_test
    percentDiff = (diff / y_test) * 100
    percentDiff = percentDiff[percentDiff != np.inf]
    absPercentDiff = np.abs(percentDiff)
    mean = np.mean(absPercentDiff)
    std = np.std(absPercentDiff)

    print(mean, std)
    plt.scatter(y_test, preds.flatten())
    plt.savefig("efficientnet_preds.png")
    """
