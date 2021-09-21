"""
Performs transfer learning on the satellite image data.

Assumes all data are stored in 'data/data' and can be accessed using
    load_data.

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet101, EfficientNetB7
from tensorflow.data import Dataset
from load_data import load_data
# from sklearn.preprocessing import MinMaxScaler


def create_cnn(inputShape=(224, 224, 3), net="EfficientNet"):
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
    elif net == "ResNet":
        base_model = ResNet101(weights='imagenet',
                               include_top=False,
                               input_shape=inputShape)
    else:
        raise ValueError("Need either EfficientNet or ResNet")

    # Set the base model to be untrainable
    base_model.trainable = False

    # Set input layer
    inputs = Input(shape=inputShape)

    # Set base model on inputs
    x = base_model(inputs, training=False)

    # Add a pooling layer
    x = GlobalAveragePooling2D()(x)
    # x=AveragePooling2D()(x)
    # x=Flatten()(x)

    # Add dense layers
    # x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # x = Dense(32, activation='relu')(x)
    # x = Dropout(0.2)(x)

    # Add final linear activation layer
    preds = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, preds)

    # return the CNN
    return model


if __name__ == "__main__":
    # Data come in as mmapped arrays
    X_train, y_train, X_test, y_test = load_data(shape=(600, 600), cache=True,
                                                 overwrite_cache=False)

    # Rescale both test and train to the training set
    y_test = (y_test - np.min(y_train)) / (np.max(y_train) - np.min(y_train))
    y_train = (y_train - np.min(y_train)) / (np.max(y_train) - np.min(y_train))

    """
    # Create generator functions
    def train_gen():
        return zip(X_train, y_train)

    def test_gen():
        return zip(X_test, y_test)

    train = Dataset.from_generator(train_gen,
                                   output_types=(np.uint8, np.float32),
                                   output_shapes=((None, 600, 600, 3), ())
                                   )

    test = Dataset.from_generator(test_gen,
                                  output_types=(np.uint8, np.float32),
                                  output_shapes=((None, 600, 600, 3), ())
                                  )
    """

    X_train = Dataset.from_generator(
        lambda: iter(X_train),
        output_signature=tf.TensorSpec(shape=(600, 600, 3), dtype=tf.uint8)
    )
    y_train = Dataset.from_generator(
        lambda: iter(y_train),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.float32)
    )
    X_test = Dataset.from_generator(
        lambda: iter(X_test),
        output_signature=tf.TensorSpec(shape=(600, 600, 3), dtype=tf.uint8)
    )
    y_test = Dataset.from_generator(
        lambda: iter(y_test),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.float32)
    )

    train = Dataset.zip((X_train, y_train)).batch(30)
    test = Dataset.zip((X_test, y_test)).batch(30)

    model = create_cnn(inputShape=(600, 600, 3))
    opt = Adam(lr=1e-6, decay=1e-3 / 200)
    model.compile(loss="mse", optimizer=opt)

    history = model.fit(train,
                        # validation_data=(X_test, y_test),
                        epochs=10  # ,
                        # batch_size=1028
                        )

    model.save("efficientnet.h5")

    model.summary()

    preds = model.predict(X_test)
    diff = preds.flatten() - y_test
    percentDiff = (diff / y_test) * 100
    percentDiff = percentDiff[percentDiff != np.inf]
    absPercentDiff = np.abs(percentDiff)
    mean = np.mean(absPercentDiff)
    std = np.std(absPercentDiff)

    print(mean, std)
    plt.scatter(y_test, preds.flatten())
    plt.savefig("models/efficientnet_preds.png")
