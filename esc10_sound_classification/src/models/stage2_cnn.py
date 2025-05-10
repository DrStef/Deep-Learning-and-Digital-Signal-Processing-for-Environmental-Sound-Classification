import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from src.utils.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

def prepare_cnn_features(data_cwt_global, labels, test_size=0.2, random_state=5):
    """
    Prepare features and labels for Stage II CNN training.
    Args:
        data_cwt_global (np.array): Combined features (N, 128*128, 3).
        labels (np.array): Class labels (0-9).
        test_size (float): Test split proportion (default: 0.2).
        random_state (int): Random seed (default: 5).
    Returns:
        train_X (np.array): Training features (N_train, 128, 128, 3).
        test_X (np.array): Test features (N_test, 128, 128, 3).
        train_y (np.array): One-hot training labels (N_train, 10).
        test_y (np.array): One-hot test labels (N_test, 10).
        train_indices (np.array): Training indices.
        test_indices (np.array): Test indices.
    Note:
        Uses stratified split to ensure balanced class representation.
    """
    # Map string labels to integers
    label_map = {
        '001-Dogbark': 0,
        '002-Rain': 1,
        '003-Seawaves': 2,
        '004-Babycry': 3,
        '005-Clocktick': 4,
        '006-Personsneeze': 5,
        '007-Helicopter': 6,
        '008-Chainsaw': 7,
        '009-Rooster': 8,
        '010-Firecrackling': 9
    }
    print(f"Debug: unique labels = {set(labels)}")
    int_labels = np.array([label_map[label] for label in labels])
    print(f"Debug: int_labels shape = {int_labels.shape}, values = {int_labels[:5]}")



    features_cwt_CNN = np.reshape(data_cwt_global, (-1, 128, 128, 3))
    ylabels = keras.utils.to_categorical(int_labels, num_classes=10, dtype='float32')

    train_X, test_X, train_indices, test_indices = train_test_split(
        features_cwt_CNN, np.arange(len(features_cwt_CNN)), test_size=test_size,
        stratify=ylabels, random_state=random_state
    )
    train_y = ylabels[train_indices]
    test_y = ylabels[test_indices]
    return train_X, test_X, train_y, test_y, train_indices, test_indices

def build_cnn_model(input_shape, num_classes):
    """
    Constructs and compiles a Convolutional Neural Network model for image classification.

    This model architecture includes multiple convolutional layers followed by max pooling,
    with increasing depth of feature maps, and ends with fully connected layers and a softmax output.

    :param input_shape: A tuple representing the shape of input data excluding the batch dimension.
    :param num_classes: Number of classes for classification.
    :return: A compiled Keras Sequential model.
    """
    model = Sequential()

    # First Conv Layer
    model.add(Conv2D(32, kernel_size=(3, 3), dilation_rate=(1, 1), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Conv Layer
    model.add(Conv2D(64, kernel_size=(3, 3), dilation_rate=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third Conv Layer
    model.add(Conv2D(128, kernel_size=(3, 3), dilation_rate=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fourth Conv Layer
    model.add(Conv2D(256, kernel_size=(3, 3), dilation_rate=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output for dense layers
    model.add(Flatten())

    # Dense Layer
    model.add(Dense(800, activation='relu'))

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Compile the model
    opt = Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        amsgrad=True
    )
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

def train_cnn_model(model, train_X, train_y, test_X, test_y, epochs=100, batch_size=32, accuracy_threshold=0.9875):
    """
    Train Stage II CNN with early stopping and learning rate reduction.
    Args:
        model: Keras model.
        train_X (np.array): Training features.
        train_y (np.array): Training labels.
        test_X (np.array): Test features.
        test_y (np.array): Test labels.
        epochs (int): Max epochs (default: 100).
        batch_size (int): Batch size (default: 32).
        accuracy_threshold (float): Stop training if val_accuracy exceeds this (default: 0.9875).
    Returns:
        history: Training history.
    """
    #class CustomEarlyStopping(keras.callbacks.Callback):
    #    def on_epoch_end(self, epoch, logs={}):
    #        if logs.get('val_accuracy') > accuracy_threshold:
    #            print(f"\nReached {accuracy_threshold*100:.2f}% accuracy, stopping training!")
    #            self.model.stop_training = True

    #callbacks = [
    #    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    #    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000001),
    #    CustomEarlyStopping()
    #]
    #history = model.fit(
    #    train_X, train_y, epochs=epochs, batch_size=batch_size,
    #    validation_data=(test_X, test_y), callbacks=callbacks
    #)
    #return history


    accuracy_threshold=0.98

    class myCallback(tf.keras.callbacks.Callback): 
        def on_epoch_end(self, epoch, logs={}): 
            if(logs.get('val_accuracy') > accuracy_threshold):   
                print("\nReached %2.2f%% accuracy, we stop training!!" %(accuracy_threshold*100))   
                self.model.stop_training = True

    custom_early_stopping = myCallback()

    history = model.fit(
        train_X, 
        train_y, 
        epochs=80, 
        steps_per_epoch=len(train_X)//16,
        #validation_split=0.2, 
        validation_data=(test_X, test_y),
        batch_size=10, 
        #verbose=2,
        callbacks=[custom_early_stopping]
    )
    return history


def evaluate_cnn_model(model, test_X, test_y, class_names):
    """
    Evaluate Stage II CNN and generate metrics.
    Args:
        model: Trained Keras model.
        test_X (np.array): Test features.
        test_y (np.array): Test labels.
        class_names (list): Class names for reporting.
    Returns:
        test_loss (float): Test loss.
        test_acc (float): Test accuracy.
        report (str): Classification report.
        confusion_fig (plt.Figure): Confusion matrix plot.
    """
    test_loss, test_acc = model.evaluate(test_X, test_y)
    predictions = model.predict(test_X, batch_size=10)
    predictions = predictions + 1e-09  # Avoid numerical issues
    test_y = test_y + 1e-09
    report = classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=class_names)
    fig = plt.figure(figsize=(10, 10))
    plot_confusion_matrix(test_y.argmax(axis=1), predictions.argmax(axis=1), class_names)
    plt.title("CNN - Confusion Matrix")
    return test_loss, test_acc, report, fig