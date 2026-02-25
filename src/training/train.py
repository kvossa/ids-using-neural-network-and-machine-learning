# from utils.logger import IDSLogger
# from utils.visualize import IDSVisualizer

# logger = IDSLogger()
# viz = IDSVisualizer()

# logger.log('INFO', 'Starting model training', model_type='LSTM')
# # ... training code ...

# # After evaluation:
# viz.plot_confusion_matrix(y_true, y_pred)
# viz.plot_roc_curve(y_true, y_scores)
# logger.log('INFO', 'Training complete', metrics={'accuracy': 0.95})

import yaml
import joblib
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import AUC, Precision, Recall, F1Score, FalsePositives
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from ..model.model import IDSModelFactory
from ..utils.logger import IDSLogger
from ..utils.train_stopper import F1EarlyStopping
from pathlib import Path

X_train = {
    "ae_input": np.load("X_train_ae.npy"),
    "cnn_input": np.load("X_train_cnn.npy"),
    "lstm_input": np.load("X_train_lstm.npy"),
}

X_val = {
    "ae_input": np.load("X_val_ae.npy"),
    "cnn_input": np.load("X_val_cnn.npy"),
    "lstm_input": np.load("X_val_lstm.npy"),
}

y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")

y_labels = np.argmax(y_train, axis=1)
classes = np.unique(y_labels)

weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_labels)

class_weight_dict = dict(zip(classes, weights))

model = IDSModelFactory.create_model()

model.compile(
    optimizer="adam", 
    metrics=["accuracy", Precision(), Recall(), F1Score(), AUC(), FalsePositives()],
    loss={
        "classification": "categorical_crossentropy",
        "reconstruction": "mse"
    },
    loss_weights={
        "classification": 1.0,
        "reconstruction": 0.3,
    }
)

f1_callback = F1EarlyStopping(
    validation_data=(
        X_val,
        {
            "classification": y_val,
            "reconstruction": X_val["ae_input"]
        }
    ),
    patience=7
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_ids_model.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)


history = model.fit(
            X_train,
            {
                "classification": y_train,
                "reconstruction": X_train["ae_input"]
            },
            validation_data=(X_val, 
                {
                    "classification": y_val,
                    "reconstruction": X_val["ae_input"]
                }
            ),
            epochs=100,
            batch_size=128,
            callbacks=[f1_callback, checkpoint, reduce_lr],
            class_weight={
                "classification": class_weight_dict
            },
            verbose=1
        )