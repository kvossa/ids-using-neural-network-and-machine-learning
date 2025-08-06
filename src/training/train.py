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
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from ..model.model import IDSModelFactory
from ..utils.logger import IDSLogger
from pathlib import path

class IDSTrainer:

    def __init__(self, config_path='configs/models_param.yaml'):
        self.logger = IDSLogger()
        self.config = self._load_config(config_path)
        self.model = None
        self.history = None

    def _load_config(self, config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.logger.log('INFO', f'Loaded config: {config}')
        return config

    def prepare_data(self, data_path):
        data = np.load(data_path)
        X_train, X_val, y_train, y_val = train_test_split(
            data['X'], data['y'],
            test_size=self.config['val_split'],
            stratify=data['y']
        )
        self.logger.log('INFO', f'Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}')
        return X_train, X_val, y_train, y_val

    def train_model(self, X_train, y_train, X_val, y_val):
        model_type = self.config['model_type']
        input_shape = X_train.shape[1:]

        if model_type == 'mlp':
            self.model = IDSModelFactory.create_mlp(input_shape[0])
        elif model_type == 'lstm':
            self.model = IDSModelFactory.create_lstm(input_shape)
        elif model_type == 'autoencoder':
            self.model = IDSModelFactory.create_autoencoder(input_shape[0])
            y_train = X_train  

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            EarlyStopping(patience=5, monitor='val_loss'),
            ModelCheckpoint('models/best_model.h5', save_best_only=True)
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        return self.history

if __name__ == "__main__":
    trainer = IDSTrainer()
    X_train, X_val, y_train, y_val = trainer.prepare_data('data/processed/train_data.npz')
    history = trainer.train_model(X_train, y_train, X_val, y_val)
	