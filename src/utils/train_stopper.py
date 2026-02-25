import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from keras.callbacks import Callback

class F1EarlyStopping(Callback):
    def __init__(self, validation_data, patience:int=7):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.patience = patience
        self.best_f1 = 0
        self.wait = 0
        self.best_weights = None
    
    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.X_val, verbose=0)

        y_pred = np.argmax(predictions["classification"], axis=1)
        y_true = np.argmax(self.y_val["classification"], axis=1)

        f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")

        print(f"\nEpoch {epoch+1} - Val F1-macro: {f1:.4f}")

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("Early stopping triggered. Restoring best weights.")
                self.model.set_weights(self.best_weights)   
                self.model.stop_training = True