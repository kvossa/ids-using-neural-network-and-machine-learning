import tensorflow as tf
from keras.callbacks import Callback

class EpochSummary(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"  Época {epoch+1:2d} | "
            f"loss: {logs.get('classification_loss', 0):.4f} | "
            f"f1: {logs.get('classification_f1_score', 0):.4f} | "
            f"recall: {logs.get('classification_recall', 0):.4f} | "
            f"precision: {logs.get('classification_precision', 0):.4f}"
        )