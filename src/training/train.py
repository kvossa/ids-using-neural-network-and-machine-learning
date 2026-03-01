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

# import yaml
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import AUC, Precision, Recall, F1Score, FalsePositives, FalseNegatives, TopKCategoricalAccuracy
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from src.model.model import IDSModelFactory
from src.utils.logger import IDSLogger
from src.utils.train_stopper import F1EarlyStopping
from src.utils.visualize import IDSVisualizer
from src.preprocessing.pipeline import IDSPipeline
from src.preprocessing.clean import DataCleaner
from src.preprocessing.encoding import CategoricalEncoder
from src.preprocessing.features_extraction import FeatureExtraction
from src.preprocessing.features_selection import FeatureSelector
from src.preprocessing.scaling import StandardScaler


from pathlib import Path

# CIC
train_df = pd.read_parquet('data/processed/CIC-IDS2017/splits/train/data.parquet')
test_df = pd.read_parquet('data/processed/CIC-IDS2017/splits/test/data.parquet')
val_df = pd.read_parquet('data/processed/CIC-IDS2017/splits/val/data.parquet')
dataset = "CIC"
stratify_column = 'attack_type'

# UNSW
# train_df = pd.read_csv(Path('data/processed/UNSW-NB15/splits/train.csv'))
# test_df = pd.read_csv(Path('data/processed/UNSW-NB15/splits/test.csv'))
# val_df = pd.read_csv(Path('data/processed/UNSW-NB15/splits/validation.csv'))
# dataset = "UNSW"
# stratify_column = 'attack_cat'

X_train = train_df.drop(stratify_column, axis=1)
y_train = train_df[stratify_column]

X_test = test_df.drop(stratify_column, axis=1)
y_test = test_df[stratify_column]

X_val = val_df.drop(stratify_column, axis=1)
y_val = val_df[stratify_column]

preprocessor = joblib.load(f'models/preprocessing/{dataset.lower()}/preprocessing.pkl')
label_encoder = joblib.load(f'models/preprocessing/{dataset.lower()}/label_encoder.pkl')

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)
X_val_processed = preprocessor.transform(X_val)

y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_val_encoded = label_encoder.transform(y_val)

classes = np.unique(y_train_encoded)

#Solo para UNSW#
# weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_encoded) 
# class_weight_dict = dict(zip(classes, weights))
# # class_weight_dict = dict(enumerate(weights))
# sample_weights = np.array([class_weight_dict[y] for y in y_train_encoded])
################

print("##shape##")
print(X_train_processed.shape)
print(X_test_processed.shape)

print("##types##")
print(X_train_processed.isna().sum().sum())
print(X_train_processed.dtypes.unique())

print("##are the same? trainset vs testset##")
print(X_train_processed.shape[1] == X_test_processed.shape[1])

print("##y_train shape")
print(y_train.shape)

window_size = 1
num_features = X_train_processed.shape[1]
num_classes = len(label_encoder.classes_)

X_train_processed_array = X_train_processed.values
X_test_processed_array = X_test_processed.values
X_val_processed_array = X_val_processed.values

X_train_seq = X_train_processed_array.reshape(X_train_processed.shape[0], 1, num_features)
X_test_seq = X_test_processed_array.reshape(X_test_processed.shape[0], 1, num_features)
X_val_seq = X_val_processed_array.reshape(X_val_processed.shape[0], 1, num_features)

X_train_inputs = {
    "ae_input": X_train_processed_array,
    "cnn_input": X_train_seq,
    "lstm_input": X_train_seq
}

X_test_inputs = {
    "ae_input": X_test_processed_array,
    "cnn_input": X_test_seq,
    "lstm_input": X_test_seq
}

X_val_inputs = {
    "ae_input": X_val_processed_array,
    "cnn_input": X_val_seq,
    "lstm_input": X_val_seq
}

y_train_ohe = to_categorical(y_train_encoded, num_classes=num_classes)
y_test_ohe = to_categorical(y_test_encoded, num_classes=num_classes)
y_val_ohe = to_categorical(y_val_encoded, num_classes=num_classes)


model = IDSModelFactory.create_model(window_size=window_size, num_classes=num_classes, num_features=num_features)

print("compiling model...")

model.compile(
    optimizer="adam", 
    metrics={
        "classification": [
            "accuracy", Precision(name="precision"), Recall(name="recall"), F1Score(name="f1_score", average="macro"), 
            AUC(name="auc", multi_label=True), FalsePositives(name="fp"), FalseNegatives(name="fn"), TopKCategoricalAccuracy(k=3)
            ],
    },
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
        X_val_inputs,
        {
            "classification": y_val_ohe,
            "reconstruction": X_val_processed
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

print("training model...")

history = model.fit(
    X_train_inputs,
    {
        "classification": y_train_ohe,
        "reconstruction": X_train_processed
    },
    validation_data=(X_val_inputs, 
        {
            "classification": y_val_ohe,
            "reconstruction": X_val_processed
        }
    ),
    epochs=5,#70-100
    batch_size=64,#128
    callbacks=[f1_callback, checkpoint, reduce_lr],
    #Only for UNSW
    # sample_weight={
        # "classification": sample_weights,
        # "reconstruction": np.ones(len(sample_weights))
    # },
    #######
    verbose=1
)

print("testing model...")

test_model = model.evaluate(X_test_inputs, {
    "classification": y_test_ohe,
    "reconstruction": X_test_processed
})

print(f"test results: {test_model}")

y_pred_probs = model.predict(X_test_inputs)["classification"]
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test_encoded

visuals_path = Path(f"reports/figures/{dataset.lower()}")

visualizer = IDSVisualizer(output_dir=visuals_path)
class_names = sorted(train_df[stratify_column].unique())
print(f"class names: {class_names}")
visualizer.plot_confusion_matrix(y_true=y_true, y_pred=y_pred, classes=(range(num_classes)))
visualizer.plot_roc_curve(y_true=y_true, y_scores=y_pred_probs, classes=class_names)

# IDSModelFactory.save_model(model)
# print("Model has been saved")