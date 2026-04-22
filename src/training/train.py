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
from pathlib import Path
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import AUC, Precision, Recall, F1Score, FalsePositives, FalseNegatives, TopKCategoricalAccuracy
from keras.utils import to_categorical
from keras.losses import CategoricalFocalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler, OneSidedSelection
from imblearn.combine import SMOTEENN, SMOTETomek

from src.model.model import IDSModelFactory
from src.utils.logger import IDSLogger
from src.utils.train_stopper import F1EarlyStopping
from src.utils.visualize import IDSVisualizer
from src.preprocessing.pipeline import IDSPipeline
from src.preprocessing.clean import DataCleaner
from src.preprocessing.features_extraction import FeatureExtraction
from src.preprocessing.features_selection import FeatureSelector
from src.preprocessing.scaling import StandardScaler
from src.preprocessing.windowing import WindowGenerator

WINDOW_SIZE = 1 # 5 para CIC, 10 para UNSW
WINDOW_STEP = 1

# CIC
# train_df = pd.read_parquet('data/processed/CIC-IDS2017/splits/train/data.parquet')
# test_df = pd.read_parquet('data/processed/CIC-IDS2017/splits/test/data.parquet')
# val_df = pd.read_parquet('data/processed/CIC-IDS2017/splits/val/data.parquet')
# dataset = "CIC"
# stratify_column = 'attack_type'
# purify_windows = False
# drop_columns = ["Label", "attack_label", "attack_type"]

# UNSW
train_df = pd.read_csv(Path('data/processed/UNSW-NB15/splits/train.csv'))
test_df = pd.read_csv(Path('data/processed/UNSW-NB15/splits/test.csv'))
val_df = pd.read_csv(Path('data/processed/UNSW-NB15/splits/validation.csv'))
dataset = "UNSW"
stratify_column = 'attack_cat'
purify_windows = False
drop_columns = ["attack_cat", "label", "id"]


X_train = train_df.drop(columns=drop_columns, axis=1)
y_train = train_df[stratify_column]

X_test = test_df.drop(columns=drop_columns, axis=1)
y_test = test_df[stratify_column]

X_val = val_df.drop(columns=drop_columns, axis=1)
y_val = val_df[stratify_column]

#multiclass/binary
preprocessor = joblib.load(f'models/preprocessing/multiclass/{dataset.lower()}/preprocessing.pkl')
label_encoder = joblib.load(f'models/preprocessing/multiclass/{dataset.lower()}/label_encoder.pkl')

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)
X_val_processed = preprocessor.transform(X_val)

y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_val_encoded = label_encoder.transform(y_val)


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

num_features = X_train_processed.shape[1]
num_classes = len(label_encoder.classes_)

print("-------------------------------------------generating time windows...")

if dataset == 'CIC':
    sort_idx = np.argsort(y_train_encoded, kind='stable')
    X_train_processed = X_train_processed.values[sort_idx]
    y_train_encoded   = y_train_encoded[sort_idx]

    sort_idx_test = np.argsort(y_test_encoded, kind='stable')
    X_test_processed = X_test_processed.values[sort_idx_test]
    y_test_encoded   = y_test_encoded[sort_idx_test]

    sort_idx_val = np.argsort(y_val_encoded, kind='stable')
    X_val_processed = X_val_processed.values[sort_idx_val]
    y_val_encoded   = y_val_encoded[sort_idx_val]

window_builder = WindowGenerator(window_size=WINDOW_SIZE, step=WINDOW_STEP, pure_windows_only=purify_windows)

X_train_ae, X_train_seq, y_train_w = window_builder.transform(X_train_processed, y_train_encoded)
X_test_ae, X_test_seq, y_test_w = window_builder.transform(X_test_processed, y_test_encoded)
X_val_ae, X_val_seq, y_val_w = window_builder.transform(X_val_processed, y_val_encoded)

print(f"Shapes after windowing")
print(f"X_train_seq: {X_train_seq.shape}")
print(f"X_train_ae: {X_train_ae.shape}")
print(f"y_train_w: {y_train_w.shape}")

print("-------------------------------------------balancing dataset...")


if dataset == 'UNSW':
    sampler = SMOTEENN(random_state=42)
elif dataset == 'CIC':
    sampler = OneSidedSelection(random_state=42)

idx = np.arange(len(X_train_ae)).reshape(-1, 1)    
idx_bal , y_train_balanced = sampler.fit_resample(X=idx,y=y_train_w)
idx_bal = idx_bal.ravel()

X_train_ae_bal = X_train_ae[idx_bal]
X_train_seq_bal = X_train_seq[idx_bal]

X_train_inputs = {
    "ae_input": X_train_ae_bal,
    "cnn_input": X_train_seq_bal,
    "lstm_input": X_train_seq_bal
}

X_test_inputs = {
    "ae_input": X_test_ae,
    "cnn_input": X_test_seq,
    "lstm_input": X_test_seq
}

X_val_inputs = {
    "ae_input": X_val_ae,
    "cnn_input": X_val_seq,
    "lstm_input": X_val_seq
}

classes = np.unique(y_train_w)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train_balanced
)
class_weight_dict = dict(zip(classes, weights))
print("Class weights:", class_weight_dict)
sample_weights = np.array([class_weight_dict[y] for y in y_train_balanced])

y_train_balanced_ohe = to_categorical(y_train_balanced, num_classes=num_classes)
y_test_ohe = to_categorical(y_test_w, num_classes=num_classes)
y_val_ohe = to_categorical(y_val_w, num_classes=num_classes)



print("-------------------------------------------compiling model...")

model = IDSModelFactory.create_model(window_size=WINDOW_SIZE, num_classes=num_classes, num_features=num_features)

model.compile(
    optimizer="adam", 
    metrics={
        "classification": [
            "accuracy", Precision(name="precision"), Recall(name="recall"), F1Score(name="f1_score", average="macro"), 
            AUC(name="auc", multi_label=True), FalsePositives(name="fp"), FalseNegatives(name="fn"), TopKCategoricalAccuracy(k=3)
            ],
    },
    loss={
        "classification": CategoricalFocalCrossentropy(gamma=3.0, alpha=0.25),
        "reconstruction": "mse"
    },
    loss_weights={
        "classification": 1.0,
        "reconstruction": 0.05,
    }
)

f1_callback = F1EarlyStopping(
    validation_data=(
        X_val_inputs,
        {
            "classification": y_val_ohe,
            "reconstruction": X_val_ae
        }
    ),
    patience=10
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"models/classification/multiclass/{dataset.lower()}/best_model.keras",
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

print("-------------------------------------------training model...")

history = model.fit(
    X_train_inputs,
    {
        "classification": y_train_balanced_ohe,
        "reconstruction": X_train_ae_bal
    },    
    validation_data=(X_val_inputs, 
        {    
            "classification": y_val_ohe,
            "reconstruction": X_val_ae
        }
    ),
    shuffle=True,
    epochs=50, #70-100
    batch_size=256,#128
    callbacks=[f1_callback, checkpoint, reduce_lr],
    sample_weight={
        "classification": sample_weights,
        "reconstruction": np.ones(len(sample_weights))
    },
    verbose=1
)

print("-------------------------------------------testing model...")

test_model = model.evaluate(X_test_inputs, {
    "classification": y_test_ohe,
    "reconstruction": X_test_ae
})

print(f"test results: {test_model}")

y_pred_probs = model.predict(X_test_inputs)["classification"]
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test_w

visuals_path = Path(f"reports/figures/{dataset.lower()}")


visualizer = IDSVisualizer(output_dir=visuals_path)
class_names = sorted(train_df[stratify_column].unique())
print(f"class names: {class_names}")

history_df = pd.DataFrame(history.history)
history_df.to_csv(f"reports/metrics/{dataset.lower()}/multiclass/training_metrics.csv")

report = classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(f"reports/metrics/{dataset.lower()}/multiclass/classification_report.csv")

visualizer.plot_confusion_matrix(y_true=y_true, y_pred=y_pred, classes=(range(num_classes)))
visualizer.plot_roc_curve(y_true=y_true, y_scores=y_pred_probs, classes=class_names)

# IDSModelFactory.save_model(model)
# print("Model has been saved")