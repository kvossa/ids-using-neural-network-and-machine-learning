import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from typing import Optional

DEFAULT_SHUFFLE_BUFFER = 100_000


def create_class_balanced_dataset(
    X_ae, X_seq, y_int,
    target_counts=None,
    jitter_std=0.005,
    batch_size=128,
    num_classes=10,
    seed=42
):
    """
    Creates dataset with class-balanced sampling.
    Each batch contains equal samples from each class.
    
    target_counts: dict mapping class_index -> target_count
        If None, uses max count across classes
    """
    from tensorflow.keras.utils import to_categorical
    
    rng = np.random.default_rng(seed)
    classes = np.unique(y_int)
    class_counts = {c: np.sum(y_int == c) for c in classes}
    
    # Determine target counts
    if target_counts is None:
        max_count = max(class_counts.values())
        target_counts = {c: min(max_count, class_counts[c] * 10) for c in classes}  # Cap at 10x original
    
    X_ae_balanced = []
    X_seq_balanced = []
    y_balanced = []
    
    for c in classes:
        class_mask = y_int == c
        current_count = class_counts[c]
        target_count = target_counts.get(c, current_count)
        
        X_ae_class = X_ae[class_mask]
        X_seq_class = X_seq[class_mask]
        
        # Oversample to target count
        indices = rng.choice(current_count, size=target_count, replace=True)
        X_ae_class = X_ae_class[indices]
        X_seq_class = X_seq_class[indices]
        
        # Add jitter
        if jitter_std > 0:
            jitter_ae = rng.normal(0, jitter_std, X_ae_class.shape).astype(np.float32)
            jitter_seq = rng.normal(0, jitter_std, X_seq_class.shape).astype(np.float32)
            X_ae_class = X_ae_class + jitter_ae
            X_seq_class = X_seq_class + jitter_seq
        
        X_ae_balanced.append(X_ae_class)
        X_seq_balanced.append(X_seq_class)
        y_balanced.append(np.full(target_count, c, dtype=np.int32))
    
    # Combine
    X_ae_all = np.concatenate(X_ae_balanced, axis=0)
    X_seq_all = np.concatenate(X_seq_balanced, axis=0)
    y_all = np.concatenate(y_balanced, axis=0)
    
    # Shuffle
    shuffle_idx = rng.permutation(len(y_all))
    X_ae_all = X_ae_all[shuffle_idx]
    X_seq_all = X_seq_all[shuffle_idx]
    y_all = y_all[shuffle_idx]
    
    y_all_ohe = to_categorical(y_all, num_classes=num_classes)
    
    n_samples = len(y_all)
    print(f"  Class-balanced: {n_samples} samples, {n_samples // num_classes} per class")
    
    dataset = Dataset.from_tensor_slices((
        {"ae_input": X_ae_all, "cnn_input": X_seq_all, "lstm_input": X_seq_all},
        {"classification": y_all_ohe, "reconstruction": X_ae_all},
    ))
    dataset = dataset.shuffle(min(n_samples, DEFAULT_SHUFFLE_BUFFER), seed=seed).repeat().batch(batch_size, drop_remainder=True)
    
    return dataset, n_samples


def create_smote_oversampled_dataset(
    X_ae, X_seq, y_int,
    oversample_rates=None,
    smote_minority_classes=None,
    smote_target_count=1000,
    jitter_std=0.005,
    original_ratio=0.7,
    batch_size=128,
    num_classes=10,
    seed=42,
    smote_strategy="smote"  # "smote", "adasyn", "borderline1", "borderline2"
):
    """
    Creates dataset with SMOTE variants for minority classes + jitter augmentation.
    
    smote_strategy:
      - "smote": Standard SMOTE
      - "adasyn": Adaptive Synthetic - focuses on boundary samples
      - "borderline1": Borderline-SMOTE (only dangerous minority neighbors)
      - "borderline2": Borderline-SMOTE (dangerous + safe minority neighbors)
    """
    if oversample_rates is None:
        oversample_rates = {}
    if smote_minority_classes is None:
        smote_minority_classes = []
    
    from tensorflow.keras.utils import to_categorical
    
    # Try to import imblearn
    HAS_IMBLEARN = False
    try:
        from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
        HAS_IMBLEARN = True
    except ImportError:
        pass
    
    rng = np.random.default_rng(seed)
    classes = np.unique(y_int)
    n_total = len(y_int)
    
    # Calculate target counts for each class
    class_counts = {c: np.sum(y_int == c) for c in classes}
    
    # === Synthetic data generation for minority classes ===
    X_ae_synthetic = []
    X_seq_synthetic = []
    y_synthetic = []
    
    if HAS_IMBLEARN and smote_minority_classes and smote_strategy != "none":
        print(f"  Applying {smote_strategy.upper()} to classes: {smote_minority_classes}")
        
        for c in smote_minority_classes:
            current_count = class_counts[c]
            if current_count >= smote_target_count:
                print(f"    Class {c}: already has {current_count} >= {smote_target_count}, skipping")
                continue
            
            # Get minority samples
            class_mask = y_int == c
            X_minority_ae = X_ae[class_mask]
            X_minority_seq = X_seq[class_mask]
            n_to_generate = smote_target_count - current_count
            
            try:
                # Set up the SMOTE variant
                k_neighbors = min(5, current_count - 1)
                if k_neighbors < 1:
                    raise ValueError("Not enough samples")
                
                if smote_strategy == "adasyn":
                    sampler = ADASYN(
                        sampling_strategy={c: smote_target_count},
                        n_neighbors=k_neighbors,
                        random_state=seed
                    )
                elif smote_strategy == "borderline1":
                    sampler = BorderlineSMOTE(
                        sampling_strategy={c: smote_target_count},
                        k_neighbors=k_neighbors,
                        kind="borderline-1",
                        random_state=seed
                    )
                elif smote_strategy == "borderline2":
                    sampler = BorderlineSMOTE(
                        sampling_strategy={c: smote_target_count},
                        k_neighbors=k_neighbors,
                        kind="borderline-2",
                        random_state=seed
                    )
                else:
                    sampler = SMOTE(
                        sampling_strategy={c: smote_target_count},
                        k_neighbors=k_neighbors,
                        random_state=seed
                    )
                
                # Fit and generate
                X_combined = X_minority_ae  # Use AE features
                y_dummy = np.full(current_count, c)
                
                X_resampled, y_resampled = sampler.fit_resample(X_combined, y_dummy)
                
                # Get synthetic samples (after original count)
                n_original = current_count
                n_syn = len(X_resampled) - n_original
                
                if n_syn > 0:
                    X_ae_syn = X_resampled[n_original:] + rng.normal(0, jitter_std, (n_syn, X_resampled.shape[1])).astype(np.float32)
                    # Reconstruct sequence windows (use nearest neighbor + noise)
                    X_seq_syn = np.zeros((n_syn, X_seq.shape[1], X_seq.shape[2]), dtype=np.float32)
                    for i in range(n_syn):
                        nn_idx = rng.integers(0, current_count)
                        X_seq_syn[i] = X_minority_seq[nn_idx] + rng.normal(0, jitter_std, X_minority_seq[nn_idx].shape).astype(np.float32)
                    
                    X_ae_synthetic.append(X_ae_syn)
                    X_seq_synthetic.append(X_seq_syn)
                    y_synthetic.append(np.full(n_syn, c, dtype=np.int32))
                    print(f"    Class {c}: generated {n_syn} synthetic samples via {smote_strategy}")
                else:
                    print(f"    Class {c}: no new samples generated")
                    
            except Exception as e:
                print(f"    Class {c}: {smote_strategy} failed ({e}), using oversampling fallback")
                # Fallback: oversample with enhanced jitter
                source_indices = rng.choice(current_count, size=n_to_generate, replace=True)
                X_ae_syn = X_minority_ae[source_indices] + rng.normal(0, jitter_std * 2, (n_to_generate, X_minority_ae.shape[1])).astype(np.float32)
                X_seq_syn = np.zeros((n_to_generate, X_seq.shape[1], X_seq.shape[2]), dtype=np.float32)
                for i in range(n_to_generate):
                    nn_idx = source_indices[i]
                    X_seq_syn[i] = X_minority_seq[nn_idx] + rng.normal(0, jitter_std * 2, X_minority_seq[nn_idx].shape).astype(np.float32)
                X_ae_synthetic.append(X_ae_syn)
                X_seq_synthetic.append(X_seq_syn)
                y_synthetic.append(np.full(n_to_generate, c, dtype=np.int32))
    
    # === Keep existing oversampling logic for other classes ===
    n_original_samples = int(n_total * original_ratio)
    original_indices = rng.choice(n_total, size=n_original_samples, replace=False)
    original_indices = np.sort(original_indices)
    
    X_ae_original = X_ae[original_indices]
    X_seq_original = X_seq[original_indices]
    y_original = y_int[original_indices]
    
    remaining_indices = np.setdiff1d(np.arange(n_total), original_indices)
    X_ae_remain = X_ae[remaining_indices]
    X_seq_remain = X_seq[remaining_indices]
    y_remain = y_int[remaining_indices]
    
    X_ae_oversampled = []
    X_seq_oversampled = []
    y_oversampled = []
    
    for c in classes:
        class_mask = y_remain == c
        if not np.any(class_mask):
            continue
        class_indices = np.where(class_mask)[0]
        n_original = len(class_indices)
        
        rate = oversample_rates.get(c, 1)
        if rate > 1:
            oversample_indices = rng.choice(n_original, size=n_original * (rate - 1), replace=True)
            X_ae_oversampled.append(X_ae_remain[class_indices][oversample_indices])
            X_seq_oversampled.append(X_seq_remain[class_indices][oversample_indices])
            y_oversampled.append(np.full(len(oversample_indices), c, dtype=np.int32))
            
            if jitter_std > 0:
                jitter_ae = rng.normal(0, jitter_std, X_ae_oversampled[-1].shape).astype(np.float32)
                jitter_seq = rng.normal(0, jitter_std, X_seq_oversampled[-1].shape).astype(np.float32)
                X_ae_oversampled[-1] = X_ae_oversampled[-1] + jitter_ae
                X_seq_oversampled[-1] = X_seq_oversampled[-1] + jitter_seq
    
    # === Combine all ===
    components_ae = [X_ae_original]
    components_seq = [X_seq_original]
    components_y = [y_original]
    
    if X_ae_synthetic:
        components_ae.append(np.concatenate(X_ae_synthetic, axis=0))
        components_seq.append(np.concatenate(X_seq_synthetic, axis=0))
        components_y.append(np.concatenate(y_synthetic, axis=0))
    
    if X_ae_oversampled:
        components_ae.append(np.concatenate(X_ae_oversampled, axis=0))
        components_seq.append(np.concatenate(X_seq_oversampled, axis=0))
        components_y.append(np.concatenate(y_oversampled, axis=0))
    
    X_ae_all = np.concatenate(components_ae, axis=0)
    X_seq_all = np.concatenate(components_seq, axis=0)
    y_all = np.concatenate(components_y, axis=0)
    
    # Shuffle
    shuffle_idx = rng.permutation(len(y_all))
    X_ae_all = X_ae_all[shuffle_idx]
    X_seq_all = X_seq_all[shuffle_idx]
    y_all = y_all[shuffle_idx]
    
    y_all_ohe = to_categorical(y_all, num_classes=num_classes)
    
    n_samples = len(y_all)
    print(f"  Total samples: {n_samples} (original: {n_original_samples})")
    
    dataset = Dataset.from_tensor_slices((
        {"ae_input": X_ae_all, "cnn_input": X_seq_all, "lstm_input": X_seq_all},
        {"classification": y_all_ohe, "reconstruction": X_ae_all},
    ))
    dataset = dataset.shuffle(min(n_samples, DEFAULT_SHUFFLE_BUFFER), seed=seed).repeat().batch(batch_size, drop_remainder=True)
    
    return dataset, n_samples


class BalancedBatchGenerator:
    def __init__(self, X_ae, X_seq, y, batch_size=128, attack_weight=30.0, normal_weight=1.0):
        self.X_ae = X_ae
        self.X_seq = X_seq
        self.y = y
        self.batch_size = batch_size
        self.n_samples = len(y)

        self._normal_idx = np.where(y == 0)[0]
        self._attack_idx = np.where(y == 1)[0]

        normal_weight_adj = normal_weight / len(self._normal_idx)
        attack_weight_adj = attack_weight / len(self._attack_idx)

        self._weights = np.zeros(self.n_samples)
        self._weights[self._normal_idx] = normal_weight_adj
        self._weights[self._attack_idx] = attack_weight_adj
        self._weights /= self._weights.sum()
        self._rng = np.random.default_rng(42)

    def __iter__(self):
        return self

    def __next__(self):
        indices = self._rng.choice(self.n_samples, size=self.batch_size, replace=True, p=self._weights)
        return (
            {
                "ae_input": self.X_ae[indices],
                "cnn_input": self.X_seq[indices],
                "lstm_input": self.X_seq[indices],
            },
            self.y[indices]
        )


class MulticlassBalancedGenerator:
    def __init__(self, X_ae, X_seq, y, batch_size=128, base_weight: float = 1.0, alpha: Optional[float] = None):
        self.X_ae = X_ae
        self.X_seq = X_seq
        self.y = y
        self.batch_size = batch_size
        self.n_samples = len(y)
        self.classes = np.unique(y)
        self._rng = np.random.default_rng(42)

        if alpha is None:
            alpha = 1.0

        counts = {c: np.sum(y == c) for c in self.classes}
        class_weights = {c: (self.n_samples / (len(self.classes) * counts[c])) ** alpha for c in self.classes}

        self._weights = np.array([class_weights[y[i]] for i in range(self.n_samples)], dtype=np.float64)
        self._weights /= self._weights.sum()

    def __iter__(self):
        return self

    def __next__(self):
        indices = self._rng.choice(self.n_samples, size=self.batch_size, replace=True, p=self._weights)
        return (
            {
                "ae_input": self.X_ae[indices],
                "cnn_input": self.X_seq[indices],
                "lstm_input": self.X_seq[indices],
            },
            self.y[indices]
        )


def create_balanced_tf_dataset(
    X_ae,
    X_seq,
    y,
    batch_size=128,
    attack_weight=1.0,
    normal_weight=1.0,
    shuffle_seed=42,
    extra_sample_weight=None,
):
    """
    Binary training pipeline with per-sample loss weights (class 0 = normal, 1 = attack).

    ``y`` may be ``(n, 2)`` one-hot (as from ``to_categorical``) or ``(n,)`` integer labels.

    Base weights follow sklearn's ``balanced`` rule ``n / (n_classes * count_k)``;
    ``normal_weight`` and ``attack_weight`` multiply the base weight of that class.
    Do not set ``attack_weight`` ≫ ``normal_weight`` when attack is already the majority
    (e.g. CIC), or the loss will keep favoring the majority class.
    """
    from tensorflow.keras.utils import to_categorical

    y_arr = np.asarray(y)
    if y_arr.ndim == 2 and y_arr.shape[1] >= 2:
        y_int = np.argmax(y_arr, axis=1).astype(np.int32)
        y_out = y_arr.astype(np.float32)
    else:
        y_int = y_arr.astype(np.int32).ravel()
        y_out = to_categorical(y_int, num_classes=2)

    n_samples = len(y_int)
    counts = np.bincount(y_int, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    base0 = n_samples / (2.0 * counts[0])
    base1 = n_samples / (2.0 * counts[1])

    per_sample = np.where(
        y_int == 0,
        base0 * float(normal_weight),
        base1 * float(attack_weight),
    ).astype(np.float32)
    if extra_sample_weight is not None:
        extra = np.asarray(extra_sample_weight, dtype=np.float32).ravel()
        if len(extra) != n_samples:
            raise ValueError(
                f"extra_sample_weight length mismatch: expected {n_samples}, got {len(extra)}"
            )
        # Avoid invalid negative/zero multipliers.
        extra = np.maximum(extra, 1e-6)
        per_sample *= extra
    per_sample /= float(np.mean(per_sample))

    sample_weight = {
        "classification": per_sample,
        "reconstruction": np.ones(n_samples, dtype=np.float32),
    }

    dataset = Dataset.from_tensor_slices((
        {"ae_input": X_ae, "cnn_input": X_seq, "lstm_input": X_seq},
        {"classification": y_out, "reconstruction": X_ae},
        sample_weight,
    ))
    dataset = dataset.shuffle(min(n_samples, DEFAULT_SHUFFLE_BUFFER), seed=int(shuffle_seed)).repeat().batch(batch_size, drop_remainder=True)
    return dataset


def create_multiclass_tf_dataset(X_ae, X_seq, y_int, y_labels=None, batch_size=128, alpha: float = 1.0):
    if y_labels is None:
        y_labels = y_int

    n_samples = len(y_int)
    classes = np.unique(y_int)
    counts = {c: np.sum(y_int == c) for c in classes}
    class_weights = {c: (n_samples / (len(classes) * counts[c])) ** alpha for c in classes}

    sample_weights = np.array([class_weights[y_int[i]] for i in range(n_samples)], dtype=np.float32)

    dataset = Dataset.from_tensor_slices((
        {"ae_input": X_ae, "cnn_input": X_seq, "lstm_input": X_seq},
        {"classification": y_labels, "reconstruction": X_ae},
    ))
    dataset = dataset.shuffle(min(n_samples, DEFAULT_SHUFFLE_BUFFER), seed=42).repeat().batch(batch_size, drop_remainder=True)
    return dataset, sample_weights


def create_oversampled_dataset(
    X_ae, X_seq, y_int,
    oversample_rates=None,
    num_classes=10,
    jitter_std=0.005,
    batch_size=128,
    seed=42
):
    """
    Creates a dataset with per-class oversampling and optional jitter augmentation.
    
    oversample_rates: dict mapping class_index → oversample_rate
        Example: {0: 20, 1: 5, 2: 5, 8: 5, 9: 20}
        Class indices: 0=Analysis, 1=Backdoor, 2=DoS, 3=Exploits, 4=Fuzzers,
                   5=Generic, 6=Normal, 7=Reconnaissance, 8=Shellcode, 9=Worms
    """
    if oversample_rates is None:
        oversample_rates = {}
    
    from tensorflow.keras.utils import to_categorical
    
    rng = np.random.default_rng(seed)
    classes = np.unique(y_int)
    
    X_ae_oversampled = []
    X_seq_oversampled = []
    y_oversampled = []
    
    for c in classes:
        class_mask = y_int == c
        class_indices = np.where(class_mask)[0]
        n_original = len(class_indices)
        
        rate = oversample_rates.get(c, 1)
        n_oversampled = n_original * rate
        
        X_ae_class = X_ae[class_indices]
        X_seq_class = X_seq[class_indices]
        y_class = y_int[class_indices]
        
        if rate > 1:
            oversample_indices = rng.choice(n_original, size=n_oversampled, replace=True)
            X_ae_oversampled.append(X_ae_class[oversample_indices])
            X_seq_oversampled.append(X_seq_class[oversample_indices])
            y_oversampled.append(y_class[oversample_indices])
        else:
            X_ae_oversampled.append(X_ae_class)
            X_seq_oversampled.append(X_seq_class)
            y_oversampled.append(y_class)
        
        if rate > 1 and jitter_std > 0:
            jitter_ae = rng.normal(0, jitter_std, X_ae_oversampled[-1].shape).astype(np.float32)
            jitter_seq = rng.normal(0, jitter_std, X_seq_oversampled[-1].shape).astype(np.float32)
            
            X_ae_oversampled[-1] = X_ae_oversampled[-1] + jitter_ae
            X_seq_oversampled[-1] = X_seq_oversampled[-1] + jitter_seq
    
    X_ae_all = np.concatenate(X_ae_oversampled, axis=0)
    X_seq_all = np.concatenate(X_seq_oversampled, axis=0)
    y_all = np.concatenate(y_oversampled, axis=0)
    
    shuffle_idx = rng.permutation(len(y_all))
    X_ae_all = X_ae_all[shuffle_idx]
    X_seq_all = X_seq_all[shuffle_idx]
    y_all = y_all[shuffle_idx]
    
    y_all_ohe = to_categorical(y_all, num_classes=num_classes)
    
    n_samples = len(y_all)
    dataset = Dataset.from_tensor_slices((
        {"ae_input": X_ae_all, "cnn_input": X_seq_all, "lstm_input": X_seq_all},
        {"classification": y_all_ohe, "reconstruction": X_ae_all},
    ))
    dataset = dataset.shuffle(min(n_samples, DEFAULT_SHUFFLE_BUFFER), seed=seed).repeat().batch(batch_size, drop_remainder=True)
    
    return dataset, n_samples


def create_hybrid_mix_dataset(
    X_ae, X_seq, y_int,
    oversample_rates=None,
    original_ratio=0.8,
    jitter_std=0.005,
    batch_size=128,
    seed=42
):
    """
    Creates a hybrid dataset mixing original and oversampled data.
    
    original_ratio: portion of original data (0.7 = 70% original, 30% oversampled)
    oversample_rates: dict mapping class_index → oversample_rate
    
    Mix strategy:
    - Sample 70% of original data (stratified if possible)
    - Add 30% oversampled data (with jitter if rate > 1)
    - Concatenate and shuffle
    """
    if oversample_rates is None:
        oversample_rates = {}
    
    from tensorflow.keras.utils import to_categorical
    
    rng = np.random.default_rng(seed)
    n_total = len(y_int)
    n_original_samples = int(n_total * original_ratio)
    n_oversample_samples = n_total - n_original_samples
    
    original_indices = rng.choice(n_total, size=n_original_samples, replace=False)
    original_indices = np.sort(original_indices)
    
    X_ae_original = X_ae[original_indices]
    X_seq_original = X_seq[original_indices]
    y_original = y_int[original_indices]
    
    classes = np.unique(y_int)
    
    X_ae_oversampled = []
    X_seq_oversampled = []
    y_oversampled = []
    
    for c in classes:
        class_mask = y_int == c
        class_indices = np.where(class_mask)[0]
        n_original = len(class_indices)
        
        rate = oversample_rates.get(c, 1)
        target_for_class = int(n_oversample_samples * (rate / sum(oversample_rates.values())))
        
        X_ae_class = X_ae[class_indices]
        X_seq_class = X_seq[class_indices]
        y_class = y_int[class_indices]
        
        if rate > 1 and target_for_class > 0:
            oversample_indices = rng.choice(n_original, size=target_for_class, replace=True)
            X_ae_oversampled.append(X_ae_class[oversample_indices])
            X_seq_oversampled.append(X_seq_class[oversample_indices])
            y_oversampled.append(y_class[oversample_indices])
            
            if jitter_std > 0:
                jitter_ae = rng.normal(0, jitter_std, X_ae_oversampled[-1].shape).astype(np.float32)
                jitter_seq = rng.normal(0, jitter_std, X_seq_oversampled[-1].shape).astype(np.float32)
                X_ae_oversampled[-1] = X_ae_oversampled[-1] + jitter_ae
                X_seq_oversampled[-1] = X_seq_oversampled[-1] + jitter_seq
        else:
            pass
    
    if y_oversampled:
        X_ae_extra = np.concatenate(X_ae_oversampled, axis=0)
        X_seq_extra = np.concatenate(X_seq_oversampled, axis=0)
        y_extra = np.concatenate(y_oversampled, axis=0)
    else:
        X_ae_extra = np.array([]).reshape(0, *X_ae.shape[1:])
        X_seq_extra = np.array([]).reshape(0, *X_seq.shape[1:])
        y_extra = np.array([], dtype=np.int32)
    
    X_ae_all = np.concatenate([X_ae_original, X_ae_extra], axis=0)
    X_seq_all = np.concatenate([X_seq_original, X_seq_extra], axis=0)
    y_all = np.concatenate([y_original, y_extra], axis=0)
    
    shuffle_idx = rng.permutation(len(y_all))
    X_ae_all = X_ae_all[shuffle_idx]
    X_seq_all = X_seq_all[shuffle_idx]
    y_all = y_all[shuffle_idx]
    
    num_classes = len(classes)
    y_all_ohe = to_categorical(y_all, num_classes=num_classes)
    
    n_samples = len(y_all)
    dataset = Dataset.from_tensor_slices((
        {"ae_input": X_ae_all, "cnn_input": X_seq_all, "lstm_input": X_seq_all},
        {"classification": y_all_ohe, "reconstruction": X_ae_all},
    ))
    dataset = dataset.shuffle(min(n_samples, DEFAULT_SHUFFLE_BUFFER), seed=seed).repeat().batch(batch_size, drop_remainder=True)
    
    return dataset, n_samples

    
        