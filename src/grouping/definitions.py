import numpy as np

# ============================================================
# CIC-IDS2017
# Original classes (label encoder alphabetical order):
#   0=BENIGN, 1=Botnet, 2=Bruteforce, 3=DDoS, 4=DoS,
#   5=Infiltration, 6=Portscan, 7=WebAttacks
# BENIGN is excluded — Stage 2 never trains on it.
# Grouping: 7 attack types → 3 groups
# ============================================================

CIC_GROUP_MAP = {
    "Botnet": "Rare",
    "Bruteforce": "Bruteforce",
    "DDoS": "Flood",
    "DoS": "Flood",
    "Infiltration": "Rare",
    "Portscan": "Rare",
    "WebAttacks": "Rare",
}

# Group names sorted by size (Flood largest first) for stable index order
CIC_GROUP_NAMES = ["Flood", "Bruteforce", "Rare"]

CIC_CLASS_ALPHA = {
    0: 0.30,  # Flood — lower alpha, well represented
    1: 0.45,  # Bruteforce — medium
    2: 0.50,  # Rare — high alpha to improve recall
}

CIC_OVERSAMPLE_RATES = {
    0: 1,    # Flood (DDoS + DoS) — well represented
    1: 5,    # Bruteforce — moderate oversample
    2: 10,   # Rare — high oversample to improve recall
}

# ============================================================
# CIC Single-Stage (includes BENIGN as a group)
# 8 original classes → 3 groups:
#   BENIGN(0), FloodBruteforce(1) = DoS+DDoS+Bruteforce,
#   Rare(2) = Botnet+Infiltration+Portscan+WebAttacks
# ============================================================

CIC_SINGLE_GROUP_MAP = {
    "BENIGN": "BENIGN",
    "Botnet": "Rare",
    "Bruteforce": "FloodBruteforce",
    "DDoS": "FloodBruteforce",
    "DoS": "FloodBruteforce",
    "Infiltration": "Rare",
    "Portscan": "Rare",
    "WebAttacks": "Rare",
}

CIC_SINGLE_GROUP_NAMES = ["BENIGN", "FloodBruteforce", "Rare"]

CIC_SINGLE_CLASS_ALPHA = {
    0: 0.35,  # BENIGN — minority, boost
    1: 0.15,  # FloodBruteforce — majority, default
    2: 0.50,  # Rare — needs focus
}

CIC_SINGLE_OVERSAMPLE_RATES = {
    0: 3,    # BENIGN — oversample to match attack classes
    1: 1,    # FloodBruteforce — well represented
    2: 2,    # Rare — slight boost
}

# ============================================================
# CIC 2-Group (diagnostic — merge Bruteforce into Flood)
# 7 attack types → 2 groups:
#   FloodBruteforce(0) = DDoS+DoS+Bruteforce
#   Rare(1) = Botnet+Infiltration+Portscan+WebAttacks
# BENIGN excluded — Stage 2 never trains on it.
# ============================================================

CIC_2GROUP_MAP = {
    "Bruteforce": "FloodBruteforce",
    "DDoS": "FloodBruteforce",
    "DoS": "FloodBruteforce",
    "Botnet": "Rare",
    "Infiltration": "Rare",
    "Portscan": "Rare",
    "WebAttacks": "Rare",
}

CIC_2GROUP_NAMES = ["FloodBruteforce", "Rare"]

CIC_2GROUP_CLASS_ALPHA = {
    0: 0.35,  # FloodBruteforce — majority
    1: 0.65,  # Rare — needs boost
}

CIC_2GROUP_OVERSAMPLE_RATES = {
    0: 1,    # FloodBruteforce — well represented
    1: 3,    # Rare — moderate oversample
}

# ============================================================
# CIC Bruteforce→Rare regroup (move Bruteforce into Rare)
# 7 attack types → 2 groups:
#   Flood(0) = DDoS+DoS (standalone, clean)
#   Rare(1) = Bruteforce+Botnet+Infiltration+Portscan+WebAttacks
# ~45:55 split. BENIGN excluded.
# ============================================================

CIC_BRUTERARE_MAP = {
    "DDoS": "Flood",
    "DoS": "Flood",
    "Bruteforce": "Rare",
    "Botnet": "Rare",
    "Infiltration": "Rare",
    "Portscan": "Rare",
    "WebAttacks": "Rare",
}

CIC_BRUTERARE_NAMES = ["Flood", "Rare"]

CIC_BRUTERARE_CLASS_ALPHA = {
    0: 0.50,  # Flood — slightly smaller group
    1: 0.50,  # Rare — slightly larger
}

CIC_BRUTERARE_OVERSAMPLE_RATES = {
    0: 1,    # Flood — well represented
    1: 1,    # Rare — well represented
}

# ============================================================
# UNSW-NB15
# Original classes (label encoder alphabetical order):
#   0=Analysis, 1=Backdoor, 2=DoS, 3=Exploits, 4=Fuzzers,
#   5=Generic, 6=Normal, 7=Reconnaissance, 8=Shellcode, 9=Worms
# Normal is excluded — Stage 2 never trains on it.
# Grouping: 9 attack types → 3 groups
# ============================================================

UNSW_GROUP_MAP = {
    "Analysis": "Rare",
    "Backdoor": "Medium",
    "DoS": "Medium",
    "Exploits": "HighFreq",
    "Fuzzers": "HighFreq",
    "Generic": "HighFreq",
    "Reconnaissance": "HighFreq",
    "Shellcode": "Rare",
    "Worms": "Rare",
}

UNSW_GROUP_NAMES = ["HighFreq", "Medium", "Rare"]

UNSW_CLASS_ALPHA = {
    0: 0.15,  # HighFreq — common
    1: 0.65,  # Medium — increased
    2: 0.85,  # Rare — increased
}

UNSW_OVERSAMPLE_RATES = {
    0: 1,    # HighFreq — well represented
    1: 10,   # Medium — increased from 5
    2: 25,   # Rare — increased from 15
}


def build_group_mapping(label_encoder, group_map, group_names, normal_label):
    """Build an array mapping origbuild_group_mappinginal class indices to group indices.

    Args:
        label_encoder: sklearn LabelEncoder with .classes_ attribute
        group_map: dict mapping class_name -> group_name (for attack classes only)
        group_names: ordered list of group names (index = group index)
        normal_label: the normal class name to exclude (e.g., 'BENIGN' or 'Normal')

    Returns:
        original_to_group: ndarray of shape (num_original_classes,) where
            original_to_group[i] = group_index if class i is an attack class
            original_to_group[i] = -1 if class i is the normal label
    """
    group_name_to_idx = {name: i for i, name in enumerate(group_names)}
    num_classes = len(label_encoder.classes_)
    original_to_group = np.full(num_classes, -1, dtype=np.int32)

    for orig_idx, class_name in enumerate(label_encoder.classes_):
        if normal_label is not None and class_name == normal_label:
            continue
        group_name = group_map.get(class_name)
        if group_name is not None:
            original_to_group[orig_idx] = group_name_to_idx[group_name]

    return original_to_group


# ============================================================
# UNSW Confusion-Based Grouping (for single-stage training)
# Merges classes the model cannot separate:
#   Analysis + Backdoor + DoS → "Medium"
#   Reconnaissance + Shellcode → "Recon-Shellcode"
#   Others kept standalone
# All 10 classes are included (Normal is a valid group).
# ============================================================

UNSW_CONFUSION_GROUP_MAP = {
    "Analysis": "Medium",
    "Backdoor": "Medium",
    "DoS": "Medium",
    "Exploits": "Exploits",
    "Fuzzers": "Fuzzers",
    "Generic": "Generic",
    "Normal": "Normal",
    "Reconnaissance": "Recon-Shellcode",
    "Shellcode": "Recon-Shellcode",
    "Worms": "Worms",
}

UNSW_CONFUSION_GROUP_NAMES = [
    "Normal", "Generic", "Exploits", "Fuzzers",
    "Recon-Shellcode", "Medium", "Worms",
]

UNSW_CONFUSION_OVERSAMPLE_RATES = {
    0: 1,    # Normal
    1: 1,    # Generic
    2: 1,    # Exploits
    3: 1,    # Fuzzers
    4: 8,    # Recon-Shellcode
    5: 15,   # Medium
    6: 50,   # Worms
}

UNSW_CONFUSION_CLASS_ALPHA = {
    0: 0.15,  # Normal
    1: 0.15,  # Generic
    2: 0.15,  # Exploits
    3: 0.25,  # Fuzzers
    4: 0.50,  # Recon-Shellcode
    5: 0.60,  # Medium
    6: 0.80,  # Worms
}
