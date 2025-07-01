# data_augmentation.py

import numpy as np

def add_gaussian_noise(X, mean=0.0, std=0.01):
    noise = np.random.normal(mean, std, X.shape)
    return X + noise

def apply_scaling(X, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return X * scale

def apply_shift(X, shift_range=(-0.1, 0.1)):
    shift = np.random.uniform(shift_range[0], shift_range[1])
    return X + shift

def apply_dropout_mask(X, dropout_prob=0.1):
    mask = np.random.binomial(1, 1-dropout_prob, X.shape)
    return X * mask

def augment_batch(X_batch):
    """
    X_batch: numpy array (batch_size, window, feature_dim)
    returns augmented X_batch
    """
    X_aug = []
    for X in X_batch:
        X1 = add_gaussian_noise(X)
        X2 = apply_scaling(X1)
        X3 = apply_shift(X2)
        X4 = apply_dropout_mask(X3)
        X_aug.append(X4)
    return np.array(X_aug, dtype=np.float32)
