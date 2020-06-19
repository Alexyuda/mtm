from scipy import signal
import numpy as np


def MTM_PWC(image, patch, alpha):
    # "Matching by Tone Mapping: Photometric Invariant Template Matching"
    # Y. Hel-Or, H.Hel-Or, E.David.
    # image = PIL grayscale image.
    # patch = PIL image grayscale pattern
    # alpha = k+1 vector of gray bin values.

    image = np.asarray(image)[:, :, 0].astype(np.single)
    patch = np.asarray(patch)[:, :, 0].astype(np.single)
    ones_filter = np.ones(patch.shape)
    m = ones_filter.shape[0] * ones_filter.shape[1]

    W1 = signal.convolve(image, ones_filter, mode='same')
    W2 = signal.convolve((image**2), ones_filter, mode='same')
    D2 = W2 - (W1**2)/m
    D2[np.where(D2 < np.finfo(np.single).eps)] = 1.0
    D1 = np.zeros(D2.shape).astype(np.single)
    for j in range(len(alpha) - 1):
        Sj = ((patch >= alpha[j]) & (patch < alpha[j + 1])).astype(np.single)
        nj = Sj.sum()
        if nj == 0:
            nj = 1
        T = signal.convolve(image, np.rot90(Sj, 2), mode='same')
        T = T**2 / nj
        D1 = D1 + T

    D = (W2 - D1) / D2
    return D




