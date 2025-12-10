import numpy as np

def normalize(vec) -> np.ndarray:
    vec = np.asarray(vec)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm
