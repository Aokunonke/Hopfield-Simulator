import numpy as np

# Internal pattern size
IMG_SIZE = (32, 32)
# On-screen preview size
PREVIEW_SIZE = (224, 224)


def sgn(z: np.ndarray) -> np.ndarray:
    out = np.ones_like(z, dtype=float)
    out[z < 0] = -1.0
    return out


class HopfieldNetwork:
    def __init__(self, n_units: int):
        self.n_units = n_units
        self.W = np.zeros((n_units, n_units), dtype=float)

    def train_hebb(self, patterns: np.ndarray):
        """Classic Hebbian learning."""
        patterns = np.asarray(patterns, dtype=float)
        P, N = patterns.shape
        if N != self.n_units:
            raise ValueError("Pattern size does not match network size.")

        self.W = (patterns.T @ patterns) / P
        np.fill_diagonal(self.W, 0.0)

    def train_pseudo_inverse(self, patterns: np.ndarray):
        """Pseudo-inverse learning rule."""
        patterns = np.asarray(patterns, dtype=float)
        P, N = patterns.shape
        if N != self.n_units:
            raise ValueError("Pattern size does not match network size.")

        X = patterns
        G = X @ X.T
        G_inv = np.linalg.pinv(G)
        self.W = X.T @ G_inv @ X
        np.fill_diagonal(self.W, 0.0)

    def energy(self, state: np.ndarray) -> float:
        state = state.reshape(-1, 1).astype(float)
        return float(-0.5 * state.T @ self.W @ state)

    def recall(self, pattern: np.ndarray, max_iters: int = 100):
        state = np.asarray(pattern, dtype=float).flatten()
        if state.size != self.n_units:
            raise ValueError("Input pattern size does not match network size.")

        energies = []
        for _ in range(max_iters):
            prev_state = state.copy()
            state = sgn(self.W @ state)
            energies.append(self.energy(state))
            if np.array_equal(state, prev_state):
                break

        return state, energies