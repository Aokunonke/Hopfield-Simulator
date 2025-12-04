import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np


# ------------- Model: Hopfield network and helpers -------------

IMG_SIZE = (32, 32)  # all images are resized to this


def sgn(z: np.ndarray) -> np.ndarray:
    """Sign activation: +1 for z >= 0, -1 for z < 0."""
    out = np.ones_like(z, dtype=float)
    out[z < 0] = -1.0
    return out


class HopfieldNetwork:
    def __init__(self, n_units: int):
        self.n_units = n_units
        self.W = np.zeros((n_units, n_units), dtype=float)

    def train_hebb(self, patterns: np.ndarray):
        """
        Train with Hebbian rule.
        patterns: shape (n_patterns, n_units) with values -1 or +1.
        """
        patterns = np.asarray(patterns, dtype=float)
        n_patterns, n_units = patterns.shape
        if n_units != self.n_units:
            raise ValueError("Pattern size does not match network size.")

        # Hebbian learning: average outer product of patterns
        self.W = (patterns.T @ patterns) / n_patterns

        # No self-connections
        np.fill_diagonal(self.W, 0.0)

    def energy(self, state: np.ndarray) -> float:
        state = state.reshape(-1, 1).astype(float)
        return float(-0.5 * state.T @ self.W @ state)

    def recall(self, pattern: np.ndarray, max_iters: int = 100, synchronous: bool = True):
        """
        Recall from an initial pattern.
        pattern: 1D array length n_units, values -1/+1.
        Returns (final_state, energy_list).
        """
        state = np.asarray(pattern, dtype=float).flatten()
        if state.size != self.n_units:
            raise ValueError("Input pattern size does not match network size.")

        energies = []
        for _ in range(max_iters):
            prev_state = state.copy()

            if synchronous:
                state = sgn(self.W @ state)
            else:
                # asynchronous update (not used by the GUI but kept available)
                for i in range(self.n_units):
                    h_i = np.dot(self.W[i, :], state)
                    state[i] = 1.0 if h_i >= 0 else -1.0

            energies.append(self.energy(state))

            if np.array_equal(state, prev_state):
                break

        return state, energies


def image_to_pattern(img_array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert grayscale image array to a Hopfield pattern (-1/+1).
    img_array: 2D numpy array.
    threshold: applied on normalized [0,1] image.
    """
    img = img_array.astype(float)
    if img.max() > 1.0:
        img = img / 255.0
    binary = np.where(img >= threshold, 1.0, -1.0)
    return binary.flatten()


def pattern_to_image(pattern: np.ndarray, shape) -> np.ndarray:
    """
    Convert 1D pattern (-1/+1) back to 2D image [0,255] for display.
    shape: (H, W).
    """
    pattern = np.asarray(pattern, dtype=float).flatten()
    img = pattern.reshape(shape)
    img = (img + 1.0) / 2.0  # -1 -> 0, +1 -> 1
    img = img * 255.0
    return img.astype("uint8")


# ------------- View/Controller: Tkinter GUI -------------

class HopfieldGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hopfield Image Simulator")

        # Data
        self.patterns = None          # training patterns (numpy array)
        self.labels = []              # labels for training patterns
        self.stored_patterns = None   # copy of patterns
        self.net = None               # HopfieldNetwork instance
        self.test_pattern = None      # base test pattern (no noise)

        # UI layout
        self._build_widgets()

    # ----- UI setup -----

    def _build_widgets(self):
        # Top buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Button(btn_frame, text="Load training images",
                  command=self.load_training_images).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Train network",
                  command=self.train_network).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Load test image",
                  command=self.load_test_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Recall",
                  command=self.recall_pattern).pack(side=tk.LEFT, padx=5)

        # Noise slider
        noise_frame = tk.Frame(self.root)
        noise_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(noise_frame, text="Noise level (%)").pack(side=tk.LEFT)
        self.noise_scale = tk.Scale(
            noise_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            length=200
        )
        self.noise_scale.set(0)
        self.noise_scale.pack(side=tk.LEFT, padx=10)

        # Image display
        img_frame = tk.Frame(self.root)
        img_frame.pack(side=tk.TOP, padx=10, pady=10)

        left_frame = tk.Frame(img_frame)
        left_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(left_frame, text="Input (after noise)").pack()
        self.canvas_input = tk.Label(left_frame)
        self.canvas_input.pack()

        right_frame = tk.Frame(img_frame)
        right_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(right_frame, text="Recalled").pack()
        self.canvas_output = tk.Label(right_frame)
        self.canvas_output.pack()

        # Status bar
        self.status = tk.Label(self.root, text="", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # ----- Helpers for image <-> pattern -----

    def _load_image_as_pattern(self, path):
        img = Image.open(path).convert("L").resize(IMG_SIZE)
        arr = np.array(img, dtype=float)
        return image_to_pattern(arr, threshold=0.5)

    def _pattern_to_photo(self, pattern):
        H, W = IMG_SIZE
        img_arr = pattern_to_image(pattern, (H, W))
        img = Image.fromarray(img_arr, mode="L").resize((128, 128))
        return ImageTk.PhotoImage(img)

    # ----- Controller actions -----

    def load_training_images(self):
        paths = filedialog.askopenfilenames(
            title="Select training images",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not paths:
            return

        patterns = []
        labels = []

        for path in paths:
            pat = self._load_image_as_pattern(path)
            patterns.append(pat)
            labels.append(os.path.basename(path))

        self.patterns = np.vstack(patterns)
        self.stored_patterns = self.patterns.copy()
        self.labels = labels

        self.status.config(text=f"Loaded {len(paths)} training images.")

    def train_network(self):
        if self.patterns is None:
            messagebox.showerror("Error", "Load training images first.")
            return

        n_units = self.patterns.shape[1]
        self.net = HopfieldNetwork(n_units)
        self.net.train_hebb(self.patterns)

        self.status.config(text="Network trained.")

    def load_test_image(self):
        path = filedialog.askopenfilename(
            title="Select test image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not path:
            return

        self.test_pattern = self._load_image_as_pattern(path)

        # Show original test image (no noise yet)
        img = self._pattern_to_photo(self.test_pattern)
        self.canvas_input.config(image=img)
        self.canvas_input.image = img

        self.status.config(text=f"Loaded test image: {os.path.basename(path)}")

    def recall_pattern(self):
        if self.net is None:
            messagebox.showerror("Error", "Train the network first.")
            return
        if self.test_pattern is None:
            messagebox.showerror("Error", "Load a test image first.")
            return

        # Copy base test pattern and apply noise
        pattern = self.test_pattern.copy()
        noise_level = self.noise_scale.get()

        if noise_level > 0:
            n_units = pattern.size
            flip_prob = noise_level / 100.0
            mask = np.random.rand(n_units) < flip_prob
            pattern[mask] *= -1.0

        # Show the (possibly noisy) input pattern
        input_img = self._pattern_to_photo(pattern)
        self.canvas_input.config(image=input_img)
        self.canvas_input.image = input_img

        # Recall
        recalled, energies = self.net.recall(pattern, max_iters=50, synchronous=True)

        # Show recalled pattern
        rec_img = self._pattern_to_photo(recalled)
        self.canvas_output.config(image=rec_img)
        self.canvas_output.image = rec_img

        # Identify closest stored pattern
        if self.stored_patterns is not None and len(self.stored_patterns) > 0:
            dists = np.sum(self.stored_patterns != recalled, axis=1)
            idx = int(np.argmin(dists))
            label = self.labels[idx]
            energy_final = energies[-1] if len(energies) > 0 else None
            self.status.config(
                text=f"Recall complete. Closest stored pattern: {label} | Final energy: {energy_final:.2f}"
            )
        else:
            self.status.config(text="Recall complete.")


if __name__ == "__main__":
    root = tk.Tk()
    app = HopfieldGUI(root)
    root.mainloop()

