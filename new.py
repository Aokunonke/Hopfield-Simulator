import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np


# ----------------- Model: Hopfield network and helpers -----------------

IMG_SIZE = (32, 32)       # internal pattern size
PREVIEW_SIZE = (224, 224) # on-screen display size


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
        """Classic Hebbian learning."""
        patterns = np.asarray(patterns, dtype=float)
        n_patterns, n_units = patterns.shape
        if n_units != self.n_units:
            raise ValueError("Pattern size does not match network size.")

        self.W = (patterns.T @ patterns) / n_patterns
        np.fill_diagonal(self.W, 0.0)

    def train_pseudo_inverse(self, patterns: np.ndarray):
        "Pseudo-inverse learning rule."
        patterns = np.asarray(patterns, dtype=float)
        P, N = patterns.shape
        if N != self.n_units:
            raise ValueError("Pattern size does not match network size.")

        X = patterns             # P x N
        G = X @ X.T              # P x P
        G_inv = np.linalg.pinv(G)
        self.W = X.T @ G_inv @ X # N x N
        np.fill_diagonal(self.W, 0.0)

    def energy(self, state: np.ndarray) -> float:
        state = state.reshape(-1, 1).astype(float)
        return float(-0.5 * state.T @ self.W @ state)

    def recall(self, pattern: np.ndarray, max_iters: int = 100):
        """
        Recall from an initial pattern (synchronous update).
        pattern: 1D array length n_units, values -1/+1.
        """
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


def image_to_pattern(img_array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert grayscale image array to Hopfield pattern (-1/+1)."""
    img = img_array.astype(float) / 255.0
    binary = np.where(img >= threshold, 1.0, -1.0)
    return binary.flatten()


def pattern_to_image(pattern: np.ndarray, shape) -> np.ndarray:
    """Convert 1D pattern (-1/+1) back to 2D 0-255 image."""
    pattern = np.asarray(pattern, dtype=float).flatten()
    img = pattern.reshape(shape)
    img = (img + 1.0) / 2.0  # -1 -> 0, +1 -> 1
    img = img * 255.0
    return img.astype("uint8")


def load_image_as_array(path: str) -> np.ndarray:
    img = Image.open(path).convert("L").resize(IMG_SIZE)
    return np.array(img, dtype=float)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return int(np.sum(a != b))


# UI

class HopfieldImageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hopfield Image Recall")

        # Data state
        self.patterns = None
        self.stored_patterns = None
        self.labels = []
        self.net = None
        self.test_pattern = None

        # Learning rule selector
        self.learning_rule = tk.StringVar(value="hebb")

        self._build_ui()

    def _build_ui(self):
        root = self.root
        root.configure(bg="#ffffff")

        main = tk.Frame(root, bg="#ffffff", bd=0)
        main.pack(padx=16, pady=16, fill=tk.BOTH, expand=True)

        # Title
        title = tk.Label(
            main,
            text="Hopfield Image Recall",
            bg="#ffffff",
            fg="#111827",
            font=("Segoe UI", 16, "bold")
        )
        title.pack(pady=(4, 0))

        subtitle = tk.Label(
            main,
            text="Train on images, add noise, and see what the network recalls.",
            bg="#ffffff",
            fg="#4b5563",
            font=("Segoe UI", 10)
        )
        subtitle.pack(pady=(0, 12))

        # Training section
        train_frame = tk.LabelFrame(
            main,
            text="1. Training images",
            bg="#ffffff",
            fg="#111827",
            bd=1,
            relief=tk.SOLID,
            font=("Segoe UI", 9, "bold")
        )
        train_frame.pack(fill=tk.X, padx=40, pady=(4, 8))

        train_inner = tk.Frame(train_frame, bg="#ffffff")
        train_inner.pack(fill=tk.X, padx=8, pady=8)

        # Left: file selector
        train_left = tk.Frame(train_inner, bg="#ffffff")
        train_left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.train_files_label = tk.Label(
            train_left,
            text="No training images selected.",
            bg="#ffffff",
            fg="#4b5563",
            font=("Segoe UI", 9)
        )
        self.train_files_label.pack(anchor="w")

        btn_select_train = tk.Button(
            train_left,
            text="Select training images",
            command=self.select_training_images,
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            relief=tk.FLAT,
            font=("Segoe UI", 9),
            padx=10,
            pady=4
        )
        btn_select_train.pack(anchor="w", pady=(4, 0))

        # Right: learning rule + train button
        train_right = tk.Frame(train_inner, bg="#ffffff")
        train_right.pack(side=tk.LEFT, fill=tk.X, expand=True)

        rule_frame = tk.Frame(train_right, bg="#ffffff")
        rule_frame.pack(anchor="e")

        tk.Label(
            rule_frame,
            text="Learning rule:",
            bg="#ffffff",
            fg="#111827",
            font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=(0, 4))

        tk.Radiobutton(
            rule_frame,
            text="Hebb",
            variable=self.learning_rule,
            value="hebb",
            bg="#ffffff",
            fg="#111827",
            selectcolor="#e5e7eb",
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT)

        tk.Radiobutton(
            rule_frame,
            text="Pseudo-inverse",
            variable=self.learning_rule,
            value="pseudo",
            bg="#ffffff",
            fg="#111827",
            selectcolor="#e5e7eb",
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=(4, 0))

        self.btn_train = tk.Button(
            train_right,
            text="Train network",
            command=self.train_network,
            bg="#10b981",
            fg="white",
            activebackground="#059669",
            relief=tk.FLAT,
            font=("Segoe UI", 9),
            padx=10,
            pady=4
        )
        self.btn_train.pack(anchor="e", pady=(6, 4))

        self.train_status = tk.Label(
            train_right,
            text="",
            bg="#ffffff",
            fg="#4b5563",
            font=("Segoe UI", 8)
        )
        self.train_status.pack(anchor="e")

        # Recall section
        access_frame = tk.LabelFrame(
            main,
            text="2. Recall from noisy input",
            bg="#ffffff",
            fg="#111827",
            bd=1,
            relief=tk.SOLID,
            font=("Segoe UI", 9, "bold")
        )
        access_frame.pack(fill=tk.BOTH, padx=40, pady=(4, 8), expand=True)

        access_inner = tk.Frame(access_frame, bg="#ffffff")
        access_inner.pack(fill=tk.BOTH, padx=8, pady=12, expand=True)

        # Left column: controls
        control_frame = tk.Frame(access_inner, bg="#ffffff")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 16))

        tk.Label(
            control_frame,
            text="Test image:",
            bg="#ffffff",
            fg="#111827",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w")

        self.test_file_label = tk.Label(
            control_frame,
            text="No test image selected.",
            bg="#ffffff",
            fg="#4b5563",
            font=("Segoe UI", 9)
        )
        self.test_file_label.pack(anchor="w")

        btn_select_test = tk.Button(
            control_frame,
            text="Select test image",
            command=self.select_test_image,
            bg="#6b7280",
            fg="white",
            activebackground="#4b5563",
            relief=tk.FLAT,
            font=("Segoe UI", 9),
            padx=10,
            pady=4
        )
        btn_select_test.pack(anchor="w", pady=(4, 12))

        tk.Label(
            control_frame,
            text="Noise level (% bits flipped):",
            bg="#ffffff",
            fg="#111827",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w")

        self.noise_scale = tk.Scale(
            control_frame,
            from_=0,
            to=60,
            orient=tk.HORIZONTAL,
            length=200,
            bg="#ffffff",
            fg="#111827",
            troughcolor="#e5e7eb",
            highlightthickness=0
        )
        self.noise_scale.set(0)
        self.noise_scale.pack(anchor="w", pady=(2, 8))

        self.btn_recall = tk.Button(
            control_frame,
            text="Recall",
            command=self.recall_and_check,
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=4,
            state=tk.DISABLED
        )
        self.btn_recall.pack(anchor="w", pady=(4, 4))

        # Right: images (centered, bigger)
        img_panel = tk.Frame(access_inner, bg="#ffffff")
        img_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        noisy_frame = tk.Frame(img_panel, bg="#ffffff")
        noisy_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            noisy_frame,
            text="Input (after noise)",
            bg="#ffffff",
            fg="#111827",
            font=("Segoe UI", 9, "bold")
        ).pack()

        self.noisy_label = tk.Label(noisy_frame, bg="#ffffff")
        self.noisy_label.pack(pady=(6, 0))

        rec_frame = tk.Frame(img_panel, bg="#ffffff")
        rec_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            rec_frame,
            text="Recalled pattern",
            bg="#ffffff",
            fg="#111827",
            font=("Segoe UI", 9, "bold")
        ).pack()

        self.rec_label = tk.Label(rec_frame, bg="#ffffff")
        self.rec_label.pack(pady=(6, 0))

        # Status bar
        self.status_bar = tk.Label(
            main,
            text="",
            bg="#ffffff",
            fg="#4b5563",
            font=("Segoe UI", 8),
            anchor="w"
        )
        self.status_bar.pack(fill=tk.X, padx=40, pady=(4, 0))

    # ----- Controller logic -----

    def select_training_images(self):
        paths = filedialog.askopenfilenames(
            title="Select training images",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not paths:
            return

        patterns = []
        labels = []

        for path in paths:
            arr = load_image_as_array(path)
            pattern = image_to_pattern(arr, threshold=0.5)
            patterns.append(pattern)
            labels.append(os.path.basename(path))

        self.patterns = np.vstack(patterns)
        self.stored_patterns = self.patterns.copy()
        self.labels = labels

        self.train_files_label.config(
            text=f"{len(paths)} training images selected."
        )
        self.train_status.config(text="")
        self.status_bar.config(text="")

    def train_network(self):
        if self.patterns is None:
            messagebox.showerror("Error", "Select training images first.")
            return

        try:
            n_units = self.patterns.shape[1]
            self.net = HopfieldNetwork(n_units)

            if self.learning_rule.get() == "pseudo":
                self.net.train_pseudo_inverse(self.patterns)
                rule_name = "Pseudo-inverse"
            else:
                self.net.train_hebb(self.patterns)
                rule_name = "Hebbian"

        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {e}")
            return

        self.train_status.config(text=f"Network trained ({rule_name}).")
        self.status_bar.config(
            text=f"Stored labels: {', '.join(self.labels)}"
        )
        self.btn_recall.config(state=tk.NORMAL)

    def select_test_image(self):
        path = filedialog.askopenfilename(
            title="Select test image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not path:
            return

        arr = load_image_as_array(path)
        self.test_pattern = image_to_pattern(arr, threshold=0.5)
        self.test_file_label.config(text=os.path.basename(path))
        self.status_bar.config(text="Test image loaded. Adjust noise, then recall.")

        img_arr = pattern_to_image(self.test_pattern, IMG_SIZE)
        self._update_image_label(self.noisy_label, img_arr)

    def recall_and_check(self):
        if self.net is None or self.stored_patterns is None:
            messagebox.showerror("Error", "Train the network first.")
            return
        if self.test_pattern is None:
            messagebox.showerror("Error", "Select a test image first.")
            return

        pattern = self.test_pattern.copy()
        n_units = pattern.size

        # Apply noise
        noise_level = self.noise_scale.get()
        if noise_level > 0:
            flip_prob = noise_level / 100.0
            mask = np.random.rand(n_units) < flip_prob
            pattern[mask] *= -1.0

        noisy_arr = pattern_to_image(pattern, IMG_SIZE)
        self._update_image_label(self.noisy_label, noisy_arr)

        # Recall
        try:
            recalled, energies = self.net.recall(pattern, max_iters=100)
        except Exception as e:
            messagebox.showerror("Error", f"Recall failed: {e}")
            return

        rec_arr = pattern_to_image(recalled, IMG_SIZE)
        self._update_image_label(self.rec_label, rec_arr)

        # Compare to stored patterns
        dists = [hamming_distance(recalled, p) for p in self.stored_patterns]
        closest_idx = int(np.argmin(dists))
        closest_label = self.labels[closest_idx]
        closest_dist = dists[closest_idx]

        final_energy = energies[-1] if energies else None

        info = [
            f"Closest label: {closest_label}",
            f"Hamming distance: {closest_dist}"
        ]
        

        self.status_bar.config(text=" | ".join(info))

    def _update_image_label(self, label_widget: tk.Label, img_array: np.ndarray):
        img = Image.fromarray(img_array, mode="L").resize(PREVIEW_SIZE)
        photo = ImageTk.PhotoImage(img)
        label_widget.config(image=photo)
        label_widget.image = photo  # keep reference


if __name__ == "__main__":
    root = tk.Tk()
    app = HopfieldImageGUI(root)
    root.mainloop()
