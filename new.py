import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np


# ----------------- Model: Hopfield network and helpers -----------------

IMG_SIZE = (64, 64)  # all images resized to this


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

        # Remove self-connections
        np.fill_diagonal(self.W, 0.0)

    def energy(self, state: np.ndarray) -> float:
        state = state.reshape(-1, 1).astype(float)
        return float(-0.5 * state.T @ self.W @ state)

    def recall(self, pattern: np.ndarray, max_iters: int = 100):
        """
        Recall from an initial pattern (synchronous update).
        pattern: 1D array length n_units, values -1/+1.
        Returns (final_state, energy_list).
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
    """
    Convert grayscale image array to a Hopfield pattern (-1/+1).
    img_array: 2D numpy array, 0-255.
    threshold in [0,1] applied after normalization.
    """
    img = img_array.astype(float)
    img = img / 255.0
    binary = np.where(img >= threshold, 1.0, -1.0)
    return binary.flatten()


def pattern_to_image(pattern: np.ndarray, shape) -> np.ndarray:
    """
    Convert 1D pattern (-1/+1) back to 2D image 0-255.
    shape: (H, W).
    """
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


# ----------------- View/Controller: Tkinter GUI -----------------

class HopfieldAccessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hopfield Image Access Door")

        # Data state
        self.patterns = None          # training patterns (N x n_units)
        self.stored_patterns = None   # copy of patterns
        self.labels = []              # filenames of training images
        self.net = None               # HopfieldNetwork
        self.test_pattern = None      # pattern from test image (no noise)

        # UI
        self._build_ui()

    

    def _build_ui(self):
        root = self.root
        root.configure(bg="#0f172a")

        main = tk.Frame(root, bg="#020617", bd=1, relief=tk.SOLID)
        main.pack(padx=16, pady=16, fill=tk.BOTH, expand=True)

        title = tk.Label(
            main,
            text="Hopfield Image Access Door",
            bg="#020617",
            fg="#e5e7eb",
            font=("Segoe UI", 14, "bold")
        )
        title.pack(anchor="w", padx=12, pady=(12, 2))

        subtitle = tk.Label(
            main,
            text="Train on authorized images, then test with a noisy image. Door opens only if the pattern matches.",
            bg="#020617",
            fg="#9ca3af",
            font=("Segoe UI", 9)
        )
        subtitle.pack(anchor="w", padx=12, pady=(0, 8))

        # Training section
        train_frame = tk.LabelFrame(
            main,
            text="1. Training (Authorized Images)",
            bg="#020617",
            fg="#e5e7eb",
            bd=1,
            relief=tk.SOLID,
            font=("Segoe UI", 9, "bold")
        )
        train_frame.pack(fill=tk.X, padx=12, pady=(8, 4))

        train_inner = tk.Frame(train_frame, bg="#020617")
        train_inner.pack(fill=tk.X, padx=8, pady=8)

        # Left: file selector
        train_left = tk.Frame(train_inner, bg="#020617")
        train_left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.train_files_label = tk.Label(
            train_left,
            text="No training images selected.",
            bg="#020617",
            fg="#9ca3af",
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

        # Right: train button
        train_right = tk.Frame(train_inner, bg="#020617")
        train_right.pack(side=tk.LEFT, fill=tk.X, expand=True)

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
        self.btn_train.pack(anchor="e", pady=(0, 4))

        self.train_status = tk.Label(
            train_right,
            text="",
            bg="#020617",
            fg="#9ca3af",
            font=("Segoe UI", 8)
        )
        self.train_status.pack(anchor="e")

        # Access attempt section
        access_frame = tk.LabelFrame(
            main,
            text="2. Access Attempt",
            bg="#020617",
            fg="#e5e7eb",
            bd=1,
            relief=tk.SOLID,
            font=("Segoe UI", 9, "bold")
        )
        access_frame.pack(fill=tk.BOTH, padx=12, pady=(8, 8), expand=True)

        access_inner = tk.Frame(access_frame, bg="#020617")
        access_inner.pack(fill=tk.BOTH, padx=8, pady=8, expand=True)

        # Left column: controls
        control_frame = tk.Frame(access_inner, bg="#020617")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            control_frame,
            text="Test image:",
            bg="#020617",
            fg="#e5e7eb",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w")

        self.test_file_label = tk.Label(
            control_frame,
            text="No test image selected.",
            bg="#020617",
            fg="#9ca3af",
            font=("Segoe UI", 9)
        )
        self.test_file_label.pack(anchor="w")

        btn_select_test = tk.Button(
            control_frame,
            text="Select test image",
            command=self.select_test_image,
            bg="#4b5563",
            fg="white",
            activebackground="#374151",
            relief=tk.FLAT,
            font=("Segoe UI", 9),
            padx=10,
            pady=4
        )
        btn_select_test.pack(anchor="w", pady=(4, 8))

        # Noise slider
        tk.Label(
            control_frame,
            text="Noise level (% bits flipped):",
            bg="#020617",
            fg="#e5e7eb",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", pady=(4, 0))

        self.noise_scale = tk.Scale(
            control_frame,
            from_=0,
            to=60,
            orient=tk.HORIZONTAL,
            length=200,
            bg="#020617",
            fg="#e5e7eb",
            troughcolor="#111827",
            highlightthickness=0
        )
        self.noise_scale.set(0)
        self.noise_scale.pack(anchor="w")

        # Threshold slider
        tk.Label(
            control_frame,
            text="Match threshold (% allowed difference):",
            bg="#020617",
            fg="#e5e7eb",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", pady=(4, 0))

        self.threshold_scale = tk.Scale(
            control_frame,
            from_=0,
            to=40,
            orient=tk.HORIZONTAL,
            length=200,
            bg="#020617",
            fg="#e5e7eb",
            troughcolor="#111827",
            highlightthickness=0
        )
        self.threshold_scale.set(10)
        self.threshold_scale.pack(anchor="w")

        self.btn_recall = tk.Button(
            control_frame,
            text="Recall and check access",
            command=self.recall_and_check,
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            relief=tk.FLAT,
            font=("Segoe UI", 9),
            padx=10,
            pady=4,
            state=tk.DISABLED
        )
        self.btn_recall.pack(anchor="w", pady=(8, 4))

        # Right column: door + images
        right_panel = tk.Frame(access_inner, bg="#020617")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(16, 0))

        # Door panel
        door_frame = tk.Frame(right_panel, bg="#020617")
        door_frame.pack(fill=tk.X)

        tk.Label(
            door_frame,
            text="Access door:",
            bg="#020617",
            fg="#e5e7eb",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w")

        self.door_canvas = tk.Canvas(
            door_frame,
            width=160,
            height=220,
            bg="#020617",
            highlightthickness=0
        )
        self.door_canvas.pack(anchor="w", pady=(4, 4))

        # Draw door rectangle
        self.door_rect = self.door_canvas.create_rectangle(
            30, 20, 130, 200,
            fill="#111827",
            outline="#ef4444",
            width=3
        )
        # Door knob
        self.door_knob = self.door_canvas.create_oval(
            110, 105, 120, 115,
            fill="#ef4444",
            outline=""
        )
        # Status text
        self.door_text = self.door_canvas.create_text(
            80, 210,
            text="ACCESS DENIED",
            fill="#fecaca",
            font=("Segoe UI", 9, "bold")
        )

        # Images panel
        img_panel = tk.Frame(right_panel, bg="#020617")
        img_panel.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        # Noisy input image
        noisy_frame = tk.Frame(img_panel, bg="#020617")
        noisy_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            noisy_frame,
            text="Input (after noise)",
            bg="#020617",
            fg="#e5e7eb",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w")

        self.noisy_label = tk.Label(noisy_frame, bg="#020617")
        self.noisy_label.pack(pady=(4, 0))

        # Recalled image
        rec_frame = tk.Frame(img_panel, bg="#020617")
        rec_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            rec_frame,
            text="Recalled pattern",
            bg="#020617",
            fg="#e5e7eb",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w")

        self.rec_label = tk.Label(rec_frame, bg="#020617")
        self.rec_label.pack(pady=(4, 0))

        # Status bar
        self.status_bar = tk.Label(
            main,
            text="",
            bg="#020617",
            fg="#9ca3af",
            font=("Segoe UI", 8),
            anchor="w"
        )
        self.status_bar.pack(fill=tk.X, padx=12, pady=(8, 8))

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
            self.net.train_hebb(self.patterns)
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {e}")
            return

        self.train_status.config(text="Network trained.")
        self.status_bar.config(
            text=f"Authorized labels: {', '.join(self.labels)}"
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
        self.status_bar.config(text="Test image loaded. Adjust noise/threshold, then recall.")

        # Show original (no noise) as noisy panel for now
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

        # Show noisy input
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

        # Threshold decision
        threshold_percent = self.threshold_scale.get()
        max_allowed_diff = int((threshold_percent / 100.0) * n_units)
        access_granted = closest_dist <= max_allowed_diff

        # Final energy
        final_energy = energies[-1] if energies else None

        info = [
            f"Closest label: {closest_label}",
            f"Hamming distance: {closest_dist}",
            f"Threshold: {threshold_percent}% ({max_allowed_diff} pixels)",
            f"Access: {'GRANTED' if access_granted else 'DENIED'}"
        ]
        if final_energy is not None:
            info.append(f"Final energy: {final_energy:.2f}")

        self.status_bar.config(text=" | ".join(info))
        self._set_door_state(access_granted)

    

    def _update_image_label(self, label_widget: tk.Label, img_array: np.ndarray):
        img = Image.fromarray(img_array, mode="L").resize((128, 128))
        photo = ImageTk.PhotoImage(img)
        label_widget.config(image=photo)
        label_widget.image = photo  # keep reference

    def _set_door_state(self, access_granted: bool):
        if access_granted:
            # Door open / granted
            self.door_canvas.itemconfig(
                self.door_rect,
                outline="#22c55e",
                fill="#111827"
            )
            self.door_canvas.itemconfig(
                self.door_knob,
                fill="#bbf7d0"
            )
            self.door_canvas.itemconfig(
                self.door_text,
                text="ACCESS GRANTED",
                fill="#bbf7d0"
            )
        else:
            # Door closed / denied
            self.door_canvas.itemconfig(
                self.door_rect,
                outline="#ef4444",
                fill="#111827"
            )
            self.door_canvas.itemconfig(
                self.door_knob,
                fill="#ef4444"
            )
            self.door_canvas.itemconfig(
                self.door_text,
                text="ACCESS DENIED",
                fill="#fecaca"
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = HopfieldAccessGUI(root)
    root.mainloop()
      