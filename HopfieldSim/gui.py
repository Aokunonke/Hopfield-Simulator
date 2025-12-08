import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

from hopfield_model import HopfieldNetwork, IMG_SIZE, PREVIEW_SIZE
from image_utils import image_to_pattern, pattern_to_image, load_image_as_array, hamming_distance


class HopfieldImageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hopfield Image Recall")

        self.patterns = None
        self.labels = []
        self.net = None
        self.test_pattern = None

        self.learning_rule = tk.StringVar(value="hebb")

        self._build_ui()

    def _build_ui(self):
        main = tk.Frame(self.root)
        main.pack(padx=20, pady=20)

        tk.Label(main, text="Hopfield Image Recall", font=("Arial", 16, "bold")).pack()

        # -------- Training --------
        train_frame = tk.LabelFrame(main, text="Training")
        train_frame.pack(fill="x", pady=10)

        self.train_files_label = tk.Label(train_frame, text="No training images selected.")
        self.train_files_label.pack(anchor="w")

        tk.Button(train_frame, text="Select Training Images",
                  command=self.select_training_images).pack(anchor="w")

        rule_frame = tk.Frame(train_frame)
        rule_frame.pack(anchor="w", pady=4)

        tk.Radiobutton(rule_frame, text="Hebb", variable=self.learning_rule,
                       value="hebb").pack(side="left")
        tk.Radiobutton(rule_frame, text="Pseudo-inverse",
                       variable=self.learning_rule, value="pseudo").pack(side="left")

        self.btn_train = tk.Button(train_frame, text="Train Network",
                                   command=self.train_network)
        self.btn_train.pack(anchor="w", pady=5)

        self.train_status = tk.Label(train_frame, text="")
        self.train_status.pack(anchor="w")

        # -------- Recall --------
        recall_frame = tk.LabelFrame(main, text="Recall")
        recall_frame.pack(fill="x", pady=10)

        self.test_file_label = tk.Label(recall_frame, text="No test image selected.")
        self.test_file_label.pack(anchor="w")

        tk.Button(recall_frame, text="Select Test Image",
                  command=self.select_test_image).pack(anchor="w")

        tk.Label(recall_frame, text="Noise %").pack(anchor="w")
        self.noise_scale = tk.Scale(recall_frame, from_=0, to=60,
                                    orient=tk.HORIZONTAL)
        self.noise_scale.pack(anchor="w")

        self.btn_recall = tk.Button(recall_frame, text="Recall",
                                    command=self.recall_and_check,
                                    state=tk.DISABLED)
        self.btn_recall.pack(anchor="w", pady=5)

        # -------- Images --------
        img_frame = tk.Frame(main)
        img_frame.pack()

        self.noisy_label = tk.Label(img_frame)
        self.noisy_label.pack(side="left", padx=10)

        self.rec_label = tk.Label(img_frame)
        self.rec_label.pack(side="left", padx=10)

        self.status_bar = tk.Label(main, text="")
        self.status_bar.pack(anchor="w")

    # -------- Controller --------

    def select_training_images(self):
        files = filedialog.askopenfilenames()
        if not files:
            return

        patterns = []
        labels = []

        for f in files:
            arr = load_image_as_array(f)
            patterns.append(image_to_pattern(arr))
            labels.append(os.path.basename(f))

        self.patterns = np.array(patterns)
        self.labels = labels
        self.train_files_label.config(text=f"{len(files)} images selected")

    def train_network(self):
        try:
            n_units = self.patterns.shape[1]
            self.net = HopfieldNetwork(n_units)

            if self.learning_rule.get() == "pseudo":
                self.net.train_pseudo_inverse(self.patterns)
            else:
                self.net.train_hebb(self.patterns)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.train_status.config(text="Training completed")
        self.btn_recall.config(state=tk.NORMAL)

    def select_test_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        arr = load_image_as_array(path)
        self.test_pattern = image_to_pattern(arr)

        img_arr = pattern_to_image(self.test_pattern, IMG_SIZE)
        self._update_image_label(self.noisy_label, img_arr)

        self.test_file_label.config(text=os.path.basename(path))

    def recall_and_check(self):
        if self.net is None or self.test_pattern is None:
            return

        pattern = self.test_pattern.copy()
        n_units = pattern.size
        flip_prob = self.noise_scale.get() / 100.0

        mask = np.random.rand(n_units) < flip_prob
        pattern[mask] *= -1.0

        noisy_arr = pattern_to_image(pattern, IMG_SIZE)
        self._update_image_label(self.noisy_label, noisy_arr)

        recalled, _ = self.net.recall(pattern)
        rec_img = pattern_to_image(recalled, IMG_SIZE)
        self._update_image_label(self.rec_label, rec_img)

        dists = [hamming_distance(recalled, p) for p in self.patterns]
        idx = int(np.argmin(dists))

        self.status_bar.config(
            text=f"Recognized as: {self.labels[idx]} | Hamming distance: {dists[idx]}"
        )

    def _update_image_label(self, label_widget, img_array):
        img = Image.fromarray(img_array, mode="L").resize(PREVIEW_SIZE)
        photo = ImageTk.PhotoImage(img)
        label_widget.config(image=photo)
        label_widget.image = photo