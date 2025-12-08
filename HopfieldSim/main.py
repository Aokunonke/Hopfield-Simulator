import tkinter as tk
from gui import HopfieldImageGUI

if __name__ == "__main__":
    root = tk.Tk()
    app = HopfieldImageGUI(root)
    root.mainloop()