# Modernisasi UI TrainingUI
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import pandas as pd
import subprocess
import threading
import time
import os
import sys
from PIL import Image, ImageTk

class RedirectOutput:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')

    def flush(self):
        pass

class TrainingUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Sentiment Classifier")
        self.root.geometry("1300x700")
        self.root.configure(bg="#181818")
        self.code_lines = []
        self.current_highlight = None
        self.plot_images = []

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Treeview", background="#2e2e2e", foreground="white", fieldbackground="#2e2e2e", rowheight=25, font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.map("TButton", background=[("active", "#404040")], foreground=[("active", "#ffffff")])

        self.create_widgets()

    def create_widgets(self):
        self.main_frame = tk.Frame(self.root, bg="#181818")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(self.main_frame, width=450, bg="#181818")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        self.center_frame = tk.Frame(self.main_frame, bg="#181818")
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(self.main_frame, width=400, bg="#181818")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        self.csv_frame = tk.LabelFrame(self.left_frame, text="CSV Viewer", bg="#181818", fg="white", font=("Segoe UI", 10, "bold"))
        self.csv_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(self.csv_frame, textvariable=self.search_var)
        self.search_entry.pack(fill=tk.X, padx=5, pady=2)
        self.search_entry.bind("<KeyRelease>", self.filter_csv)

        self.tree_frame = tk.Frame(self.csv_frame)
        self.tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(self.tree_frame, columns=("text", "label"), show="headings")
        self.tree.heading("text", text="Text")
        self.tree.heading("label", text="Label")
        self.tree.column("label", width=50, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        self.tree_scroll = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.tree_scroll.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=self.tree_scroll.set)

        self.code_notebook = ttk.Notebook(self.center_frame)
        self.code_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.code_tab = tk.Frame(self.code_notebook, bg="#1e1e1e")
        self.plot_tab = tk.Frame(self.code_notebook, bg="#1e1e1e")

        self.code_notebook.add(self.code_tab, text="Code")
        self.code_notebook.add(self.plot_tab, text="Training Results")

        self.code_canvas = tk.Canvas(self.code_tab, bg="#1e1e1e")
        self.code_canvas.pack(side=tk.LEFT, fill="both", expand=True)
        self.code_scroll = ttk.Scrollbar(self.code_tab, orient="vertical", command=self.code_canvas.yview)
        self.code_scroll.pack(side="right", fill="y")
        self.code_canvas.configure(yscrollcommand=self.code_scroll.set)

        self.code_text = tk.Text(
            self.code_canvas, wrap="none", bg="#1e1e1e", fg="white",
            insertbackground="white", font=("Consolas", 11)
        )
        self.code_window = self.code_canvas.create_window((0, 0), window=self.code_text, anchor="nw")
        self.code_text.bind("<Configure>", lambda e: self.code_canvas.configure(scrollregion=self.code_canvas.bbox("all")))
        self.code_canvas.bind("<Configure>", self._resize_code_text)
        self.code_text.config(state="disabled")

        self.plot_scroll_canvas = tk.Canvas(self.plot_tab, bg="#1e1e1e")
        self.plot_scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.plot_scrollbar = ttk.Scrollbar(self.plot_tab, orient="vertical", command=self.plot_scroll_canvas.yview)
        self.plot_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.plot_scroll_canvas.configure(yscrollcommand=self.plot_scrollbar.set)

        self.plot_inner_frame = tk.Frame(self.plot_scroll_canvas, bg="#1e1e1e")
        self.plot_canvas_window = self.plot_scroll_canvas.create_window((0, 0), window=self.plot_inner_frame, anchor='nw')
        self.plot_inner_frame.bind("<Configure>", lambda e: self.plot_scroll_canvas.configure(scrollregion=self.plot_scroll_canvas.bbox("all")))

        self.terminal = tk.Text(
            self.right_frame, height=10, bg="#121212", fg="#00FF00",
            insertbackground='white', font=("Consolas", 11)
        )
        self.terminal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.btn_frame = tk.Frame(self.root, bg="#181818")
        self.btn_frame.pack(side=tk.BOTTOM, pady=10)

        self.start_btn = ttk.Button(self.btn_frame, text="Start Training", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.restart_btn = ttk.Button(self.btn_frame, text="Restart", command=self.restart_app)
        self.restart_btn.pack(side=tk.LEFT, padx=10)

        self.exit_btn = ttk.Button(self.btn_frame, text="Exit", command=self.exit_app)
        self.exit_btn.pack(side=tk.LEFT, padx=10)

        self.df_csv = None
        self.load_csv("dataset/movie.csv")
        self.load_code("main.py")

    def _resize_code_text(self, event):
        self.code_canvas.itemconfig(self.code_window, width=event.width, height=event.height)

    def filter_csv(self, event=None):
        keyword = self.search_var.get().lower()
        for i in self.tree.get_children():
            self.tree.delete(i)
        if self.df_csv is not None:
            for row in self.df_csv.itertuples(index=False):
                text = str(row.text)
                label = row.label
                if keyword in text.lower():
                    display_text = text[:100]
                    if keyword:
                        start_idx = display_text.lower().find(keyword)
                        if start_idx != -1:
                            end_idx = start_idx + len(keyword)
                            display_text = (
                                display_text[:start_idx] + '[[' + display_text[start_idx:end_idx] + ']]' + display_text[end_idx:]
                            )
                    display_text = display_text.replace('[[', '[43m').replace(']]', '[0m')  # dummy highlight in case terminal render
                    self.tree.insert('', tk.END, values=(text[:100].replace(keyword, keyword.upper()), label))

    def load_csv(self, path):
        try:
            self.df_csv = pd.read_csv(path)
            self.filter_csv()
        except Exception as e:
            self.terminal.insert(tk.END, f"Error loading CSV: {e}\n")

    def load_code(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.code_lines = f.readlines()
            self.code_text.config(state="normal")
            self.code_text.delete("1.0", tk.END)
            for line in self.code_lines:
                self.code_text.insert(tk.END, line)
            self.code_text.config(state="disabled")
        except Exception as e:
            self.terminal.insert(tk.END, f"Error loading code: {e}\n")

    def highlight_line(self, line_number):
        self.code_text.tag_remove("highlight", "1.0", tk.END)
        if line_number > 0 and line_number <= len(self.code_lines):
            line_index = f"{line_number}.0"
            self.code_text.tag_add("highlight", line_index, f"{line_number}.end")
            self.code_text.tag_config("highlight", background="yellow", foreground="black")

    def start_training(self):
        self.start_btn.config(state='disabled', text="Running...")
        self.terminal.config(state='normal')
        self.terminal.delete("1.0", tk.END)
        self.terminal.config(state='disabled')
        for widget in self.plot_inner_frame.winfo_children():
            widget.destroy()
        threading.Thread(target=self.run_main_script, daemon=True).start()

    def run_main_script(self):
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        line_idx = 0
        for line in process.stdout:
            self.terminal.config(state='normal')
            self.terminal.insert(tk.END, line)
            self.terminal.see(tk.END)
            self.terminal.config(state='disabled')
            self.highlight_line(line_idx)
            self.root.update()
            time.sleep(0.05)
            line_idx += 1
        process.wait()
        self.show_plot_images()
        self.start_btn.config(state='normal', text="Start Training")

    def show_plot_images(self):
        image_dir = "plots"
        y_offset = 10
        self.plot_images.clear()
        for fname in os.listdir(image_dir):
            if fname.endswith(".png"):
                try:
                    img_path = os.path.join(image_dir, fname)
                    img = Image.open(img_path)
                    img = img.resize((700, 350), Image.Resampling.LANCZOS)
                    tkimg = ImageTk.PhotoImage(img)
                    lbl = tk.Label(self.plot_inner_frame, image=tkimg, bg="#1e1e1e")
                    lbl.image = tkimg
                    lbl.pack(pady=10)
                    self.plot_images.append(tkimg)
                except Exception as e:
                    self.terminal.insert(tk.END, f"Error loading image {fname}: {e}\n")

    def restart_app(self):
        self.terminal.config(state='normal')
        self.terminal.delete("1.0", tk.END)
        self.terminal.config(state='disabled')
        for widget in self.plot_inner_frame.winfo_children():
            widget.destroy()
        self.code_text.config(state='normal')
        self.code_text.delete("1.0", tk.END)
        for line in self.code_lines:
            self.code_text.insert(tk.END, line)
        self.code_text.config(state='disabled')
        self.start_training()

    def exit_app(self):
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingUI(root)
    root.mainloop()
