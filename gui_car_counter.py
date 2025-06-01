import os
import sys
import uuid
import tempfile
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2

# Paths to detection scripts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_SCRIPT = os.path.join(BASE_DIR, "count_cars_generic.py")
VIDEO_SCRIPT = os.path.join(BASE_DIR, "count_cars_video_generic.py")

class CarCounterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Car Counter GUI")
        self.geometry("800x600")
        self.resizable(True, True)

        self.input_path = tk.StringVar()
        self.model_type = tk.StringVar()
        self.model_path = tk.StringVar()
        self.input_type = tk.StringVar(value="image")
        self._filetypes = [("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        self.output_image = None
        self.video_playback_label = None

        self.entry_mpath = None  # NEW: Reference to model path entry
        self.btn_browse_model = None  # NEW: Reference to browse button

        self._build_widgets()
        self._on_input_type_changed()

    def _build_widgets(self):
        padding = {"padx": 10, "pady": 5}

        lbl_input_type = ttk.Label(self, text="Input Type:")
        lbl_input_type.grid(row=0, column=0, sticky="w", **padding)

        input_type_combo = ttk.Combobox(
            self,
            textvariable=self.input_type,
            values=["image", "video"],
            state="readonly",
            width=20
        )
        input_type_combo.grid(row=0, column=1, sticky="w", **padding)
        input_type_combo.bind("<<ComboboxSelected>>", self._on_input_type_changed)

        self.lbl_input_file = ttk.Label(self, text="Input Image:")
        self.lbl_input_file.grid(row=1, column=0, sticky="w", **padding)

        entry_input = ttk.Entry(self, textvariable=self.input_path, width=50)
        entry_input.grid(row=1, column=1, sticky="ew", **padding)

        btn_browse_file = ttk.Button(self, text="Browse…", command=self._browse_input)
        btn_browse_file.grid(row=1, column=2, sticky="e", **padding)

        lbl_mtype = ttk.Label(self, text="Model Type:")
        lbl_mtype.grid(row=2, column=0, sticky="w", **padding)

        combobox_mtype = ttk.Combobox(
            self,
            textvariable=self.model_type,
            values=["yolo", "fasterrcnn-original", "fasterrcnn-finetuned"],
            width=25,
            state="readonly"
        )
        combobox_mtype.grid(row=2, column=1, sticky="w", **padding)
        combobox_mtype.bind("<<ComboboxSelected>>", self._on_model_type_changed)

        lbl_mpath = ttk.Label(self, text="Model File:")
        lbl_mpath.grid(row=3, column=0, sticky="w", **padding)

        self.entry_mpath = ttk.Entry(self, textvariable=self.model_path, width=50)
        self.entry_mpath.grid(row=3, column=1, sticky="ew", **padding)

        self.btn_browse_model = ttk.Button(self, text="Browse…", command=self._browse_model)
        self.btn_browse_model.grid(row=3, column=2, sticky="e", **padding)

        btn_run = ttk.Button(self, text="Run Detection", command=self._on_run_clicked)
        btn_run.grid(row=4, column=0, columnspan=3, pady=(15, 10))

        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)
        self.rowconfigure(5, weight=1)
        self.columnconfigure(1, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#eee")
        self.canvas.pack(fill="both", expand=True)

    def _on_input_type_changed(self, event=None):
        t = self.input_type.get()
        if t == "image":
            self._filetypes = [("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
            self.lbl_input_file.config(text="Input Image:")
        else:
            self._filetypes = [("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
            self.lbl_input_file.config(text="Input Video:")

    def _on_model_type_changed(self, event=None):
        mtype = self.model_type.get()
        if mtype == "yolo":
            self.model_path.set("yolov8l.pt")
        elif mtype == "fasterrcnn-finetuned":
            self.model_path.set("models/fasterrcnn_car.pth")
        elif mtype == "fasterrcnn-original":
            self.model_path.set("")

        if mtype == "fasterrcnn-original":
            self.entry_mpath.configure(state="disabled")
            self.btn_browse_model.configure(state="disabled")
        else:
            self.entry_mpath.configure(state="normal")
            self.btn_browse_model.configure(state="normal")

    def _browse_input(self):
        path = filedialog.askopenfilename(title="Select Input File", filetypes=self._filetypes)
        if path:
            self.input_path.set(path)

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch models", "*.pt *.pth"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)

    def _on_run_clicked(self):
        input_type = self.input_type.get().strip()
        input_path = self.input_path.get().strip()
        mtype = self.model_type.get().strip()
        mpath = self.model_path.get().strip()

        if not input_path or not os.path.isfile(input_path):
            messagebox.showerror("Error", f"Please select a valid {input_type} file.")
            return

        if not mtype:
            messagebox.showerror("Error", "Please select a model type.")
            return

        if mtype != "fasterrcnn-original" and (not mpath or not os.path.isfile(mpath)):
            messagebox.showerror("Error", "Please select a valid model file.")
            return

        script_path = IMAGE_SCRIPT if input_type == "image" else VIDEO_SCRIPT
        if not os.path.isfile(script_path):
            messagebox.showerror("Error", f"Detection script not found:\n{script_path}")
            return

        self._set_widgets_state("disabled")
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            text="Processing...",
            fill="gray",
            font=("TkDefaultFont", 16)
        )

        threading.Thread(
            target=self._run_detection,
            args=(input_type, input_path, mtype, mpath),
            daemon=True
        ).start()

    def _set_widgets_state(self, state):
        for child in self.winfo_children():
            try:
                child.configure(state=state)
            except:
                pass

    def _run_detection(self, input_type, input_path, model_type, model_path):
        script_path = IMAGE_SCRIPT if input_type == "image" else VIDEO_SCRIPT
        ext = ".mp4" if input_type == "video" else ".jpg"
        output_path = os.path.join(tempfile.gettempdir(), f"car_count_{uuid.uuid4().hex[:8]}{ext}")

        flag = "--video-path" if input_type == "video" else "--image-path"
        cmd = [
            sys.executable,
            script_path,
            "--model-type", model_type,
            flag, input_path,
            "--output-path", output_path
        ]
        if model_type != "fasterrcnn-original":
            cmd += ["--model-path", model_path]

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.after(0, lambda: self._set_widgets_state("normal"))
            return

        if result.returncode != 0:
            self.after(0, lambda: messagebox.showerror("Detection Failed", result.stderr))
            self.after(0, lambda: self._set_widgets_state("normal"))
            return

        if input_type == "video":
            self.after(0, lambda: self._play_video_on_canvas(output_path))
            return

        try:
            img = Image.open(output_path)
            canvas_w = self.canvas_frame.winfo_width()
            canvas_h = self.canvas_frame.winfo_height()
            img_ratio = img.width / img.height
            canvas_ratio = canvas_w / canvas_h

            if img_ratio > canvas_ratio:
                new_w = canvas_w
                new_h = int(canvas_w / img_ratio)
            else:
                new_h = canvas_h
                new_w = int(canvas_h * img_ratio)

            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = Image.ANTIALIAS  # fallback for older versions

            img_resized = img.resize((new_w, new_h), resample_filter)
            #img_resized = img.resize((new_w, new_h), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(img_resized)

            def show_image():
                self.canvas.delete("all")
                x = (canvas_w - new_w) // 2
                y = (canvas_h - new_h) // 2
                self.canvas.create_image(x, y, anchor="nw", image=photo)
                self.output_image = photo
                self._set_widgets_state("normal")

            self.after(0, show_image)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error displaying image", str(e)))
            self.after(0, lambda: self._set_widgets_state("normal"))

    def _play_video_on_canvas(self, video_path):
        self.canvas.delete("all")
        if self.video_playback_label:
            self.video_playback_label.destroy()

        self.video_playback_label = ttk.Label(self.canvas_frame)
        self.video_playback_label.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.update()
        cap = cv2.VideoCapture(video_path)

        def update_frame():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                self.video_playback_label.destroy()
                self._set_widgets_state("normal")
                return

            frame_w = self.canvas_frame.winfo_width()
            frame_h = self.canvas_frame.winfo_height()
            frame = cv2.resize(frame, (frame_w, frame_h))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.video_playback_label.imgtk = img_tk
            self.video_playback_label.configure(image=img_tk)
            self.after(30, update_frame)

        self.after(0, update_frame)

if __name__ == "__main__":
    app = CarCounterGUI()
    app.mainloop()
