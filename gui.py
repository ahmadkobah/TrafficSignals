"""
Traffic Signal Classifier - Complete GUI
Training and Prediction with original dark style
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from classifier import TrafficSignalClassifier
from cnn_classifier import CNNClassifier


class TrafficSignalGUI:
    """Complete GUI for Traffic Signal Classification"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Traffic Signal Classifier")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Colors - Original dark style
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.accent_color = "#4a9eff"
        self.button_bg = "#3c3c3c"
        self.success_color = "#4CAF50"
        self.warning_color = "#ff9800"
        
        self.root.configure(bg=self.bg_color)
        
        # Style configuration
        self.setup_styles()
        
        # Classifiers
        self.knn_svm = None
        self.cnn = None
        self.selected_image = None
        self.training_thread = None
        
        # Build GUI
        self.setup_gui()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def setup_styles(self):
        """Setup ttk styles for dark theme"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure dark theme
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color, font=('Segoe UI', 10))
        style.configure("TButton", background=self.button_bg, foreground=self.fg_color, font=('Segoe UI', 10), padding=8)
        style.configure("TLabelframe", background=self.bg_color, foreground=self.fg_color)
        style.configure("TLabelframe.Label", background=self.bg_color, foreground=self.accent_color, font=('Segoe UI', 11, 'bold'))
        style.configure("TNotebook", background=self.bg_color)
        style.configure("TNotebook.Tab", background=self.button_bg, foreground=self.fg_color, padding=[15, 5])
        style.configure("TEntry", fieldbackground="#1e1e1e", foreground=self.fg_color)
        style.configure("TProgressbar", background=self.accent_color, troughcolor=self.button_bg)
        
        # Map for hover effects
        style.map("TButton",
                  background=[('active', self.accent_color)],
                  foreground=[('active', 'white')])
        style.map("TNotebook.Tab",
                  background=[('selected', self.accent_color)])
    
    def setup_gui(self):
        """Setup the main GUI"""
        # Main container
        main = ttk.Frame(self.root, padding=15)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_frame = ttk.Frame(main)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(title_frame, text="Traffic Signal Classifier", 
                               font=('Segoe UI', 20, 'bold'),
                               bg=self.bg_color, fg=self.accent_color)
        title_label.pack()
        
        subtitle = tk.Label(title_frame, text="KNN | SVM | CNN | HOG Features",
                           font=('Segoe UI', 10),
                           bg=self.bg_color, fg="#888888")
        subtitle.pack()
        
        # Notebook (tabs)
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Tab 1: Training
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="  Training  ")
        self.setup_training_tab()
        
        # Tab 2: Prediction
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text="  Prediction  ")
        self.setup_prediction_tab()
        
        # Bottom buttons
        bottom_frame = ttk.Frame(main)
        bottom_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Status label
        self.status_label = tk.Label(bottom_frame, text="Ready", 
                                     font=('Segoe UI', 9),
                                     bg=self.bg_color, fg="#888888")
        self.status_label.pack(side=tk.LEFT)
        
        # Exit button
        exit_btn = tk.Button(bottom_frame, text="Exit", 
                            font=('Segoe UI', 10, 'bold'),
                            bg="#dc3545", fg="white",
                            activebackground="#c82333",
                            relief=tk.FLAT, padx=20, pady=5,
                            command=self.on_close)
        exit_btn.pack(side=tk.RIGHT)
    
    def setup_training_tab(self):
        """Setup the training tab"""
        # Paths frame
        paths_frame = ttk.LabelFrame(self.training_tab, text="Dataset Paths", padding=10)
        paths_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Training path
        train_frame = ttk.Frame(paths_frame)
        train_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(train_frame, text="Training:").pack(side=tk.LEFT)
        self.train_path_var = tk.StringVar(value="dataset/train")
        self.train_path_entry = tk.Entry(train_frame, textvariable=self.train_path_var, 
                                         width=50, bg="#1e1e1e", fg=self.fg_color,
                                         insertbackground=self.fg_color, relief=tk.FLAT)
        self.train_path_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        train_browse = tk.Button(train_frame, text="Browse", bg=self.button_bg, fg=self.fg_color,
                                 relief=tk.FLAT, command=lambda: self.browse_folder(self.train_path_var))
        train_browse.pack(side=tk.RIGHT)
        
        # Validation path
        val_frame = ttk.Frame(paths_frame)
        val_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(val_frame, text="Validation:").pack(side=tk.LEFT)
        self.val_path_var = tk.StringVar(value="dataset/validation")
        self.val_path_entry = tk.Entry(val_frame, textvariable=self.val_path_var,
                                       width=50, bg="#1e1e1e", fg=self.fg_color,
                                       insertbackground=self.fg_color, relief=tk.FLAT)
        self.val_path_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        val_browse = tk.Button(val_frame, text="Browse", bg=self.button_bg, fg=self.fg_color,
                               relief=tk.FLAT, command=lambda: self.browse_folder(self.val_path_var))
        val_browse.pack(side=tk.RIGHT)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(self.training_tab, text="Training Settings", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Epochs (for CNN)
        epochs_frame = ttk.Frame(settings_frame)
        epochs_frame.pack(fill=tk.X, pady=5)
        ttk.Label(epochs_frame, text="CNN Epochs:").pack(side=tk.LEFT)
        self.epochs_var = tk.StringVar(value="30")
        epochs_entry = tk.Entry(epochs_frame, textvariable=self.epochs_var, width=10,
                               bg="#1e1e1e", fg=self.fg_color, insertbackground=self.fg_color, relief=tk.FLAT)
        epochs_entry.pack(side=tk.LEFT, padx=10)
        ttk.Label(epochs_frame, text="(Early stopping enabled)", foreground="#888888").pack(side=tk.LEFT)
        
        # Training buttons
        btn_frame = ttk.Frame(self.training_tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=15)
        
        train_knn_btn = tk.Button(btn_frame, text="Train KNN/SVM", 
                                  font=('Segoe UI', 11, 'bold'),
                                  bg=self.success_color, fg="white",
                                  activebackground="#45a049",
                                  relief=tk.FLAT, padx=25, pady=10,
                                  command=self.train_knn_svm)
        train_knn_btn.pack(side=tk.LEFT, padx=5)
        
        train_cnn_btn = tk.Button(btn_frame, text="Train CNN", 
                                  font=('Segoe UI', 11, 'bold'),
                                  bg=self.accent_color, fg="white",
                                  activebackground="#357abd",
                                  relief=tk.FLAT, padx=25, pady=10,
                                  command=self.train_cnn)
        train_cnn_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.training_tab, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # Training log
        log_frame = ttk.LabelFrame(self.training_tab, text="Training Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.training_log = tk.Text(log_frame, height=12, 
                                    bg="#1e1e1e", fg="#00ff00",
                                    font=('Consolas', 9),
                                    relief=tk.FLAT)
        self.training_log.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(self.training_log, command=self.training_log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.training_log.config(yscrollcommand=scrollbar.set)
    
    def setup_prediction_tab(self):
        """Setup the prediction tab"""
        # Model loading frame
        model_frame = ttk.LabelFrame(self.prediction_tab, text="Models", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        btn_row = ttk.Frame(model_frame)
        btn_row.pack(fill=tk.X)
        
        load_knn_btn = tk.Button(btn_row, text="Load KNN/SVM",
                                 font=('Segoe UI', 10),
                                 bg=self.button_bg, fg="white",
                                 relief=tk.FLAT, padx=15, pady=5,
                                 command=self.load_knn_svm)
        load_knn_btn.pack(side=tk.LEFT, padx=5)
        
        load_cnn_btn = tk.Button(btn_row, text="Load CNN",
                                 font=('Segoe UI', 10),
                                 bg=self.button_bg, fg="white",
                                 relief=tk.FLAT, padx=15, pady=5,
                                 command=self.load_cnn)
        load_cnn_btn.pack(side=tk.LEFT, padx=5)
        
        self.model_status = tk.Label(btn_row, text="No model loaded",
                                     font=('Segoe UI', 9),
                                     bg=self.bg_color, fg="#888888")
        self.model_status.pack(side=tk.LEFT, padx=20)
        
        # Image selection
        img_frame = ttk.LabelFrame(self.prediction_tab, text="Image", padding=10)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        select_btn = tk.Button(img_frame, text="Select Image",
                               font=('Segoe UI', 10, 'bold'),
                               bg=self.accent_color, fg="white",
                               relief=tk.FLAT, padx=20, pady=8,
                               command=self.select_image)
        select_btn.pack(pady=10)
        
        self.image_label = tk.Label(img_frame, text="No image selected",
                                    bg=self.bg_color, fg="#888888",
                                    font=('Segoe UI', 10))
        self.image_label.pack(pady=10)
        
        # Predict buttons
        pred_frame = ttk.Frame(self.prediction_tab)
        pred_frame.pack(fill=tk.X, padx=10, pady=10)
        
        pred_knn_btn = tk.Button(pred_frame, text="Predict (KNN/SVM)",
                                 font=('Segoe UI', 11, 'bold'),
                                 bg=self.success_color, fg="white",
                                 relief=tk.FLAT, padx=20, pady=8,
                                 command=lambda: self.predict('knn_svm'))
        pred_knn_btn.pack(side=tk.LEFT, padx=5)
        
        pred_cnn_btn = tk.Button(pred_frame, text="Predict (CNN)",
                                 font=('Segoe UI', 11, 'bold'),
                                 bg=self.accent_color, fg="white",
                                 relief=tk.FLAT, padx=20, pady=8,
                                 command=lambda: self.predict('cnn'))
        pred_cnn_btn.pack(side=tk.LEFT, padx=5)
        
        # Results
        result_frame = ttk.LabelFrame(self.prediction_tab, text="Results", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.result_text = tk.Text(result_frame, height=10,
                                   bg="#1e1e1e", fg="#00ff00",
                                   font=('Consolas', 10),
                                   relief=tk.FLAT)
        self.result_text.pack(fill=tk.BOTH, expand=True)
    
    def browse_folder(self, var):
        """Browse for folder"""
        folder = filedialog.askdirectory()
        if folder:
            var.set(folder)
    
    def log(self, message):
        """Add message to training log"""
        self.training_log.insert(tk.END, message + "\n")
        self.training_log.see(tk.END)
        self.root.update_idletasks()
    
    def train_knn_svm(self):
        """Train KNN/SVM classifier"""
        train_path = self.train_path_var.get()
        val_path = self.val_path_var.get()
        
        if not os.path.exists(train_path):
            messagebox.showerror("Error", f"Training path not found: {train_path}")
            return
        
        self.training_log.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.status_label.config(text="Training KNN/SVM...")
        
        def train_thread():
            try:
                self.log("=" * 50)
                self.log("TRAINING KNN/SVM CLASSIFIER")
                self.log("=" * 50)
                self.log(f"Training path: {train_path}")
                self.log(f"Validation path: {val_path}")
                self.log("")
                
                self.knn_svm = TrafficSignalClassifier()
                self.progress_var.set(20)
                
                self.log("Loading training data and extracting features...")
                self.knn_svm.train(train_path, val_path)
                self.progress_var.set(80)
                
                self.log("")
                self.log("Saving model...")
                self.knn_svm.save_model('models/traffic_classifier')
                self.progress_var.set(100)
                
                self.log("")
                self.log("=" * 50)
                self.log("KNN/SVM Training Complete!")
                self.log("=" * 50)
                self.status_label.config(text="KNN/SVM trained successfully!")
                messagebox.showinfo("Success", "KNN/SVM model trained and saved!")
                
            except Exception as e:
                self.log(f"\nError: {e}")
                self.status_label.config(text="Training failed!")
                messagebox.showerror("Error", str(e))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def train_cnn(self):
        """Train CNN classifier"""
        train_path = self.train_path_var.get()
        val_path = self.val_path_var.get()
        
        if not os.path.exists(train_path):
            messagebox.showerror("Error", f"Training path not found: {train_path}")
            return
        
        try:
            epochs = int(self.epochs_var.get())
        except ValueError:
            epochs = 30
        
        self.training_log.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.status_label.config(text="Training CNN...")
        
        def train_thread():
            try:
                self.log("=" * 50)
                self.log("TRAINING CNN CLASSIFIER (Transfer Learning)")
                self.log("=" * 50)
                self.log(f"Training path: {train_path}")
                self.log(f"Validation path: {val_path}")
                self.log(f"Epochs: {epochs}")
                self.log("")
                
                self.cnn = CNNClassifier()
                self.progress_var.set(10)
                
                self.log("Building MobileNetV2 model...")
                self.log("(This may take a few minutes)")
                self.log("")
                
                self.cnn.train(train_path, val_path, epochs=epochs)
                self.progress_var.set(90)
                
                self.log("")
                self.log("Saving model...")
                self.cnn.save_model('models/cnn_model')
                self.progress_var.set(100)
                
                self.log("")
                self.log("=" * 50)
                self.log("CNN Training Complete!")
                self.log("=" * 50)
                self.status_label.config(text="CNN trained successfully!")
                messagebox.showinfo("Success", "CNN model trained and saved!")
                
            except Exception as e:
                self.log(f"\nError: {e}")
                self.status_label.config(text="Training failed!")
                messagebox.showerror("Error", str(e))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def load_knn_svm(self):
        """Load KNN/SVM model"""
        try:
            self.knn_svm = TrafficSignalClassifier()
            self.knn_svm.load_model('models/traffic_classifier')
            self.model_status.config(text="KNN/SVM loaded", fg=self.success_color)
            self.status_label.config(text="KNN/SVM model loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load KNN/SVM: {e}")
    
    def load_cnn(self):
        """Load CNN model"""
        try:
            self.cnn = CNNClassifier()
            self.cnn.load_model('models/cnn_model')
            self.model_status.config(text="CNN loaded", fg=self.success_color)
            self.status_label.config(text="CNN model loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CNN: {e}")
    
    def select_image(self):
        """Select an image for prediction"""
        path = filedialog.askopenfilename(
            title="Select Traffic Sign Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if path:
            self.selected_image = path
            
            # Display image
            img = Image.open(path)
            img.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            self.status_label.config(text=f"Selected: {os.path.basename(path)}")
    
    def predict(self, method):
        """Run prediction"""
        if not self.selected_image:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
        
        self.result_text.delete(1.0, tk.END)
        
        try:
            if method == 'knn_svm':
                if not self.knn_svm:
                    messagebox.showwarning("Warning", "Please load KNN/SVM model first!")
                    return
                results = self.knn_svm.predict(self.selected_image, method='ensemble')
                model_name = "KNN/SVM"
            else:
                if not self.cnn:
                    messagebox.showwarning("Warning", "Please load CNN model first!")
                    return
                results = self.cnn.predict(self.selected_image)
                model_name = "CNN"
            
            # Display results
            output = "=" * 45 + "\n"
            output += f"  PREDICTION: {results['top_class'].upper()}\n"
            output += f"  Confidence: {results['top_confidence']:.1%}\n"
            output += f"  Method: {model_name}\n"
            output += "=" * 45 + "\n\n"
            output += "All predictions:\n"
            output += "-" * 45 + "\n"
            
            for i, pred in enumerate(results['predictions'][:5], 1):
                bar_len = int(pred['confidence'] * 30)
                bar = "#" * bar_len + "-" * (30 - bar_len)
                output += f"  {i}. {pred['class']:15} {bar} {pred['confidence']:.1%}\n"
            
            self.result_text.insert(tk.END, output)
            self.status_label.config(text=f"Prediction: {results['top_class']} ({results['top_confidence']:.1%})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
    
    def on_close(self):
        """Handle window close"""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.root.destroy()
            sys.exit(0)
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()


def main():
    """Main entry point"""
    app = TrafficSignalGUI()
    app.run()


if __name__ == "__main__":
    main()
