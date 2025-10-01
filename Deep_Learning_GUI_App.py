#!/usr/bin/env python3
"""
Deep Learning GUI Application
A user-friendly interface for training deep learning models without coding knowledge.
Created for non-technical users to easily train neural networks.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    TF_AVAILABLE = True
    print(f"TensorFlow {tf.__version__} loaded successfully!")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not found. Installing...")

class DeepLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Deep Learning Model Trainer - Easy Mode")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.history = None
        self.dataset_loaded = False
        self.model_trained = False
        
        # Setup the GUI
        self.setup_gui()
        
        # Check TensorFlow availability
        if not TF_AVAILABLE:
            self.log_message("⚠️ TensorFlow not installed. Please install it first.")
            self.show_install_dialog()

    def setup_gui(self):
        """Setup the main GUI interface"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🧠 Deep Learning Model Trainer",
            font=('Arial', 24, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(
            title_frame,
            text="Train AI models with just a few clicks - No coding required!",
            font=('Arial', 12),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        subtitle_label.pack()
        
        # Create main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel for controls
        left_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        
        # Right panel for results
        right_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        right_frame.pack(side='right', fill='both', expand=True)
        
        self.setup_control_panel(left_frame)
        self.setup_results_panel(right_frame)

    def setup_control_panel(self, parent):
        """Setup the left control panel with all the buttons"""
        # Control panel title
        control_title = tk.Label(
            parent,
            text="🎮 Control Panel",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        control_title.pack(pady=10)
        
        # Dataset Section
        dataset_frame = tk.LabelFrame(parent, text="📊 Step 1: Choose Dataset", font=('Arial', 12, 'bold'))
        dataset_frame.pack(fill='x', padx=10, pady=5)
        
        # Dataset selection
        self.dataset_var = tk.StringVar(value="MNIST Digits")
        datasets = [
            "MNIST Digits (Handwritten Numbers)",
            "CIFAR-10 (Objects & Animals)", 
            "Fashion-MNIST (Clothing)",
            "Upload Your Own Data"
        ]
        
        for dataset in datasets:
            tk.Radiobutton(
                dataset_frame,
                text=dataset,
                variable=self.dataset_var,
                value=dataset.split()[0],
                font=('Arial', 10)
            ).pack(anchor='w', padx=5, pady=2)
        
        tk.Button(
            dataset_frame,
            text="📥 Load Dataset",
            command=self.load_dataset,
            bg='#3498db',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=20,
            pady=5
        ).pack(pady=10)
        
        # Model Configuration Section
        model_frame = tk.LabelFrame(parent, text="🏗️ Step 2: Configure Model", font=('Arial', 12, 'bold'))
        model_frame.pack(fill='x', padx=10, pady=5)
        
        # Model type selection
        tk.Label(model_frame, text="Model Type:", font=('Arial', 10, 'bold')).pack(anchor='w', padx=5)
        self.model_type_var = tk.StringVar(value="Simple CNN")
        model_types = ["Simple CNN", "Deep CNN", "Dense Network"]
        
        for model_type in model_types:
            tk.Radiobutton(
                model_frame,
                text=model_type,
                variable=self.model_type_var,
                value=model_type,
                font=('Arial', 10)
            ).pack(anchor='w', padx=5, pady=1)
        
        # Training parameters
        tk.Label(model_frame, text="Training Epochs:", font=('Arial', 10, 'bold')).pack(anchor='w', padx=5, pady=(10,0))
        self.epochs_var = tk.IntVar(value=5)
        epochs_scale = tk.Scale(
            model_frame,
            from_=1,
            to_=20,
            orient='horizontal',
            variable=self.epochs_var,
            length=200
        )
        epochs_scale.pack(padx=5, pady=2)
        
        tk.Button(
            model_frame,
            text="🏗️ Build Model",
            command=self.build_model,
            bg='#e67e22',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=20,
            pady=5
        ).pack(pady=10)
        
        # Training Section
        training_frame = tk.LabelFrame(parent, text="🚀 Step 3: Train Model", font=('Arial', 12, 'bold'))
        training_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(
            training_frame,
            text="🚀 Start Training",
            command=self.start_training,
            bg='#27ae60',
            fg='white',
            font=('Arial', 12, 'bold'),
            relief='flat',
            padx=20,
            pady=10
        ).pack(pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            training_frame,
            variable=self.progress_var,
            maximum=100,
            length=200
        )
        self.progress_bar.pack(pady=5)
        
        # Status label
        self.status_label = tk.Label(
            training_frame,
            text="Ready to start",
            font=('Arial', 10),
            fg='#7f8c8d'
        )
        self.status_label.pack(pady=5)
        
        # Evaluation Section
        eval_frame = tk.LabelFrame(parent, text="📊 Step 4: Test & Deploy", font=('Arial', 12, 'bold'))
        eval_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(
            eval_frame,
            text="📊 Evaluate Model",
            command=self.evaluate_model,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=20,
            pady=5
        ).pack(pady=5)
        
        tk.Button(
            eval_frame,
            text="💾 Save Model",
            command=self.save_model,
            bg='#34495e',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=20,
            pady=5
        ).pack(pady=5)
        
        tk.Button(
            eval_frame,
            text="🔮 Test Prediction",
            command=self.test_prediction,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=20,
            pady=5
        ).pack(pady=5)

    def setup_results_panel(self, parent):
        """Setup the right results panel"""
        # Results panel title
        results_title = tk.Label(
            parent,
            text="📈 Results & Visualizations",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        results_title.pack(pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Log tab
        log_frame = tk.Frame(self.notebook)
        self.notebook.add(log_frame, text='📝 Activity Log')
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Visualization tab
        viz_frame = tk.Frame(self.notebook)
        self.notebook.add(viz_frame, text='📊 Charts')
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.patch.set_facecolor('white')
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Results tab
        results_frame = tk.Frame(self.notebook)
        self.notebook.add(results_frame, text='🎯 Results')
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=('Arial', 11),
            bg='#f8f9fa'
        )
        self.results_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initial log message
        self.log_message("🎉 Welcome to Deep Learning Model Trainer!")
        self.log_message("👋 Follow the steps on the left to train your AI model.")
        self.log_message("💡 No coding knowledge required - just click buttons!")

    def log_message(self, message):
        """Add message to the activity log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()

    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update()

    def show_install_dialog(self):
        """Show installation dialog for TensorFlow"""
        result = messagebox.askyesno(
            "Install TensorFlow",
            "TensorFlow is required but not installed.\nWould you like to install it now?\n\n(This may take a few minutes)"
        )
        if result:
            self.install_tensorflow()

    def install_tensorflow(self):
        """Install TensorFlow in a separate thread"""
        def install():
            self.log_message("📦 Installing TensorFlow... Please wait...")
            self.update_status("Installing TensorFlow...")
            try:
                import subprocess
                result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.log_message("✅ TensorFlow installed successfully!")
                    self.log_message("🔄 Please restart the application.")
                    messagebox.showinfo("Success", "TensorFlow installed successfully!\nPlease restart the application.")
                else:
                    self.log_message(f"❌ Installation failed: {result.stderr}")
            except Exception as e:
                self.log_message(f"❌ Error installing TensorFlow: {str(e)}")
            
            self.update_status("Ready to start")
        
        threading.Thread(target=install, daemon=True).start()

    def load_dataset(self):
        """Load the selected dataset"""
        if not TF_AVAILABLE:
            messagebox.showerror("Error", "TensorFlow is not installed!")
            return
            
        dataset_name = self.dataset_var.get()
        self.log_message(f"📥 Loading {dataset_name} dataset...")
        self.update_status("Loading dataset...")
        
        try:
            if dataset_name == "MNIST":
                (self.X_train, self.y_train), (self.X_test, self.y_test) = keras.datasets.mnist.load_data()
                self.X_train = self.X_train.astype('float32') / 255.0
                self.X_test = self.X_test.astype('float32') / 255.0
                self.X_train = self.X_train.reshape(-1, 28, 28, 1)
                self.X_test = self.X_test.reshape(-1, 28, 28, 1)
                self.num_classes = 10
                self.dataset_type = "image"
                
            elif dataset_name == "CIFAR-10":
                (self.X_train, self.y_train), (self.X_test, self.y_test) = keras.datasets.cifar10.load_data()
                self.X_train = self.X_train.astype('float32') / 255.0
                self.X_test = self.X_test.astype('float32') / 255.0
                self.y_train = self.y_train.flatten()
                self.y_test = self.y_test.flatten()
                self.num_classes = 10
                self.dataset_type = "image"
                
            elif dataset_name == "Fashion-MNIST":
                (self.X_train, self.y_train), (self.X_test, self.y_test) = keras.datasets.fashion_mnist.load_data()
                self.X_train = self.X_train.astype('float32') / 255.0
                self.X_test = self.X_test.astype('float32') / 255.0
                self.X_train = self.X_train.reshape(-1, 28, 28, 1)
                self.X_test = self.X_test.reshape(-1, 28, 28, 1)
                self.num_classes = 10
                self.dataset_type = "image"
                
            elif dataset_name == "Upload":
                self.load_custom_dataset()
                return
            
            self.dataset_loaded = True
            self.log_message(f"✅ Dataset loaded successfully!")
            self.log_message(f"📊 Training samples: {self.X_train.shape[0]}")
            self.log_message(f"📊 Test samples: {self.X_test.shape[0]}")
            self.log_message(f"📊 Input shape: {self.X_train.shape[1:]}")
            self.update_status("Dataset loaded")
            
            # Show sample images
            self.show_sample_data()
            
        except Exception as e:
            self.log_message(f"❌ Error loading dataset: {str(e)}")
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.update_status("Error loading dataset")

    def load_custom_dataset(self):
        """Load custom dataset from file"""
        file_path = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                data = pd.read_csv(file_path)
                self.log_message(f"📄 Loaded file: {os.path.basename(file_path)}")
                self.log_message(f"📊 Data shape: {data.shape}")
                
                # Simple preprocessing - assume last column is target
                X = data.iloc[:, :-1].values
                y = data.iloc[:, -1].values
                
                # Split data
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Scale data
                scaler = StandardScaler()
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
                
                self.num_classes = len(np.unique(y))
                self.dataset_type = "tabular"
                self.dataset_loaded = True
                
                self.log_message("✅ Custom dataset loaded and preprocessed!")
                self.update_status("Custom dataset loaded")
                
            except Exception as e:
                self.log_message(f"❌ Error loading custom dataset: {str(e)}")
                messagebox.showerror("Error", f"Failed to load custom dataset: {str(e)}")

    def show_sample_data(self):
        """Display sample data in the visualization panel"""
        if self.dataset_type == "image":
            # Clear previous plots
            for ax in self.axes.flat:
                ax.clear()
            
            # Show sample images
            for i in range(4):
                ax = self.axes[i//2, i%2]
                if self.X_train.shape[-1] == 1:
                    ax.imshow(self.X_train[i].squeeze(), cmap='gray')
                else:
                    ax.imshow(self.X_train[i])
                ax.set_title(f'Sample {i+1}: Class {self.y_train[i]}')
                ax.axis('off')
            
            self.fig.suptitle('Sample Data from Dataset')
            self.canvas.draw()

    def build_model(self):
        """Build the neural network model"""
        if not self.dataset_loaded:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
            
        self.log_message("🏗️ Building neural network model...")
        self.update_status("Building model...")
        
        try:
            model_type = self.model_type_var.get()
            
            if self.dataset_type == "image":
                if model_type == "Simple CNN":
                    self.model = models.Sequential([
                        layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.X_train.shape[1:]),
                        layers.MaxPooling2D((2, 2)),
                        layers.Conv2D(64, (3, 3), activation='relu'),
                        layers.MaxPooling2D((2, 2)),
                        layers.Flatten(),
                        layers.Dense(64, activation='relu'),
                        layers.Dropout(0.5),
                        layers.Dense(self.num_classes, activation='softmax')
                    ])
                    
                elif model_type == "Deep CNN":
                    self.model = models.Sequential([
                        layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.X_train.shape[1:]),
                        layers.BatchNormalization(),
                        layers.Conv2D(32, (3, 3), activation='relu'),
                        layers.MaxPooling2D((2, 2)),
                        layers.Dropout(0.25),
                        
                        layers.Conv2D(64, (3, 3), activation='relu'),
                        layers.BatchNormalization(),
                        layers.Conv2D(64, (3, 3), activation='relu'),
                        layers.MaxPooling2D((2, 2)),
                        layers.Dropout(0.25),
                        
                        layers.Flatten(),
                        layers.Dense(512, activation='relu'),
                        layers.BatchNormalization(),
                        layers.Dropout(0.5),
                        layers.Dense(self.num_classes, activation='softmax')
                    ])
                    
            else:  # tabular data
                self.model = models.Sequential([
                    layers.Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)),
                    layers.Dropout(0.3),
                    layers.Dense(64, activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(self.num_classes, activation='softmax')
                ])
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.log_message("✅ Model built successfully!")
            self.log_message(f"🧠 Model type: {model_type}")
            self.log_message(f"🔢 Total parameters: {self.model.count_params():,}")
            self.update_status("Model built")
            
            # Show model summary in results
            model_summary = []
            self.model.summary(print_fn=lambda x: model_summary.append(x))
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "🏗️ Model Architecture:\n\n")
            self.results_text.insert(tk.END, "\n".join(model_summary))
            
        except Exception as e:
            self.log_message(f"❌ Error building model: {str(e)}")
            messagebox.showerror("Error", f"Failed to build model: {str(e)}")
            self.update_status("Error building model")

    def start_training(self):
        """Start model training in a separate thread"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please build a model first!")
            return
            
        def train():
            try:
                self.log_message("🚀 Starting model training...")
                self.update_status("Training model...")
                self.progress_var.set(0)
                
                epochs = self.epochs_var.get()
                
                # Custom callback to update progress
                class ProgressCallback(keras.callbacks.Callback):
                    def __init__(self, app, total_epochs):
                        self.app = app
                        self.total_epochs = total_epochs
                        
                    def on_epoch_end(self, epoch, logs=None):
                        progress = ((epoch + 1) / self.total_epochs) * 100
                        self.app.progress_var.set(progress)
                        self.app.log_message(f"📈 Epoch {epoch + 1}/{self.total_epochs} - "
                                           f"Loss: {logs['loss']:.4f} - "
                                           f"Accuracy: {logs['accuracy']:.4f}")
                        self.app.root.update()
                
                callback = ProgressCallback(self, epochs)
                
                # Train the model
                self.history = self.model.fit(
                    self.X_train, self.y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(self.X_test, self.y_test),
                    callbacks=[callback],
                    verbose=0
                )
                
                self.model_trained = True
                self.log_message("🎉 Training completed successfully!")
                self.update_status("Training completed")
                self.progress_var.set(100)
                
                # Show training results
                self.show_training_results()
                
                messagebox.showinfo("Success", "Model training completed successfully!")
                
            except Exception as e:
                self.log_message(f"❌ Training failed: {str(e)}")
                messagebox.showerror("Error", f"Training failed: {str(e)}")
                self.update_status("Training failed")
        
        threading.Thread(target=train, daemon=True).start()

    def show_training_results(self):
        """Display training results and plots"""
        if self.history is None:
            return
            
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot training history
        epochs_range = range(1, len(self.history.history['loss']) + 1)
        
        # Loss plot
        self.axes[0, 0].plot(epochs_range, self.history.history['loss'], 'b-', label='Training Loss')
        self.axes[0, 0].plot(epochs_range, self.history.history['val_loss'], 'r-', label='Validation Loss')
        self.axes[0, 0].set_title('Model Loss')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True)
        
        # Accuracy plot
        self.axes[0, 1].plot(epochs_range, self.history.history['accuracy'], 'b-', label='Training Accuracy')
        self.axes[0, 1].plot(epochs_range, self.history.history['val_accuracy'], 'r-', label='Validation Accuracy')
        self.axes[0, 1].set_title('Model Accuracy')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Accuracy')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True)
        
        # Final metrics
        final_loss = self.history.history['val_loss'][-1]
        final_acc = self.history.history['val_accuracy'][-1]
        
        self.axes[1, 0].text(0.5, 0.7, f'Final Validation Loss\n{final_loss:.4f}', 
                            ha='center', va='center', fontsize=16, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        self.axes[1, 0].text(0.5, 0.3, f'Final Validation Accuracy\n{final_acc:.4f} ({final_acc*100:.2f}%)', 
                            ha='center', va='center', fontsize=16,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        self.axes[1, 0].set_title('Final Results')
        self.axes[1, 0].set_xlim(0, 1)
        self.axes[1, 0].set_ylim(0, 1)
        self.axes[1, 0].axis('off')
        
        # Model performance summary
        self.axes[1, 1].axis('off')
        summary_text = f"""
Model Performance Summary:

✅ Training Completed
📊 Final Accuracy: {final_acc*100:.2f}%
📉 Final Loss: {final_loss:.4f}
⏱️ Epochs Trained: {len(epochs_range)}
🎯 Model Status: Ready for Use

Next Steps:
• Evaluate on test data
• Save your trained model
• Make predictions
        """
        self.axes[1, 1].text(0.05, 0.95, summary_text, fontsize=10, 
                            verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        self.canvas.draw()

    def evaluate_model(self):
        """Evaluate the trained model"""
        if not self.model_trained:
            messagebox.showwarning("Warning", "Please train the model first!")
            return
            
        self.log_message("📊 Evaluating model performance...")
        self.update_status("Evaluating model...")
        
        try:
            # Evaluate on test set
            test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
            
            # Make predictions
            predictions = self.model.predict(self.X_test, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Generate detailed results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "🎯 MODEL EVALUATION RESULTS\n")
            self.results_text.insert(tk.END, "=" * 50 + "\n\n")
            
            self.results_text.insert(tk.END, f"📊 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
            self.results_text.insert(tk.END, f"📉 Test Loss: {test_loss:.4f}\n\n")
            
            # Classification report
            if len(np.unique(self.y_test)) <= 20:  # Only for reasonable number of classes
                report = classification_report(self.y_test, predicted_classes)
                self.results_text.insert(tk.END, "📋 Detailed Classification Report:\n")
                self.results_text.insert(tk.END, "-" * 40 + "\n")
                self.results_text.insert(tk.END, report + "\n")
            
            # Performance interpretation
            self.results_text.insert(tk.END, "\n🤖 AI Performance Interpretation:\n")
            self.results_text.insert(tk.END, "-" * 40 + "\n")
            
            if test_accuracy >= 0.95:
                self.results_text.insert(tk.END, "🌟 EXCELLENT: Your model is performing exceptionally well!\n")
            elif test_accuracy >= 0.85:
                self.results_text.insert(tk.END, "👍 GOOD: Your model is performing well.\n")
            elif test_accuracy >= 0.70:
                self.results_text.insert(tk.END, "👌 FAIR: Your model is performing reasonably.\n")
            else:
                self.results_text.insert(tk.END, "⚠️ NEEDS IMPROVEMENT: Consider training longer or adjusting parameters.\n")
            
            self.log_message(f"✅ Evaluation completed! Test accuracy: {test_accuracy:.4f}")
            self.update_status("Evaluation completed")
            
        except Exception as e:
            self.log_message(f"❌ Evaluation failed: {str(e)}")
            messagebox.showerror("Error", f"Evaluation failed: {str(e)}")

    def save_model(self):
        """Save the trained model"""
        if not self.model_trained:
            messagebox.showwarning("Warning", "Please train the model first!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.model.save(file_path)
                self.log_message(f"💾 Model saved successfully: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"Model saved successfully!\n{file_path}")
                self.update_status("Model saved")
            except Exception as e:
                self.log_message(f"❌ Error saving model: {str(e)}")
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")

    def test_prediction(self):
        """Test prediction on a random sample"""
        if not self.model_trained:
            messagebox.showwarning("Warning", "Please train the model first!")
            return
            
        try:
            # Get a random test sample
            idx = np.random.randint(0, len(self.X_test))
            sample = self.X_test[idx:idx+1]
            true_label = self.y_test[idx]
            
            # Make prediction
            prediction = self.model.predict(sample, verbose=0)
            predicted_label = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Show results
            result_text = f"""
🔮 PREDICTION TEST RESULT
========================

🎯 True Label: {true_label}
🤖 Predicted Label: {predicted_label}
📊 Confidence: {confidence:.4f} ({confidence*100:.2f}%)

{'✅ CORRECT PREDICTION!' if predicted_label == true_label else '❌ WRONG PREDICTION'}

Sample #{idx} from test set
            """
            
            messagebox.showinfo("Prediction Result", result_text)
            self.log_message(f"🔮 Prediction test: True={true_label}, Predicted={predicted_label}, Confidence={confidence:.4f}")
            
        except Exception as e:
            self.log_message(f"❌ Prediction test failed: {str(e)}")
            messagebox.showerror("Error", f"Prediction test failed: {str(e)}")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = DeepLearningApp(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
