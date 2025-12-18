"""
GUI interface for AI Voice Clone.
Provides a graphical user interface for voice recording, training, and cloning.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import sys
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ai_voice_clone.audio_input import AudioInput
from ai_voice_clone.feature_extraction import FeatureExtractor
from ai_voice_clone.model import VoiceCloner
from ai_voice_clone.training import Trainer
from ai_voice_clone.inference import InferenceEngine
from ai_voice_clone.config import Config


class VoiceCloneGUI:
    """Main GUI application for AI Voice Clone."""

    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("AI Voice Clone")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # Initialize components
        self.config = Config()
        self.config.load()
        self.audio_input = AudioInput()
        self.feature_extractor = FeatureExtractor(self.config)
        self.model = VoiceCloner(self.config)
        self.trainer = Trainer(self.model, self.feature_extractor, self.config)
        self.inference_engine = InferenceEngine(self.model, self.feature_extractor, self.config)

        # Current audio files
        self.reference_audio_path = None
        self.output_audio_path = None

        # Recording state
        self.is_recording = False
        self.recording_thread = None

        self.setup_ui()
        self.setup_logging()

    def setup_logging(self):
        """Setup logging for GUI."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('voice_clone_gui.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_ui(self):
        """Setup the user interface."""
        # Create main notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.record_tab = ttk.Frame(self.notebook)
        self.train_tab = ttk.Frame(self.notebook)
        self.clone_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.record_tab, text="🎤 Record Voice")
        self.notebook.add(self.train_tab, text="🧠 Train Model")
        self.notebook.add(self.clone_tab, text="🔊 Clone Voice")
        self.notebook.add(self.settings_tab, text="⚙️ Settings")

        # Setup each tab
        self.setup_record_tab()
        self.setup_train_tab()
        self.setup_clone_tab()
        self.setup_settings_tab()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def setup_record_tab(self):
        """Setup the voice recording tab."""
        frame = ttk.LabelFrame(self.record_tab, text="Voice Recording", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Duration selection
        ttk.Label(frame, text="Recording Duration (seconds):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.duration_var = tk.IntVar(value=5)
        duration_spin = ttk.Spinbox(frame, from_=1, to=30, textvariable=self.duration_var, width=10)
        duration_spin.grid(row=0, column=1, sticky=tk.W, pady=5)

        # Output file selection
        ttk.Label(frame, text="Save to:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.record_output_var = tk.StringVar(value="my_voice.wav")
        output_entry = ttk.Entry(frame, textvariable=self.record_output_var, width=40)
        output_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(frame, text="Browse...", command=self.browse_record_output).grid(row=1, column=2, padx=5)

        # Record button
        self.record_button = ttk.Button(frame, text="🎤 Start Recording", command=self.start_recording)
        self.record_button.grid(row=2, column=0, columnspan=3, pady=20)

        # Progress bar for recording
        self.record_progress = ttk.Progressbar(frame, mode='determinate', length=300)
        self.record_progress.grid(row=3, column=0, columnspan=3, pady=10)

        # Recording status
        self.record_status_var = tk.StringVar(value="Ready to record")
        ttk.Label(frame, textvariable=self.record_status_var).grid(row=4, column=0, columnspan=3, pady=5)

    def setup_train_tab(self):
        """Setup the model training tab."""
        frame = ttk.LabelFrame(self.train_tab, text="Model Training", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Input audio selection
        ttk.Label(frame, text="Training Audio File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.train_input_var = tk.StringVar()
        train_input_entry = ttk.Entry(frame, textvariable=self.train_input_var, width=40)
        train_input_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(frame, text="Browse...", command=self.browse_train_input).grid(row=0, column=2, padx=5)

        # Training parameters
        ttk.Label(frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.epochs_var = tk.IntVar(value=50)
        epochs_spin = ttk.Spinbox(frame, from_=10, to=200, textvariable=self.epochs_var, width=10)
        epochs_spin.grid(row=1, column=1, sticky=tk.W, pady=5)

        # Train button
        self.train_button = ttk.Button(frame, text="🚀 Start Training", command=self.start_training)
        self.train_button.grid(row=2, column=0, columnspan=3, pady=20)

        # Training progress
        self.train_progress = ttk.Progressbar(frame, mode='determinate', length=300)
        self.train_progress.grid(row=3, column=0, columnspan=3, pady=10)

        # Training status
        self.train_status_var = tk.StringVar(value="Ready to train")
        ttk.Label(frame, textvariable=self.train_status_var).grid(row=4, column=0, columnspan=3, pady=5)

        # Training log
        ttk.Label(frame, text="Training Log:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.train_log_text = tk.Text(frame, height=8, width=60, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(frame, command=self.train_log_text.yview)
        self.train_log_text.config(yscrollcommand=scrollbar.set)
        self.train_log_text.grid(row=6, column=0, columnspan=2, pady=5)
        scrollbar.grid(row=6, column=2, sticky=(tk.N, tk.S))

    def setup_clone_tab(self):
        """Setup the voice cloning tab."""
        frame = ttk.LabelFrame(self.clone_tab, text="Voice Cloning", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Reference audio selection
        ttk.Label(frame, text="Reference Audio:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.clone_ref_var = tk.StringVar()
        clone_ref_entry = ttk.Entry(frame, textvariable=self.clone_ref_var, width=40)
        clone_ref_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(frame, text="Browse...", command=self.browse_clone_ref).grid(row=0, column=2, padx=5)

        # Text input
        ttk.Label(frame, text="Text to Generate:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.clone_text_var = tk.StringVar(value="Hello, this is my cloned voice!")
        text_entry = ttk.Entry(frame, textvariable=self.clone_text_var, width=40)
        text_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

        # Output file selection
        ttk.Label(frame, text="Save Output to:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.clone_output_var = tk.StringVar(value="cloned_voice.wav")
        clone_output_entry = ttk.Entry(frame, textvariable=self.clone_output_var, width=40)
        clone_output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(frame, text="Browse...", command=self.browse_clone_output).grid(row=2, column=2, padx=5)

        # Clone button
        self.clone_button = ttk.Button(frame, text="🎵 Generate Voice", command=self.start_cloning)
        self.clone_button.grid(row=3, column=0, columnspan=3, pady=20)

        # Cloning progress
        self.clone_progress = ttk.Progressbar(frame, mode='indeterminate', length=300)
        self.clone_progress.grid(row=4, column=0, columnspan=3, pady=10)

        # Cloning status
        self.clone_status_var = tk.StringVar(value="Ready to clone")
        ttk.Label(frame, textvariable=self.clone_status_var).grid(row=5, column=0, columnspan=3, pady=5)

    def setup_settings_tab(self):
        """Setup the settings tab."""
        frame = ttk.LabelFrame(self.settings_tab, text="Settings", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Audio settings
        audio_frame = ttk.LabelFrame(frame, text="Audio Settings", padding=5)
        audio_frame.pack(fill=tk.X, pady=5)

        ttk.Label(audio_frame, text="Sample Rate (Hz):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.sample_rate_var = tk.IntVar(value=self.config.get('audio.sample_rate', 22050))
        ttk.Entry(audio_frame, textvariable=self.sample_rate_var, width=15).grid(row=0, column=1, pady=2)

        ttk.Label(audio_frame, text="Channels:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.channels_var = tk.IntVar(value=self.config.get('audio.channels', 1))
        ttk.Entry(audio_frame, textvariable=self.channels_var, width=15).grid(row=1, column=1, pady=2)

        # Model settings
        model_frame = ttk.LabelFrame(frame, text="Model Settings", padding=5)
        model_frame.pack(fill=tk.X, pady=10)

        ttk.Label(model_frame, text="Hidden Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.hidden_size_var = tk.IntVar(value=self.config.get('model.encoder_hidden_size', 256))
        ttk.Entry(model_frame, textvariable=self.hidden_size_var, width=15).grid(row=0, column=1, pady=2)

        ttk.Label(model_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.learning_rate_var = tk.DoubleVar(value=self.config.get('training.learning_rate', 0.001))
        ttk.Entry(model_frame, textvariable=self.learning_rate_var, width=15).grid(row=1, column=1, pady=2)

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="💾 Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔄 Reset to Defaults", command=self.reset_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📂 Open Config File", command=self.open_config_file).pack(side=tk.LEFT, padx=5)

    # Event handlers
    def browse_record_output(self):
        """Browse for recording output file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filename:
            self.record_output_var.set(filename)

    def browse_train_input(self):
        """Browse for training input file."""
        filename = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")]
        )
        if filename:
            self.train_input_var.set(filename)

    def browse_clone_ref(self):
        """Browse for cloning reference audio."""
        filename = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")]
        )
        if filename:
            self.clone_ref_var.set(filename)

    def browse_clone_output(self):
        """Browse for cloning output file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filename:
            self.clone_output_var.set(filename)

    def start_recording(self):
        """Start audio recording."""
        if self.is_recording:
            return

        duration = self.duration_var.get()
        output_path = self.record_output_var.get()

        if not output_path:
            messagebox.showerror("Error", "Please specify an output file.")
            return

        self.is_recording = True
        self.record_button.config(text="⏹️ Stop Recording", state=tk.DISABLED)
        self.record_status_var.set("Recording...")

        # Start recording in a separate thread
        self.recording_thread = threading.Thread(
            target=self._record_audio_thread,
            args=(duration, output_path)
        )
        self.recording_thread.start()

    def _record_audio_thread(self, duration, output_path):
        """Recording thread function."""
        try:
            # Update progress bar
            self.record_progress['value'] = 0
            for i in range(duration):
                if not self.is_recording:
                    break
                self.root.after(1000, lambda i=i: self.record_progress.config(value=(i+1)/duration*100))
                import time
                time.sleep(1)

            if self.is_recording:
                # Record audio
                audio_data = self.audio_input.record_audio(duration=duration)
                self.audio_input.save_audio(audio_data, output_path)

                self.root.after(0, lambda: self.record_status_var.set(f"Recording saved to: {output_path}"))
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Recording saved to {output_path}"))

        except Exception as e:
            self.logger.error(f"Recording error: {str(e)}")
            self.root.after(0, lambda: self.record_status_var.set(f"Recording failed: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Recording failed: {str(e)}"))

        finally:
            self.is_recording = False
            self.root.after(0, lambda: self.record_button.config(text="🎤 Start Recording", state=tk.NORMAL))

    def start_training(self):
        """Start model training."""
        input_file = self.train_input_var.get()
        epochs = self.epochs_var.get()

        if not input_file or not os.path.exists(input_file):
            messagebox.showerror("Error", "Please select a valid training audio file.")
            return

        self.train_button.config(state=tk.DISABLED)
        self.train_status_var.set("Training in progress...")
        self.train_progress['value'] = 0

        # Clear log
        self.train_log_text.config(state=tk.NORMAL)
        self.train_log_text.delete(1.0, tk.END)
        self.train_log_text.config(state=tk.DISABLED)

        # Start training in a separate thread
        training_thread = threading.Thread(
            target=self._train_model_thread,
            args=(input_file, epochs)
        )
        training_thread.start()

    def _train_model_thread(self, input_file, epochs):
        """Training thread function."""
        try:
            # Update progress callback
            def progress_callback(epoch, loss):
                progress = (epoch + 1) / epochs * 100
                self.root.after(0, lambda: self.train_progress.config(value=progress))
                self.root.after(0, lambda: self.train_status_var.set(".1f"))
                # Log to text widget
                self.root.after(0, lambda: self._append_to_log(".1f"))

            # Train model (simplified - in real implementation, pass progress callback)
            results = self.trainer.train(input_file)

            self.root.after(0, lambda: self.train_status_var.set("Training completed!"))
            self.root.after(0, lambda: self.train_progress.config(value=100))
            self.root.after(0, lambda: messagebox.showinfo("Success", "Model training completed!"))

        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            self.root.after(0, lambda: self.train_status_var.set(f"Training failed: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {str(e)}"))

        finally:
            self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL))

    def _append_to_log(self, text):
        """Append text to training log."""
        self.train_log_text.config(state=tk.NORMAL)
        self.train_log_text.insert(tk.END, text + "\n")
        self.train_log_text.see(tk.END)
        self.train_log_text.config(state=tk.DISABLED)

    def start_cloning(self):
        """Start voice cloning."""
        ref_audio = self.clone_ref_var.get()
        text = self.clone_text_var.get()
        output_file = self.clone_output_var.get()

        if not ref_audio or not os.path.exists(ref_audio):
            messagebox.showerror("Error", "Please select a valid reference audio file.")
            return

        if not text.strip():
            messagebox.showerror("Error", "Please enter text to generate.")
            return

        if not output_file:
            messagebox.showerror("Error", "Please specify an output file.")
            return

        self.clone_button.config(state=tk.DISABLED)
        self.clone_status_var.set("Generating voice...")
        self.clone_progress.start()

        # Start cloning in a separate thread
        cloning_thread = threading.Thread(
            target=self._clone_voice_thread,
            args=(ref_audio, text, output_file)
        )
        cloning_thread.start()

    def _clone_voice_thread(self, ref_audio, text, output_file):
        """Cloning thread function."""
        try:
            # Load model if available
            model_path = "models/voice_clone_model.pt"
            if os.path.exists(model_path):
                self.inference_engine.load_model(model_path)
            else:
                self.root.after(0, lambda: messagebox.showwarning("Warning",
                    "No trained model found. Please train a model first. Using basic generation."))

            # Generate voice
            cloned_audio = self.inference_engine.generate_voice(text, ref_audio)

            # Save output
            self.audio_input.save_audio(cloned_audio, output_file)

            self.root.after(0, lambda: self.clone_status_var.set(f"Voice cloned and saved to: {output_file}"))
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Voice cloned and saved to {output_file}"))

        except Exception as e:
            self.logger.error(f"Cloning error: {str(e)}")
            self.root.after(0, lambda: self.clone_status_var.set(f"Cloning failed: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Cloning failed: {str(e)}"))

        finally:
            self.root.after(0, lambda: self.clone_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.clone_progress.stop())

    def save_settings(self):
        """Save current settings to config."""
        try:
            # Update config with current values
            self.config.set('audio.sample_rate', self.sample_rate_var.get())
            self.config.set('audio.channels', self.channels_var.get())
            self.config.set('model.encoder_hidden_size', self.hidden_size_var.get())
            self.config.set('training.learning_rate', self.learning_rate_var.get())

            # Save to file
            self.config.save()

            messagebox.showinfo("Success", "Settings saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

    def reset_settings(self):
        """Reset settings to defaults."""
        self.config = Config()  # Reload defaults
        self._update_settings_display()
        messagebox.showinfo("Success", "Settings reset to defaults!")

    def _update_settings_display(self):
        """Update settings display with current config values."""
        self.sample_rate_var.set(self.config.get('audio.sample_rate', 22050))
        self.channels_var.set(self.config.get('audio.channels', 1))
        self.hidden_size_var.set(self.config.get('model.encoder_hidden_size', 256))
        self.learning_rate_var.set(self.config.get('training.learning_rate', 0.001))

    def open_config_file(self):
        """Open the configuration file in default editor."""
        config_path = Path("config.yaml")
        if config_path.exists():
            os.startfile(str(config_path))  # Windows
        else:
            messagebox.showerror("Error", "Configuration file not found.")


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = VoiceCloneGUI(root)

    # Set window icon if available
    try:
        root.iconbitmap("icon.ico")  # Optional icon file
    except:
        pass

    root.mainloop()


if __name__ == "__main__":
    main()