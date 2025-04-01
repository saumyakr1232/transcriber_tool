#!/usr/bin/env python3

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import time

# Import our modules
try:
    from transcriber import AudioTranscriber
    from config import get_config, Config
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Make sure transcriber.py and config.py are in the same directory as this script.")
    sys.exit(1)

class TranscriberApp:
    """GUI application for the audio transcriber."""
    
    def __init__(self, root):
        """Initialize the application.
        
        Args:
            root: The tkinter root window
        """
        self.root = root
        self.root.title("Audio Transcriber")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # Load configuration
        self.config = get_config()
        
        # Set up the transcriber
        self.setup_transcriber()
        
        # Create the UI
        self.create_ui()
    
    def setup_transcriber(self):
        """Set up the transcriber with a model."""
        # Check if models directory exists
        models_dir = Path("./models")
        if not models_dir.exists():
            # Create the models directory
            models_dir.mkdir(exist_ok=True)
            
            # Show different messages based on selected engine
            engine = self.config.get_engine()
            if engine == "vosk":
                messagebox.showinfo(
                    "Model Required", 
                    "No Vosk model found. Please download a model from \n"
                    "https://alphacephei.com/vosk/models \n"
                    "and extract it to the ./models/ directory."
                )
            elif engine == "whisper":
                messagebox.showinfo(
                    "Whisper Selected",
                    "Using Whisper engine. The model will be downloaded automatically\n"
                    "on first use if not already downloaded."
                )
        
        # Try to initialize the transcriber
        try:
            self.transcriber = AudioTranscriber()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize transcriber: {e}")
            self.root.destroy()
    
    def create_ui(self):
        """Create the user interface."""
        # Create menu bar
        self.create_menu()
        
        # Create a main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Select Audio File", padding="10")
        file_frame.pack(fill=tk.X, pady=10)
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.RIGHT, padx=5)
        
        # Output file section
        output_frame = ttk.LabelFrame(main_frame, text="Output File (Optional)", padding="10")
        output_frame.pack(fill=tk.X, pady=10)
        
        self.output_path = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side=tk.RIGHT, padx=5)
        
        # Transcription section
        transcription_frame = ttk.LabelFrame(main_frame, text="Transcription", padding="10")
        transcription_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Transcription text area
        self.transcription_text = tk.Text(transcription_frame, wrap=tk.WORD, height=10)
        self.transcription_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Scrollbar for text area
        scrollbar = ttk.Scrollbar(transcription_frame, command=self.transcription_text.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.transcription_text.config(yscrollcommand=scrollbar.set)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(anchor=tk.W, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Transcribe", command=self.start_transcription).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save", command=self.save_transcription).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Settings", command=self.open_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.destroy).pack(side=tk.RIGHT, padx=5)
    
    def browse_file(self):
        """Open a file dialog to select an audio file."""
        filetypes = [
            ("Audio Files", "*.mp4 *.wav"),
            ("MP4 Files", "*.mp4"),
            ("WAV Files", "*.wav"),
            ("All Files", "*.*")
        ]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.file_path.set(filename)
            # Suggest an output filename
            base_name = os.path.splitext(os.path.basename(filename))[0]
            self.output_path.set(f"{base_name}_transcription.txt")
    
    def browse_output(self):
        """Open a file dialog to select an output file."""
        filetypes = [
            ("Text Files", "*.txt"),
            ("All Files", "*.*")
        ]
        filename = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".txt")
        if filename:
            self.output_path.set(filename)
    
    def start_transcription(self):
        """Start the transcription process in a separate thread."""
        # Check if a file is selected
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select an audio file.")
            return
        
        # Check if the file exists
        if not os.path.exists(self.file_path.get()):
            messagebox.showerror("Error", f"File {self.file_path.get()} does not exist.")
            return
        
        # Clear the transcription text
        self.transcription_text.delete(1.0, tk.END)
        
        # Update status
        self.status_var.set("Transcribing...")
        self.progress_var.set(0)
        
        # Start transcription in a separate thread
        threading.Thread(target=self._transcribe_thread, daemon=True).start()
    
    def _transcribe_thread(self):
        """Run the transcription in a separate thread."""
        try:
            # Update progress periodically
            self._update_progress_thread()
            
            # Perform transcription
            text = self.transcriber.transcribe_file(self.file_path.get())
            
            # Update the UI with the result
            self.root.after(0, lambda: self._update_transcription(text))
        except Exception as e:
            self.root.after(0, lambda: self._show_error(f"Transcription failed: {e}"))
    
    def _update_progress_thread(self):
        """Update the progress bar periodically."""
        # This is a simple simulation since we don't have real progress info
        progress_thread = threading.Thread(target=self._progress_simulator, daemon=True)
        progress_thread.start()
    
    def _progress_simulator(self):
        """Simulate progress updates."""
        for i in range(1, 100):
            time.sleep(0.1)  # Adjust based on expected transcription time
            self.progress_var.set(i)
            # Stop if we're done
            if self.status_var.get() != "Transcribing...":
                break
    
    def _update_transcription(self, text):
        """Update the transcription text area with the result."""
        self.transcription_text.delete(1.0, tk.END)
        self.transcription_text.insert(tk.END, text)
        self.status_var.set("Transcription complete")
        self.progress_var.set(100)
        
        # Save automatically if output path is specified
        if self.output_path.get():
            self.save_transcription()
    
    def _show_error(self, message):
        """Show an error message."""
        messagebox.showerror("Error", message)
        self.status_var.set("Error")
        self.progress_var.set(0)
    
    def save_transcription(self):
        """Save the transcription to a file."""
        # Get the transcription text
        text = self.transcription_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showinfo("Info", "No transcription to save.")
            return
        
        # Get the output path
        output_path = self.output_path.get()
        if not output_path:
            # Ask for a file path
            output_path = filedialog.asksaveasfilename(
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
                defaultextension=".txt"
            )
            if not output_path:
                return
            self.output_path.set(output_path)
        
        # Save the file
        try:
            with open(output_path, "w") as f:
                f.write(text)
            messagebox.showinfo("Success", f"Transcription saved to {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")
    
    def create_menu(self):
        """Create the application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Audio File", command=self.browse_file)
        file_menu.add_command(label="Save Transcription", command=self.save_transcription)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Configure Transcription", command=self.open_settings)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def open_settings(self):
        """Open the settings dialog."""
        SettingsDialog(self.root, self.config, self.reload_transcriber)
    
    def reload_transcriber(self):
        """Reload the transcriber with updated configuration."""
        try:
            self.transcriber = AudioTranscriber()
            self.status_var.set(f"Using {self.config.get_engine().capitalize()} engine")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize transcriber: {e}")
    
    def show_about(self):
        """Show the about dialog."""
        messagebox.showinfo(
            "About Audio Transcriber",
            "Audio Transcription Tool\n\n"
            "A tool for transcribing audio files using offline speech recognition.\n\n"
            "Supports both Vosk and Whisper engines for transcription."
        )
    
    def clear_all(self):
        """Clear all fields and reset the UI."""
        self.file_path.set("")
        self.output_path.set("")
        self.transcription_text.delete(1.0, tk.END)
        self.status_var.set("Ready")
        self.progress_var.set(0)


class SettingsDialog:
    """Dialog for configuring transcription settings."""
    
    def __init__(self, parent, config, callback=None):
        """Initialize the settings dialog.
        
        Args:
            parent: The parent window
            config: The configuration object
            callback: Function to call when settings are saved
        """
        self.parent = parent
        self.config = config
        self.callback = callback
        
        # Create the dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Transcription Settings")
        self.dialog.geometry("500x400")
        self.dialog.minsize(400, 300)
        self.dialog.transient(parent)  # Set to be on top of the parent window
        self.dialog.grab_set()  # Modal dialog
        
        # Create the UI
        self.create_ui()
    
    def create_ui(self):
        """Create the settings dialog UI."""
        # Create a notebook (tabbed interface)
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # General settings tab
        general_frame = ttk.Frame(notebook, padding=10)
        notebook.add(general_frame, text="General")
        
        # Engine selection
        ttk.Label(general_frame, text="Speech Recognition Engine:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.engine_var = tk.StringVar(value=self.config.get("engine", "vosk"))
        engine_frame = ttk.Frame(general_frame)
        engine_frame.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Radiobutton(engine_frame, text="Vosk", variable=self.engine_var, value="vosk").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(engine_frame, text="Whisper", variable=self.engine_var, value="whisper").pack(side=tk.LEFT, padx=5)
        
        # Vosk settings tab
        vosk_frame = ttk.Frame(notebook, padding=10)
        notebook.add(vosk_frame, text="Vosk Settings")
        
        ttk.Label(vosk_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.vosk_model_path = tk.StringVar(value=self.config.get("models.vosk.model_path", ""))
        path_frame = ttk.Frame(vosk_frame)
        path_frame.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Entry(path_frame, textvariable=self.vosk_model_path, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Button(path_frame, text="Browse", command=self.browse_vosk_model).pack(side=tk.LEFT, padx=5)
        
        # Help text
        ttk.Label(vosk_frame, text="Download Vosk models from: https://alphacephei.com/vosk/models").grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=10)
        
        # Whisper settings tab
        whisper_frame = ttk.Frame(notebook, padding=10)
        notebook.add(whisper_frame, text="Whisper Settings")
        
        # Model size
        ttk.Label(whisper_frame, text="Model Size:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.whisper_model_size = tk.StringVar(value=self.config.get("models.whisper.model_size", "base"))
        sizes = [("Tiny (fastest, least accurate)", "tiny"), 
                 ("Base", "base"), 
                 ("Small", "small"), 
                 ("Medium", "medium"), 
                 ("Large (slowest, most accurate)", "large")]
        
        size_frame = ttk.Frame(whisper_frame)
        size_frame.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        for i, (text, value) in enumerate(sizes):
            ttk.Radiobutton(size_frame, text=text, variable=self.whisper_model_size, value=value).grid(row=i, column=0, sticky=tk.W)
        
        # Use GPU
        self.use_gpu = tk.BooleanVar(value=self.config.get("models.whisper.use_gpu", True))
        ttk.Checkbutton(whisper_frame, text="Use GPU if available", variable=self.use_gpu).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Language
        ttk.Label(whisper_frame, text="Language (leave empty for auto-detection):").grid(row=2, column=0, sticky=tk.W, pady=5)
        
        self.whisper_language = tk.StringVar(value=self.config.get("models.whisper.language", ""))
        ttk.Entry(whisper_frame, textvariable=self.whisper_language, width=10).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Save", command=self.save_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def browse_vosk_model(self):
        """Open a directory dialog to select a Vosk model directory."""
        directory = filedialog.askdirectory(title="Select Vosk Model Directory")
        if directory:
            self.vosk_model_path.set(directory)
    
    def save_settings(self):
        """Save the settings to the configuration."""
        # Save engine selection
        self.config.set("engine", self.engine_var.get())
        
        # Save Vosk settings
        self.config.set("models.vosk.model_path", self.vosk_model_path.get())
        
        # Save Whisper settings
        self.config.set("models.whisper.model_size", self.whisper_model_size.get())
        self.config.set("models.whisper.use_gpu", self.use_gpu.get())
        self.config.set("models.whisper.language", self.whisper_language.get())
        
        # Save configuration to file
        self.config.save()
        
        # Call the callback function if provided
        if self.callback:
            self.callback()
        
        # Close the dialog
        self.dialog.destroy()


def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()