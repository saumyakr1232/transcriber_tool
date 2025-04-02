#!/usr/bin/env python3

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import time
import tempfile

# Import our modules
try:
    from config import get_config, Config
    from transcribers import TranscriberFactory
    from live_audio_capture import AudioRecorder  # Changed from LiveAudioCapture
    from audio_visualizer import DualAudioVisualizer
    from live_transcriber import LiveTranscriber
    import numpy as np
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Make sure all required modules are in the same directory as this script.")
    sys.exit(1)

class TranscriberApp:
    """GUI application for the audio transcriber with file and live transcription support."""
    
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
        
        # Live audio capture and transcription
        self.live_audio_capture = None
        self.live_transcriber = None
        self.is_recording = False
        self.is_transcribing = False
        
        # Update interval for visualization (ms)
        self.update_interval = 50
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
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
            self.transcriber = TranscriberFactory.create_transcriber(self.config.config)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize transcriber: {e}")
            self.root.destroy()
    
    def create_ui(self):
        """Create the user interface."""
        # Create menu bar
        self.create_menu()
        
        # Create a notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.file_tab = ttk.Frame(self.notebook, padding="10")
        self.live_tab = ttk.Frame(self.notebook, padding="10")
        
        self.notebook.add(self.file_tab, text="File Transcription")
        self.notebook.add(self.live_tab, text="Live Transcription")
        
        # Create UI for file transcription tab
        self.create_file_tab()
        
        # Create UI for live transcription tab
        self.create_live_tab()
    
    def create_menu(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Audio File", command=self.browse_file)
        file_menu.add_command(label="Save Transcription", command=self.save_transcription)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Engine menu
        engine_menu = tk.Menu(menubar, tearoff=0)
        
        # Create variables for radio buttons
        self.engine_var = tk.StringVar(value=self.config.get_engine())
        
        # Get available engines from factory
        available_engines = TranscriberFactory.get_available_engines()
        
        for engine in available_engines:
            engine_menu.add_radiobutton(
                label=engine.capitalize(), 
                variable=self.engine_var, 
                value=engine,
                command=self.change_engine
            )
        
        menubar.add_cascade(label="Engine", menu=engine_menu)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Preferences", command=self.open_settings)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_file_tab(self):
        """Create the UI for the file transcription tab."""
        # File selection section
        file_frame = ttk.LabelFrame(self.file_tab, text="Select Audio File", padding="10")
        file_frame.pack(fill=tk.X, pady=10)
        
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.RIGHT, padx=5)
        
        # Output file section
        output_frame = ttk.LabelFrame(self.file_tab, text="Output File (Optional)", padding="10")
        output_frame.pack(fill=tk.X, pady=10)
        
        self.output_path = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side=tk.RIGHT, padx=5)
        
        # Transcription section
        transcription_frame = ttk.LabelFrame(self.file_tab, text="Transcription", padding="10")
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
        self.progress = ttk.Progressbar(self.file_tab, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.file_tab, textvariable=self.status_var)
        status_label.pack(anchor=tk.W, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.file_tab)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Transcribe", command=self.start_transcription).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save", command=self.save_transcription).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_all).pack(side=tk.LEFT, padx=5)
    
    def create_live_tab(self):
        """Create the UI for the live transcription tab."""
        # Live transcription section
        live_transcription_frame = ttk.LabelFrame(self.live_tab, text="Live Transcription", padding="10")
        live_transcription_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add text widget for live transcription
        self.live_transcription_text = tk.Text(live_transcription_frame, wrap=tk.WORD, height=10)
        self.live_transcription_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Add scrollbar for live transcription
        live_scrollbar = ttk.Scrollbar(live_transcription_frame, command=self.live_transcription_text.yview)
        live_scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.live_transcription_text.config(yscrollcommand=live_scrollbar.set)
        
        # Status label
        self.live_status_var = tk.StringVar(value="Ready")
        live_status_label = ttk.Label(self.live_tab, textvariable=self.live_status_var)
        live_status_label.pack(anchor=tk.W, pady=5)
        
        # Buttons
        live_button_frame = ttk.Frame(self.live_tab)
        live_button_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(live_button_frame, text="Start Recording", command=self.toggle_recording)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(live_button_frame, text="Save Transcription", command=self.save_live_transcription).pack(side=tk.LEFT, padx=5)
        ttk.Button(live_button_frame, text="Clear", command=self.clear_live).pack(side=tk.LEFT, padx=5)
        
        # Output file for live transcription
        self.live_output_path = tk.StringVar()
        self.live_output_path.set("live_transcription.txt")
    
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
        for i in range(1, 101):
            # Check if transcription is still running
            if self.status_var.get() != "Transcribing...":
                break
            
            # Update progress
            self.root.after(0, lambda val=i: self.progress_var.set(val))
            
            # Sleep for a short time
            time.sleep(0.1)
    
    def _update_transcription(self, text):
        """Update the transcription text area with the result."""
        # Update status
        self.status_var.set("Transcription complete")
        self.progress_var.set(100)
        
        # Update text area
        if text:
            self.transcription_text.delete(1.0, tk.END)
            self.transcription_text.insert(tk.END, text)
        else:
            self._show_error("Transcription failed.")
    
    def _show_error(self, message):
        """Show an error message."""
        self.status_var.set("Error")
        messagebox.showerror("Error", message)
    
    def save_transcription(self):
        """Save the transcription to a file."""
        # Get the transcription text
        text = self.transcription_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showerror("Error", "No transcription to save.")
            return
        
        # Get the output file path
        output_path = self.output_path.get()
        if not output_path:
            # Open a file dialog
            self.browse_output()
            output_path = self.output_path.get()
            if not output_path:
                return
        
        # Save the transcription
        try:
            with open(output_path, "w") as f:
                f.write(text)
            messagebox.showinfo("Success", f"Transcription saved to {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save transcription: {e}")
    
    def save_live_transcription(self):
        """Save the live transcription to a file."""
        # Get the transcription text
        text = self.live_transcription_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showerror("Error", "No transcription to save.")
            return
        
        # Get the output file path
        output_path = self.live_output_path.get()
        if not output_path:
            # Open a file dialog
            filetypes = [
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ]
            output_path = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".txt")
            if not output_path:
                return
            self.live_output_path.set(output_path)
        
        # Save the transcription
        try:
            with open(output_path, "w") as f:
                f.write(text)
            messagebox.showinfo("Success", f"Transcription saved to {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save transcription: {e}")
    
    def clear_all(self):
        """Clear all input and output fields in the file tab."""
        self.file_path.set("")
        self.output_path.set("")
        self.transcription_text.delete(1.0, tk.END)
        self.status_var.set("Ready")
        self.progress_var.set(0)
    
    def clear_live(self):
        """Clear the live transcription text."""
        self.live_transcription_text.delete(1.0, tk.END)
        self.live_status_var.set("Ready")
    
    def toggle_recording(self):
        """Toggle recording and transcription on/off."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording and transcribing audio."""
        try:
            # Update UI
            self.live_status_var.set("Initializing...")
            self.start_button.config(text="Stop Recording", state="disabled")
            self.root.update()
            
            # Initialize live audio capture if not already done
            if self.live_audio_capture is None:
                # Create live audio capture instance
                self.live_audio_capture = AudioRecorder()
            
            # Initialize live transcriber if not already done
            if self.live_transcriber is None:
                # Create live transcriber instance using factory
                self.live_transcriber = LiveTranscriber(
                    config=self.config.config,
                    transcription_callback=self._update_live_transcription
                )
            
            # Start recording
            self.live_audio_capture.start_recording()
            self.is_recording = True
            
            # Start transcription
            self.live_transcriber.start_transcription()
            self.is_transcribing = True
            
            # Start processing in a separate thread
            self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
            self.processing_thread.start()
            
            # Update UI
            self.live_status_var.set("Recording and transcribing...")
            self.start_button.config(text="Stop Recording", state="normal")
        
        except Exception as e:
            self.live_status_var.set("Error")
            messagebox.showerror("Error", f"Failed to start recording: {e}")
            self.start_button.config(text="Start Recording", state="normal")
            # Clean up resources on error
            self._cleanup_resources()
    
    def stop_recording(self):
        """Stop recording and transcribing audio."""
        try:
            # Stop transcription
            if self.live_transcriber and self.is_transcribing:
                self.live_transcriber.stop_transcription()
                self.is_transcribing = False
            
            # Stop recording
            if self.live_audio_capture and self.is_recording:
                self.live_audio_capture.stop_recording()
                self.is_recording = False
            
            # Clean up resources
            self._cleanup_resources()
            
            # Update UI
            self.live_status_var.set("Ready")
            self.start_button.config(text="Start Recording", state="normal")
        
        except Exception as e:
            self.live_status_var.set("Error")
            messagebox.showerror("Error", f"Failed to stop recording: {e}")
            self.start_button.config(text="Start Recording", state="normal")
    
    def _cleanup_resources(self):
        """Clean up resources and release memory."""
        # Wait for processing thread to finish
        if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # Clear audio queue
        if self.live_audio_capture:
            while not self.live_audio_capture.audio_queue.empty():
                try:
                    self.live_audio_capture.audio_queue.get_nowait()
                except queue.Empty:
                    break
        
        # Reset instances
        self.live_audio_capture = None
        self.live_transcriber = None
        self.processing_thread = None
    
    def _process_audio(self):
        """Process audio from the recorder and send to transcriber"""
        while self.is_recording or not self.live_audio_capture.audio_queue.empty():
            frames = self.live_audio_capture.get_audio()
            if frames is not None and len(frames) > 0:
                self.live_transcriber.add_audio_data(frames)
            time.sleep(0.1)
    
    def _update_live_transcription(self, text):
        """Update the live transcription text area with the result."""
        if not text:
            return
            
        self.live_transcription_text.config(state=tk.NORMAL)
        if self.live_transcription_text.index('end-1c') != '1.0':
            self.live_transcription_text.insert(tk.END, " ")
        self.live_transcription_text.insert(tk.END, text)
        self.live_transcription_text.see(tk.END)
        self.live_transcription_text.config(state=tk.DISABLED)
    
    def change_engine(self):
        """Change the transcription engine."""
        # Get the selected engine
        engine = self.engine_var.get()
        
        # Update the configuration
        self.config.set("engine", engine)
        self.config.save()
        
        # Reinitialize the transcriber
        try:
            self.transcriber = AudioTranscriber()
            
            # Also reinitialize the live transcriber if it exists
            if self.live_transcriber:
                # Stop transcription if running
                if self.is_transcribing:
                    self.live_transcriber.stop_transcription()
                    self.is_transcribing = False
                
                # Create a new live transcriber
                def transcription_callback(text):
                    if text:
                        # Update the UI with the result
                        self.root.after(0, lambda: self._update_live_transcription(text))
                
                self.live_transcriber = LiveTranscriber(
                    config=self.config.config,
                    transcription_callback=transcription_callback
                )
                
                # Restart transcription if recording
                if self.is_recording:
                    self.live_transcriber.start_transcription()
                    self.is_transcribing = True
            
            messagebox.showinfo("Engine Changed", f"Transcription engine changed to {engine.capitalize()}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize transcriber: {e}")
    
    def open_settings(self):
        """Open the settings dialog."""
        # Create a new top-level window
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("500x400")
        settings_window.minsize(400, 300)
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Create a notebook for tabs
        notebook = ttk.Notebook(settings_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs for different settings
        general_tab = ttk.Frame(notebook, padding=10)
        vosk_tab = ttk.Frame(notebook, padding=10)
        whisper_tab = ttk.Frame(notebook, padding=10)
        
        notebook.add(general_tab, text="General")
        notebook.add(vosk_tab, text="Vosk")
        notebook.add(whisper_tab, text="Whisper")
        
        # General settings
        ttk.Label(general_tab, text="Engine:").grid(row=0, column=0, sticky=tk.W, pady=5)
        engine_var = tk.StringVar(value=self.config.get_engine())
        engine_combo = ttk.Combobox(general_tab, textvariable=engine_var, state="readonly")
        engine_combo["values"] = TranscriberFactory.get_available_engines()
        engine_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Vosk settings
        ttk.Label(vosk_tab, text="Model Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        vosk_model_path = tk.StringVar(value=self.config.get("models.vosk.model_path", ""))
        ttk.Entry(vosk_tab, textvariable=vosk_model_path, width=40).grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Button(vosk_tab, text="Browse", command=lambda: self._browse_model_dir(vosk_model_path)).grid(row=0, column=2, padx=5)
        
        # Whisper settings
        ttk.Label(whisper_tab, text="Model Size:").grid(row=0, column=0, sticky=tk.W, pady=5)
        whisper_model_size = tk.StringVar(value=self.config.get("models.whisper.model_size", "base"))
        size_combo = ttk.Combobox(whisper_tab, textvariable=whisper_model_size, state="readonly")
        size_combo["values"] = ["tiny", "base", "small", "medium", "large"]
        size_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(whisper_tab, text="Use GPU:").grid(row=1, column=0, sticky=tk.W, pady=5)
        use_gpu = tk.BooleanVar(value=self.config.get("models.whisper.use_gpu", True))
        ttk.Checkbutton(whisper_tab, variable=use_gpu).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(whisper_tab, text="Language:").grid(row=2, column=0, sticky=tk.W, pady=5)
        language = tk.StringVar(value=self.config.get("models.whisper.language", ""))
        ttk.Entry(whisper_tab, textvariable=language, width=10).grid(row=2, column=1, sticky=tk.W, pady=5)
        ttk.Label(whisper_tab, text="(Leave empty for auto-detection)").grid(row=2, column=2, sticky=tk.W, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(settings_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Save", command=lambda: self._save_settings(
            engine_var.get(),
            vosk_model_path.get(),
            whisper_model_size.get(),
            use_gpu.get(),
            language.get(),
            settings_window
        )).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _browse_model_dir(self, path_var):
        """Open a directory dialog to select a model directory."""
        directory = filedialog.askdirectory()
        if directory:
            path_var.set(directory)
    
    def _save_settings(self, engine, vosk_model_path, whisper_model_size, use_gpu, language, window):
        """Save the settings and close the dialog."""
        # Update the configuration
        self.config.set("engine", engine)
        self.config.set("models.vosk.model_path", vosk_model_path)
        self.config.set("models.whisper.model_size", whisper_model_size)
        self.config.set("models.whisper.use_gpu", use_gpu)
        self.config.set("models.whisper.language", language)
        
        # Save the configuration
        self.config.save()
        
        # Update the engine variable in the menu
        self.engine_var.set(engine)
        
        # Reinitialize the transcriber
        try:
            self.transcriber = AudioTranscriber()
            messagebox.showinfo("Settings Saved", "Settings saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize transcriber: {e}")
        
        # Close the dialog
        window.destroy()
    
    def show_about(self):
        """Show the about dialog."""
        messagebox.showinfo(
            "About Audio Transcriber",
            "Audio Transcriber\n\n"
            "A tool for transcribing audio from MP4 or WAV files\n"
            "and capturing live audio for real-time transcription.\n\n"
            "Supported engines: Vosk, Whisper"
        )
    
    def on_close(self):
        """Handle window close event."""
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Clean up resources
        self._cleanup_resources()
        
        # Destroy the window
        self.root.destroy()


def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()