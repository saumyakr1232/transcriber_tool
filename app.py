#!/usr/bin/env python3

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import time

# Import our transcriber module
try:
    from transcriber import AudioTranscriber
except ImportError:
    print("Error: Could not import the transcriber module.")
    print("Make sure transcriber.py is in the same directory as this script.")
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
            messagebox.showinfo(
                "Model Required", 
                "No Vosk model found. Please download a model from \n"
                "https://alphacephei.com/vosk/models \n"
                "and extract it to the ./models/ directory."
            )
        
        # Try to initialize the transcriber
        try:
            self.transcriber = AudioTranscriber()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize transcriber: {e}")
            self.root.destroy()
    
    def create_ui(self):
        """Create the user interface."""
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
    
    def clear_all(self):
        """Clear all fields and reset the UI."""
        self.file_path.set("")
        self.output_path.set("")
        self.transcription_text.delete(1.0, tk.END)
        self.status_var.set("Ready")
        self.progress_var.set(0)


def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()