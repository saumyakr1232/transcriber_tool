import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from pathlib import Path
import time
import tempfile
import queue
import traceback

# Import our modules
try:
    from config import get_config, Config
    from transcribers import TranscriberFactory
    from audio_recorder1 import AudioRecorder2  # Using AudioRecorder2 for dual audio capture
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

        # Live audio capture and transcription
        self.live_audio_capture = AudioRecorder2()

        # Initialize queues for transcription
        self.mic_queue = queue.Queue()
        self.system_queue = queue.Queue()

        # Create the UI
        self.create_ui()
        self.is_recording = False
        self.is_transcribing = False

        # Create separate transcribers for mic and system audio
        self.mic_queue = queue.Queue()
        self.system_queue = queue.Queue()

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

            # Initialize transcribers
            self.mic_transcriber = LiveTranscriber(
                config=self.config.config,
                transcription_callback=self.handle_mic_transcription,
                transcriber=self.transcriber
            )

            self.system_transcriber = LiveTranscriber(
                config=self.config.config,
                transcription_callback=self.handle_system_transcription,
                transcriber=self.transcriber
            )

            # Show loading dialog
            loading_dialog = tk.Toplevel(self.root)
            loading_dialog.title("Loading Model")
            loading_dialog.geometry("300x150")
            loading_dialog.transient(self.root)
            loading_dialog.grab_set()

            # Center the dialog
            loading_dialog.geometry("+%d+%d" % (
                self.root.winfo_x() + self.root.winfo_width()/2 - 150,
                self.root.winfo_y() + self.root.winfo_height()/2 - 75
            ))

            # Add loading message and progress bar
            ttk.Label(loading_dialog, text="Loading transcription model...", padding=10).pack()
            progress = ttk.Progressbar(loading_dialog, mode='indeterminate')
            progress.pack(padx=20, pady=10, fill=tk.X)
            progress.start()

            # Update the dialog
            loading_dialog.update()

            # Load the model
            self.transcriber.load_model()

            # Close the dialog
            loading_dialog.destroy()

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

        # Initialize text summarizer
        try:
            from text_summarizer import TextSummarizer
            self.summarizer = TextSummarizer(self.config)
        except Exception as e:
            print(f"Warning: Text summarization not available: {e}")
            self.summarizer = None

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
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True)
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
        ttk.Button(button_frame, text="Add Subtitles", command=self.add_subtitles_to_video).pack(side=tk.LEFT, padx=5)

        # Add summarize button if summarizer is available
        if self.summarizer:
            ttk.Button(button_frame, text="Summarize", command=self.summarize_transcription).pack(side=tk.LEFT, padx=5)

    def create_live_tab(self):
        """Create the UI for the live transcription tab."""
        # Device selection frame
        device_frame = ttk.LabelFrame(self.live_tab, text="Audio Device Selection", padding="10")
        device_frame.pack(fill=tk.X, pady=(0, 10))

        # Microphone selection
        ttk.Label(device_frame, text="Microphone:").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        self.mic_var = tk.StringVar()
        self.mic_dropdown = ttk.Combobox(device_frame, textvariable=self.mic_var, state="readonly")
        self.mic_dropdown['values'] = [str(mic) for mic in self.live_audio_capture.get_available_mics()]
        if self.mic_dropdown['values']:
            self.mic_dropdown.current(0)
        self.mic_dropdown.grid(row=0, column=1, sticky=tk.EW)

        # System audio selection
        ttk.Label(device_frame, text="System Audio:").grid(row=1, column=0, padx=(0, 5), sticky=tk.W)
        self.system_var = tk.StringVar()
        self.system_dropdown = ttk.Combobox(device_frame, textvariable=self.system_var, state="readonly")
        self.system_dropdown['values'] = [str(dev) for dev in self.live_audio_capture.get_available_system_devices()]
        if self.system_dropdown['values']:
            self.system_dropdown.current(0)
        self.system_dropdown.grid(row=1, column=1, sticky=tk.EW)

        # Configure grid weights
        device_frame.columnconfigure(1, weight=1)

        # Live transcription section
        live_transcription_frame = ttk.LabelFrame(self.live_tab, text="Live Transcription", padding="10")
        live_transcription_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Add text widget for live transcription
        self.live_transcription_text = tk.Text(live_transcription_frame, wrap=tk.WORD, height=10)
        self.live_transcription_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Configure text colors
        self.live_transcription_text.tag_configure("blue", foreground="blue")
        self.live_transcription_text.tag_configure("green", foreground="green")

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

        ttk.Button(live_button_frame, text="Save Transcription",
                   command=self.save_live_transcription).pack(side=tk.LEFT, padx=5)
        ttk.Button(live_button_frame, text="Clear", command=self.clear_live).pack(side=tk.LEFT, padx=5)

        # Output file for live transcription
        self.live_output_path = tk.StringVar()
        self.live_output_path.set("live_transcription.txt")

        # Start queue processing
        self.process_transcription_queue()

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

            # Perform transcription with timestamps
            text, segments = self.transcriber.transcribe_file_with_timestamps(self.file_path.get())

            # Store segments for later use (e.g., adding subtitles)
            self.segments = segments

            # Update the UI with the result
            self.root.after(0, lambda: self._update_transcription(text, segments))
        except Exception as e:
            self._show_error(f"Transcription failed: {e}")

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

    def _update_transcription(self, text, segments=None):
        """Update the transcription text area with the result."""
        # Update status
        self.status_var.set("Transcription complete")
        self.progress_var.set(100)

        # Update text area
        if text:
            self.transcription_text.delete(1.0, tk.END)

            # If we have segments with timestamps, display them
            if segments and len(segments) > 0:
                # Store segments for later use (e.g., adding subtitles)
                self.segments = segments

                for segment in segments:
                    # Format timestamp as [MM:SS]
                    start_time = segment.get("start", 0)
                    minutes = int(start_time // 60)
                    seconds = int(start_time % 60)
                    timestamp = f"[{minutes:02d}:{seconds:02d}] "

                    # Add the timestamped segment
                    self.transcription_text.insert(tk.END, timestamp + segment["text"] + "\n\n")
            else:
                # Just insert the full text if no segments
                self.segments = None
                self.transcription_text.insert(tk.END, text)
        else:
            self.segments = None
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

    def add_subtitles_to_video(self):
        """Add subtitles to the video file using the current transcription."""
        # Check if a file is selected
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select a video file.")
            return

        # Check if the file exists
        if not os.path.exists(self.file_path.get()):
            messagebox.showerror("Error", f"File {self.file_path.get()} does not exist.")
            return

        # Check if we have transcription segments
        if not hasattr(self, 'segments') or not self.segments:
            messagebox.showerror("Error", "No transcription segments available. Please transcribe the file first.")
            return

        # Ask for output file location
        filetypes = [
            ("MP4 Files", "*.mp4"),
            ("All Files", "*.*")
        ]
        output_path = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".mp4")
        if not output_path:
            return

        # Update status
        self.status_var.set("Adding subtitles...")
        self.progress_var.set(0)
        self.root.update()

        # Start subtitle addition in a separate thread
        threading.Thread(target=self._add_subtitles_thread, args=(output_path,), daemon=True).start()

    def _add_subtitles_thread(self, output_path):
        """Run the subtitle addition in a separate thread."""
        try:
            # Import the subtitle adder
            from subtitler import SubtitleAdder

            # Create subtitle adder with default style
            subtitle_adder = SubtitleAdder()

            # Add subtitles to the video
            subtitle_adder.add_subtitles_to_video(
                self.file_path.get(),
                self.segments,
                output_path
            )

            # Update UI
            self.root.after(0, lambda: self._subtitles_complete(output_path))
        except Exception as e:
            traceback.print_exc()
            error_message = f"Failed to add subtitles: {e}"
            self.root.after(0, lambda msg=error_message: self._show_error(msg))

    def _subtitles_complete(self, output_path):
        """Called when subtitle addition is complete."""
        self.status_var.set("Subtitles added successfully")
        self.progress_var.set(100)
        messagebox.showinfo("Success", f"Subtitles added successfully. Video saved to:\n{output_path}")

    def summarize_transcription(self):
        """Summarize the current transcription text."""
        if not self.summarizer:
            messagebox.showerror("Error", "Text summarization is not available.")
            return

        # Get the current transcription text
        text = self.transcription_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No text to summarize.")
            return

        try:
            # Update status
            self.status_var.set("Summarizing...")
            self.root.update()

            # Generate summary
            summary = self.summarizer.summarize(text)

            if summary:
                # Show summary in a new window
                summary_window = tk.Toplevel(self.root)
                summary_window.title("Text Summary")
                summary_window.geometry("600x400")

                # Add text widget for summary
                summary_text = scrolledtext.ScrolledText(summary_window, wrap=tk.WORD)
                summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                # Insert summary
                summary_text.insert(tk.END, summary)
                summary_text.config(state=tk.DISABLED)

                # Add save button
                def save_summary():
                    file_path = filedialog.asksaveasfilename(
                        defaultextension=".txt",
                        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
                    )
                    if file_path:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(summary)
                        messagebox.showinfo("Success", "Summary saved successfully.")

                ttk.Button(summary_window, text="Save Summary", command=save_summary).pack(pady=10)
            else:
                messagebox.showerror("Error", "Failed to generate summary.")

        except Exception as e:
            messagebox.showerror("Error", f"Error generating summary: {e}")
        finally:
            self.status_var.set("Ready")
            self.root.update()

    def clear_live(self):
        """Clear the live transcription text."""
        print("Clearing live transcription...")
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

            # Set selected devices
            selected_mic = next((mic for mic in self.live_audio_capture.get_available_mics()
                                if str(mic) == self.mic_var.get()), None)
            selected_system = next(
                (dev for dev in self.live_audio_capture.get_available_system_devices() if str(dev) == self.system_var.get()), None)

            self.live_audio_capture.set_mic_device(selected_mic)
            self.live_audio_capture.set_system_device(selected_system)

            # Start recording
            self.live_audio_capture.start_recording()
            self.is_recording = True

            # Start both transcribers
            self.mic_transcriber.start_transcription()
            self.system_transcriber.start_transcription()
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
        if not self.is_recording:
            return

        try:
            # Stop audio capture
            self.live_audio_capture.stop_recording()

            # Stop transcription if transcribers are initialized
            if hasattr(self, 'mic_transcriber'):
                self.mic_transcriber.stop_transcription()
            if hasattr(self, 'system_transcriber'):
                self.system_transcriber.stop_transcription()

            # Update UI
            self.is_recording = False
            self.live_status_var.set("Ready")
            self.start_button.config(text="Start Recording", state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop recording: {e}")
            self.start_button.config(state="normal")

    def _cleanup_resources(self):
        """Clean up resources and release memory."""
        # Wait for processing thread to finish
        if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        # Clear audio queue
        if self.live_audio_capture:
            while not self.live_audio_capture.mic_queue.empty():
                try:
                    self.live_audio_capture.mic_queue.get_nowait()
                except queue.Empty:
                    break
            while not self.live_audio_capture.system_queue.empty():
                try:
                    self.live_audio_capture.system_queue.get_nowait()
                except queue.Empty:
                    break

        # Reset instances
        # self.live_audio_capture = None
        # self.live_transcriber = None
        # self.processing_thread = None

    def _process_audio(self):
        """Process audio from both channels and send to respective transcribers"""
        # Track recording start time for timestamp calculation
        self.recording_start_time = time.time()

        while self.is_recording or not (self.live_audio_capture.mic_queue.empty() and self.live_audio_capture.system_queue.empty()):
            # Process microphone audio
            mic_frames = self.live_audio_capture.get_mic_audio()
            if mic_frames is not None and len(mic_frames) > 0:
                # Calculate current timestamp relative to recording start
                current_time = time.time() - self.recording_start_time
                self.mic_transcriber.add_audio_data(mic_frames)

            # Process system audio
            system_frames = self.live_audio_capture.get_system_audio()
            if system_frames is not None and len(system_frames) > 0:
                # Calculate current timestamp relative to recording start
                current_time = time.time() - self.recording_start_time
                self.system_transcriber.add_audio_data(system_frames)

            time.sleep(0.1)

    def handle_mic_transcription(self, text, timestamp=None):
        """Callback function for microphone transcription results"""
        if text:
            # Include timestamp if available
            if timestamp is not None:
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                formatted_time = f"[{minutes:02d}:{seconds:02d}] "
                self.mic_queue.put((formatted_time + text, "blue"))
            else:
                self.mic_queue.put((text, "blue"))

    def handle_system_transcription(self, text, timestamp=None):
        """Callback function for system audio transcription results"""
        if text:
            # Include timestamp if available
            if timestamp is not None:
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                formatted_time = f"[{minutes:02d}:{seconds:02d}] "
                self.system_queue.put((formatted_time + text, "green"))
            else:
                self.system_queue.put((text, "green"))

    def process_transcription_queue(self):
        """Process transcriptions from both microphone and system audio queues."""
        # Process microphone transcriptions
        while not self.mic_queue.empty():
            transcription, color = self.mic_queue.get()

            self.live_transcription_text.config(state=tk.NORMAL)
            if self.live_transcription_text.index('end-1c') != '1.0':
                self.live_transcription_text.insert(tk.END, "\n")
            self.live_transcription_text.insert(tk.END, f"Mic: {transcription}", color)
            self.live_transcription_text.see(tk.END)
            self.live_transcription_text.config(state=tk.DISABLED)

        # Process system transcriptions
        while not self.system_queue.empty():
            transcription, color = self.system_queue.get()

            self.live_transcription_text.config(state=tk.NORMAL)
            if self.live_transcription_text.index('end-1c') != '1.0':
                self.live_transcription_text.insert(tk.END, "\n")
            self.live_transcription_text.insert(tk.END, f"System: {transcription}", color)
            self.live_transcription_text.see(tk.END)
            self.live_transcription_text.config(state=tk.DISABLED)

        # Schedule the next check
        self.root.after(100, self.process_transcription_queue)

    def change_engine(self):
        """Change the transcription engine."""
        # Get the selected engine
        engine = self.engine_var.get()

        # Update the configuration
        self.config.set("engine", engine)
        self.config.save()

        # Reinitialize the transcriber
        try:
            self.stop_recording()
            self.setup_transcriber()
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
        ttk.Button(vosk_tab, text="Browse", command=lambda: self._browse_model_dir(
            vosk_model_path)).grid(row=0, column=2, padx=5)

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
            self.setup_transcriber()
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
