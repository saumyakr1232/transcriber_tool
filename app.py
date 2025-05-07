import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path
import time
import tempfile
import queue
import traceback
import customtkinter as ctk
import ffmpeg

import ssl
import certifi
import urllib.request

ssl_context = ssl.create_default_context(cafile=certifi.where())


# Set appearance mode and default color theme
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

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
            root: The customtkinter root window
        """
        self.root = root
        self.root.title("Audio Transcriber")
        self.root.geometry("900x600")
        self.root.minsize(800, 400)

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
        
        # Audio input states
        self.mic_enabled = True
        self.system_enabled = True
        
        # Device monitoring
        self.device_monitor_active = False
        self.last_known_mics = set(str(mic) for mic in self.live_audio_capture.get_available_mics())
        self.last_known_system_devices = set(str(dev) for dev in self.live_audio_capture.get_available_system_devices())
        self.device_change_detected = False
        self.start_device_monitoring()

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
            loading_dialog = ctk.CTkToplevel(self.root)
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
            ctk.CTkLabel(loading_dialog, text="Loading transcription model...", padx=10, pady=10).pack()
            progress = ctk.CTkProgressBar(loading_dialog)
            progress.pack(padx=20, pady=10, fill=tk.X)
            progress.configure(mode="indeterminate")
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

        # Create a tabview for tabs
        self.tabview = ctk.CTkTabview(self.root, corner_radius=10)
        self.tabview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.tabview.add("File Transcription")
        self.tabview.add("Live Transcription")

        # Set default tab
        self.tabview.set("File Transcription")

        # Get tab frames
        self.file_tab = self.tabview.tab("File Transcription")
        self.live_tab = self.tabview.tab("Live Transcription")

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

        self.root.configure(menu=menubar)

    def create_file_tab(self):
        """Create the UI for the file transcription tab."""
        # File selection section
        file_frame = ctk.CTkFrame(self.file_tab)
        file_frame.pack(fill=tk.X, pady=10, padx=10)

        ctk.CTkLabel(file_frame, text="Select Audio File", font=ctk.CTkFont(
            weight="bold")).pack(anchor=tk.W, pady=(5, 10))

        file_input_frame = ctk.CTkFrame(file_frame)
        file_input_frame.pack(fill=tk.X)

        self.file_path = tk.StringVar()
        ctk.CTkEntry(file_input_frame, textvariable=self.file_path, width=400).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ctk.CTkButton(file_input_frame, text="Browse", command=self.browse_file, width=100).pack(side=tk.RIGHT, padx=5)

        # Output file section
        output_frame = ctk.CTkFrame(self.file_tab)
        output_frame.pack(fill=tk.X, pady=10, padx=10)

        ctk.CTkLabel(output_frame, text="Output File (Optional)",
                     font=ctk.CTkFont(weight="bold")).pack(anchor=tk.W, pady=(5, 10))

        output_input_frame = ctk.CTkFrame(output_frame)
        output_input_frame.pack(fill=tk.X)

        self.output_path = tk.StringVar()
        ctk.CTkEntry(output_input_frame, textvariable=self.output_path,
                     width=400).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ctk.CTkButton(output_input_frame, text="Browse", command=self.browse_output,
                      width=100).pack(side=tk.RIGHT, padx=5)

        # Transcription section
        transcription_frame = ctk.CTkFrame(self.file_tab)
        transcription_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)

        ctk.CTkLabel(transcription_frame, text="Transcription",
                     font=ctk.CTkFont(weight="bold")).pack(anchor=tk.W, pady=(5, 10))

        # Transcription text area with scrollbar
        text_container = ctk.CTkFrame(transcription_frame)
        text_container.pack(fill=tk.BOTH, expand=True)

        self.transcription_text = ctk.CTkTextbox(text_container, wrap="word")
        self.transcription_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ctk.CTkProgressBar(self.file_tab, variable=self.progress_var)
        self.progress.pack(fill=tk.X, pady=10, padx=10)
        # self.progress.set(0)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ctk.CTkLabel(self.file_tab, textvariable=self.status_var)
        status_label.pack(anchor=tk.W, pady=5, padx=10)

        # Buttons
        button_frame = ctk.CTkFrame(self.file_tab)
        button_frame.pack(fill=tk.X, pady=10, padx=10)

        ctk.CTkButton(button_frame, text="Transcribe", command=self.start_transcription).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(button_frame, text="Save", command=self.save_transcription).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(button_frame, text="Clear", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(button_frame, text="Add Subtitles",
                      command=self.add_subtitles_to_video).pack(side=tk.LEFT, padx=5)

        # Add summarize button if summarizer is available
        if self.summarizer:
            ctk.CTkButton(button_frame, text="Summarize",
                          command=self.summarize_transcription).pack(side=tk.LEFT, padx=5)

    def create_live_tab(self):
        """Create the UI for the live transcription tab."""
        # Device selection frame
        device_frame = ctk.CTkFrame(self.live_tab)
        device_frame.pack(fill=tk.X, pady=10, padx=10)

        ctk.CTkLabel(device_frame, text="Audio Device Selection",
                     font=ctk.CTkFont(weight="bold")).pack(anchor=tk.W, pady=(5, 10))

        # Device selection grid
        device_grid = ctk.CTkFrame(device_frame)
        device_grid.pack(fill=tk.X)

        # Microphone selection
        ctk.CTkLabel(device_grid, text="Microphone:").grid(row=0, column=0, padx=(0, 5), pady=5, sticky=tk.W)
        self.mic_var = tk.StringVar()
        self.mic_dropdown = ctk.CTkOptionMenu(device_grid, variable=self.mic_var, values=[
                                              str(mic) for mic in self.live_audio_capture.get_available_mics()])
        if self.mic_dropdown._values:
            self.mic_var.set(self.mic_dropdown._values[0])
        self.mic_dropdown.grid(row=0, column=1, sticky=tk.EW, pady=5)
        
        # Microphone enable/disable button
        self.mic_toggle_button = ctk.CTkButton(device_grid, text="Disable", width=80, 
                                             command=self.toggle_mic_input)
        self.mic_toggle_button.grid(row=0, column=2, padx=5, pady=5)

        # System audio selection
        ctk.CTkLabel(device_grid, text="System Audio:").grid(row=1, column=0, padx=(0, 5), pady=5, sticky=tk.W)
        self.system_var = tk.StringVar()
        self.system_dropdown = ctk.CTkOptionMenu(device_grid, variable=self.system_var, values=[
                                                 str(dev) for dev in self.live_audio_capture.get_available_system_devices()])
        if self.system_dropdown._values:
            self.system_var.set(self.system_dropdown._values[0])
        self.system_dropdown.grid(row=1, column=1, sticky=tk.EW, pady=5)
        
        # System audio enable/disable button
        self.system_toggle_button = ctk.CTkButton(device_grid, text="Disable", width=80, 
                                                command=self.toggle_system_input)
        self.system_toggle_button.grid(row=1, column=2, padx=5, pady=5)
        
        # Refresh devices button
        self.refresh_button = ctk.CTkButton(device_grid, text="Refresh Devices", 
                                          command=self.refresh_audio_devices)
        self.refresh_button.grid(row=2, column=0, columnspan=3, padx=5, pady=10, sticky=tk.EW)
        
        # Device status label
        self.device_status_var = tk.StringVar(value="Audio devices ready")
        self.device_status_label = ctk.CTkLabel(device_grid, textvariable=self.device_status_var, 
                                              text_color="gray50")
        self.device_status_label.grid(row=3, column=0, columnspan=3, padx=5, pady=(0, 5), sticky=tk.W)

        # Configure grid weights
        device_grid.columnconfigure(1, weight=1)
        
        # Audio visualizer section
        visualizer_frame = ctk.CTkFrame(self.live_tab)
        visualizer_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ctk.CTkLabel(visualizer_frame, text="Audio Visualization",
                     font=ctk.CTkFont(weight="bold")).pack(anchor=tk.W, pady=(5, 10))
        
        # Create the dual audio visualizer
        self.audio_visualizer = DualAudioVisualizer(visualizer_frame, width=800, height=200)
        self.audio_visualizer.pack(fill=tk.X, padx=5, pady=5)

        # Live transcription section
        live_transcription_frame = ctk.CTkFrame(self.live_tab)
        live_transcription_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)

        ctk.CTkLabel(live_transcription_frame, text="Live Transcription",
                     font=ctk.CTkFont(weight="bold")).pack(anchor=tk.W, pady=(5, 10))

        # Add text widget for live transcription with scrolling capability
        # Create a container frame for the text widget
        text_container = ctk.CTkFrame(live_transcription_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        # Create the text widget with scrollbar support
        self.live_transcription_text = ctk.CTkTextbox(text_container, wrap="word")
        self.live_transcription_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Configure text colors (will need to be handled differently in CTkTextbox)
        # We'll use tags in the insert method

        # Status label
        self.live_status_var = tk.StringVar(value="Ready")
        live_status_label = ctk.CTkLabel(self.live_tab, textvariable=self.live_status_var)
        live_status_label.pack(anchor=tk.W, pady=5, padx=10)

        # Output file section
        output_frame = ctk.CTkFrame(self.live_tab)
        output_frame.pack(fill=tk.X, pady=10, padx=10)

        ctk.CTkLabel(output_frame, text="Output File",
                     font=ctk.CTkFont(weight="bold")).pack(anchor=tk.W, pady=(5, 10))

        output_input_frame = ctk.CTkFrame(output_frame)
        output_input_frame.pack(fill=tk.X)

        self.live_output_path = tk.StringVar()
        self.live_output_path.set("live_transcription.txt")
        ctk.CTkEntry(output_input_frame, textvariable=self.live_output_path,
                     width=400).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ctk.CTkButton(output_input_frame, text="Browse", command=self.browse_live_output,
                      width=100).pack(side=tk.RIGHT, padx=5)

        # Buttons
        live_button_frame = ctk.CTkFrame(self.live_tab)
        live_button_frame.pack(fill=tk.X, pady=10, padx=10)

        self.start_button = ctk.CTkButton(live_button_frame, text="Start Recording", command=self.toggle_recording)
        self.start_button.pack(side=tk.LEFT, padx=5)

        ctk.CTkButton(live_button_frame, text="Save Transcription",
                      command=self.save_live_transcription).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(live_button_frame, text="Clear", command=self.clear_live).pack(side=tk.LEFT, padx=5)

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
            
    def browse_live_output(self):
        """Open a file dialog to select an output file for live transcription."""
        filetypes = [
            ("Text Files", "*.txt"),
            ("All Files", "*.*")
        ]
        filename = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".txt")
        if filename:
            self.live_output_path.set(filename)

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
        self.progress_var.set(0.0)

        # Start transcription in a separate thread
        threading.Thread(target=self._transcribe_thread, daemon=True).start()

    def _transcribe_thread(self):
        """Run the transcription in a separate thread."""
        try:
            # Set up chunking and progress tracking
            file_path = self.file_path.get()
            self.total_chunks = 0
            self.processed_chunks = 0
            self.all_segments = []
            self.combined_text = ""

            # Create a queue for tracking progress
            self.chunk_progress_queue = queue.Queue()

            # Create a temporary directory for chunks
            with tempfile.TemporaryDirectory() as temp_dir:
                # Start real progress tracking thread
                progress_thread = threading.Thread(target=self._track_chunk_progress, daemon=True)
                progress_thread.start()

                # Split audio into 5-minute chunks (300 seconds)
                chunk_duration = 300  # seconds
                chunk_paths = self._split_audio_file(file_path, temp_dir, chunk_duration)

                self.total_chunks = len(chunk_paths)
                self.status_var.set(f"Transcribing {self.total_chunks} chunks...")

                # Process each chunk in parallel
                chunk_threads = []
                for i, chunk_path in enumerate(chunk_paths):
                    thread = threading.Thread(
                        target=self._process_chunk,
                        args=(chunk_path, i, len(chunk_paths)),
                        daemon=True
                    )
                    thread.start()
                    chunk_threads.append(thread)

                # Wait for all chunks to be processed
                for thread in chunk_threads:
                    thread.join()

                # Sort segments by start time
                self.all_segments.sort(key=lambda x: x.get("start", 0))

                # Update the UI with the result
                self.root.after(0, lambda: self._update_transcription(self.combined_text, self.all_segments))
        except Exception as e:
            traceback.print_exc()
            self._show_error(f"Transcription failed: {e}")

    def _split_audio_file(self, file_path, temp_dir, chunk_duration):
        """Split the audio file into smaller chunks.

        Args:
            file_path: Path to the audio file
            temp_dir: Directory to store the chunks
            chunk_duration: Duration of each chunk in seconds

        Returns:
            List of paths to the chunked audio files
        """
        try:
            # Get file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            is_video = file_ext == ".mp4"

            # Get file duration using ffmpeg
            probe = ffmpeg.probe(file_path)
            duration = float(probe['format']['duration'])

            chunk_paths = []
            # Create chunks based on duration
            for i, start_time in enumerate(range(0, int(duration), chunk_duration)):
                # Calculate end time (cap at total duration)
                end_time = min(start_time + chunk_duration, duration)

                # Create chunk file path
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")

                # Create the ffmpeg command to extract the chunk
                (ffmpeg
                    .input(file_path, ss=start_time, to=end_time)
                    .output(chunk_path, acodec='pcm_s16le', ac=1, ar='16k')
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                 )

                chunk_paths.append(chunk_path)

                # Update initial progress - normalized to 0-1 range
                self.root.after(0, lambda val=(5+int(5*i/max(1, len(range(0, int(duration), chunk_duration)))))/100:
                                self.progress_var.set(val))

            return chunk_paths
        except Exception as e:
            print(f"Error splitting audio: {e}")
            raise

    def _process_chunk(self, chunk_path, chunk_index, total_chunks):
        """Process a single audio chunk.

        Args:
            chunk_path: Path to the audio chunk
            chunk_index: Index of the chunk
            total_chunks: Total number of chunks
        """
        try:
            # Transcribe the chunk
            text, segments = self.transcriber.transcribe_file_with_timestamps(chunk_path)

            # Adjust timestamps for this chunk
            chunk_duration = 300  # seconds, same as in _split_audio_file
            start_offset = chunk_index * chunk_duration

            # Accurately adjust timestamps with precise offset calculation
            for segment in segments:
                if "start" in segment:
                    segment["start"] = float(segment["start"]) + start_offset
                if "end" in segment:
                    segment["end"] = float(segment["end"]) + start_offset

            # Add to all segments
            self.all_segments.extend(segments)

            # Append to combined text
            if text:
                self.combined_text += text + " "

            # Report progress
            self.chunk_progress_queue.put(1)

        except Exception as e:
            print(f"Error processing chunk {chunk_index}: {e}")
            # Still report progress even on error
            self.chunk_progress_queue.put(1)

    def _track_chunk_progress(self):
        """Track the progress of chunk transcription and update the progress bar."""
        processed = 0
        while processed < self.total_chunks:
            try:
                # Wait for a chunk to complete
                self.chunk_progress_queue.get(timeout=0.5)
                processed += 1
                self.processed_chunks = processed

                # Calculate progress percentage (10% for splitting + 90% for transcription)
                if self.total_chunks > 0:
                    progress = (10 + int(90 * processed / self.total_chunks)) / 100
                    # Update the progress bar - already normalized to 0-1 range
                    self.root.after(0, lambda val=progress: self.progress_var.set(val))
                    # Update status
                    self.root.after(0, lambda val=processed, total=self.total_chunks:
                                    self.status_var.set(f"Transcribed {val}/{total} chunks..."))
            except queue.Empty:
                pass

        # Make sure we reach 100% when all chunks are processed
        self.root.after(0, lambda: self.progress_var.set(1.0))

    def _update_transcription(self, text, segments=None):
        """Update the transcription text area with the result."""
        # Update status
        self.status_var.set("Transcription complete")
        self.progress_var.set(1.0)

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
            messagebox.showerror("Error", "Please specify an output file path.")
            return

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
        self.progress_var.set(0.0)

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
        self.progress_var.set(0.0)
        self.root.update()

        # Start subtitle addition in a separate thread
        threading.Thread(target=self._add_subtitles_thread, args=(output_path,), daemon=True).start()

    def _add_subtitles_thread(self, output_path):
        """Run the subtitle addition in a separate thread."""
        try:
            # Import the subtitle adder
            from subtitler import SubtitleAdder

            # Progress callback function to update the progress bar
            def update_progress(progress_value, message=None):
                # progress_value is already normalized to 0-1
                self.root.after(0, lambda val=progress_value: self.progress_var.set(val))
                # Update status message with percentage
                percent = int(progress_value * 100)
                self.root.after(0, lambda p=percent: self.status_var.set(
                    f"Adding subtitles... {p}%" if message is None else message))

            # Create subtitle adder with default style and progress callback
            subtitle_adder = SubtitleAdder(config=self.config, progress_callback=update_progress)

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
        self.progress_var.set(1.0)
        messagebox.showinfo("Success", f"Subtitles added successfully. Video saved to:\n{output_path}")

    def summarize_transcription(self):
        """Summarize the current transcription text."""
        if not self.summarizer:
            messagebox.showerror("Error", "Text summarization is not available.")
            return

        # Get the current transcription text
        text = self.transcription_text.get("0.0", "end").strip()
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
                summary_window = ctk.CTkToplevel(self.root)
                summary_window.title("Text Summary")
                summary_window.geometry("600x400")

                # Add text widget for summary
                summary_frame = ctk.CTkFrame(summary_window)
                summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                # Add title
                ctk.CTkLabel(summary_frame, text="Summary", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(0, 10))

                # Add text widget for summary
                summary_text = ctk.CTkTextbox(summary_frame, wrap="word")
                summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                # Insert summary
                summary_text.insert("0.0", summary)
                summary_text.configure(state="disabled")

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

                ctk.CTkButton(summary_frame, text="Save Summary", command=save_summary).pack(pady=10)
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

    def toggle_mic_input(self):
        """Toggle microphone input on/off."""
        self.mic_enabled = not self.mic_enabled
        if self.mic_enabled:
            self.mic_toggle_button.configure(text="Disable")
            self.mic_dropdown.configure(state="normal")
        else:
            self.mic_toggle_button.configure(text="Enable")
            self.mic_dropdown.configure(state="disabled")
    
    def toggle_system_input(self):
        """Toggle system audio input on/off."""
        self.system_enabled = not self.system_enabled
        if self.system_enabled:
            self.system_toggle_button.configure(text="Disable")
            self.system_dropdown.configure(state="normal")
        else:
            self.system_toggle_button.configure(text="Enable")
            self.system_dropdown.configure(state="disabled")
    
    def start_recording(self):
        """Start recording and transcribing audio."""
        try:
            # Check if at least one input is enabled
            if not self.mic_enabled and not self.system_enabled:
                messagebox.showwarning("Warning", "At least one audio input must be enabled.")
                return
                
            # Update UI
            self.live_status_var.set("Initializing...")
            self.start_button.configure(text="Stop Recording", state="disabled")
            self.root.update()
            
            # Reinitialize transcribers to pick up any config changes (like silence threshold)
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

            # Set selected devices
            selected_mic = None
            selected_system = None
            
            if self.mic_enabled:
                selected_mic = next((mic for mic in self.live_audio_capture.get_available_mics()
                                    if str(mic) == self.mic_var.get()), None)
            
            if self.system_enabled:
                selected_system = next(
                    (dev for dev in self.live_audio_capture.get_available_system_devices() 
                     if str(dev) == self.system_var.get()), None)

            self.live_audio_capture.set_mic_device(selected_mic)
            self.live_audio_capture.set_system_device(selected_system)

            # Start recording
            self.live_audio_capture.start_recording()
            self.is_recording = True

            # Start transcribers based on enabled inputs
            if self.mic_enabled:
                self.mic_transcriber.start_transcription()
            if self.system_enabled:
                self.system_transcriber.start_transcription()
            self.is_transcribing = True
            
            # Start audio visualizer
            self.audio_visualizer.start()

            # Start processing in a separate thread
            self.processing_thread = threading.Thread(target=self._process_audio, daemon=True)
            self.processing_thread.start()

            # Update UI
            self.live_status_var.set("Recording and transcribing...")
            self.start_button.configure(text="Stop Recording", state="normal")
            
            # Disable toggle buttons during recording
            self.mic_toggle_button.configure(state="disabled")
            self.system_toggle_button.configure(state="disabled")

        except Exception as e:
            self.live_status_var.set("Error")
            messagebox.showerror("Error", f"Failed to start recording: {e}")
            self.start_button.configure(text="Start Recording", state="normal")
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
                
            # Stop audio visualizer
            self.audio_visualizer.stop()

            # Update UI
            self.is_recording = False
            
            # Re-enable toggle buttons
            self.mic_toggle_button.configure(state="normal")
            self.system_toggle_button.configure(state="normal")
            self.live_status_var.set("Ready")
            self.start_button.configure(text="Start Recording", state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop recording: {e}")
            self.start_button.configure(state="normal")

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
        """Process audio from both channels and send to respective transcribers and visualizer"""
        # Track recording start time for timestamp calculation
        self.recording_start_time = time.time()

        while self.is_recording or not (self.live_audio_capture.mic_queue.empty() and self.live_audio_capture.system_queue.empty()):
            # Process microphone audio
            mic_frames = self.live_audio_capture.get_mic_audio()
            if mic_frames is not None and len(mic_frames) > 0:
                # Calculate current timestamp relative to recording start
                current_time = time.time() - self.recording_start_time
                self.mic_transcriber.add_audio_data(mic_frames)
                # Update mic visualizer
                self.audio_visualizer.update_mic_data(mic_frames)

            # Process system audio
            system_frames = self.live_audio_capture.get_system_audio()
            if system_frames is not None and len(system_frames) > 0:
                # Calculate current timestamp relative to recording start
                current_time = time.time() - self.recording_start_time
                self.system_transcriber.add_audio_data(system_frames)
                # Update system visualizer
                self.audio_visualizer.update_system_data(system_frames)

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

            # Insert with color formatting
            if self.live_transcription_text.get("0.0", "end-1c") != "":
                self.live_transcription_text.insert("end", "\n")

            # Insert with color (blue for mic)
            text_color = "#0000FF" if color == "blue" else "#000000"
            self.live_transcription_text.insert("end", f"Mic: {transcription}", {"text_color": text_color})
            self.live_transcription_text.see("end")

        # Process system transcriptions
        while not self.system_queue.empty():
            transcription, color = self.system_queue.get()

            # Insert with color formatting
            if self.live_transcription_text.get("0.0", "end-1c") != "":
                self.live_transcription_text.insert("end", "\n")

            # Insert with color (green for system)
            text_color = "#00AA00" if color == "green" else "#000000"
            self.live_transcription_text.insert("end", f"System: {transcription}", {"text_color": text_color})
            self.live_transcription_text.see("end")

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
        settings_window = ctk.CTkToplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("600x500")
        settings_window.minsize(500, 400)
        settings_window.transient(self.root)
        settings_window.grab_set()

        # Create a tabview for tabs
        tabview = ctk.CTkTabview(settings_window)
        tabview.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs for different settings
        tabview.add("General")
        tabview.add("Vosk")
        tabview.add("Whisper")
        tabview.add("Subtitles")
        tabview.add("Summarizer")

        # Set default tab
        tabview.set("General")

        # Get tab frames
        general_tab = tabview.tab("General")
        vosk_tab = tabview.tab("Vosk")
        whisper_tab = tabview.tab("Whisper")
        subtitles_tab = tabview.tab("Subtitles")
        summarizer_tab = tabview.tab("Summarizer")

        # General settings
        general_frame = ctk.CTkFrame(general_tab)
        general_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ctk.CTkLabel(general_frame, text="Engine:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        engine_var = tk.StringVar(value=self.config.get_engine())
        engine_combo = ctk.CTkOptionMenu(general_frame, variable=engine_var,
                                         values=TranscriberFactory.get_available_engines())
        engine_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Silence detection settings
        ctk.CTkLabel(general_frame, text="Silence Threshold:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        silence_threshold_var = tk.StringVar(value=str(self.config.get("silence_threshold", 0.01)))
        silence_threshold_entry = ctk.CTkEntry(general_frame, textvariable=silence_threshold_var, width=100)
        silence_threshold_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)
        ctk.CTkLabel(general_frame, text="(Lower values = more sensitive, 0.005-0.02 recommended)").grid(
            row=1, column=2, sticky=tk.W, pady=5, padx=5)

        # Vosk settings
        vosk_frame = ctk.CTkFrame(vosk_tab)
        vosk_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ctk.CTkLabel(vosk_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        vosk_model_path = tk.StringVar(value=self.config.get("models.vosk.model_path", ""))
        ctk.CTkEntry(vosk_frame, textvariable=vosk_model_path, width=300).grid(
            row=0, column=1, sticky=tk.W, pady=5, padx=5)
        ctk.CTkButton(vosk_frame, text="Browse", command=lambda: self._browse_model_dir(
            vosk_model_path), width=80).grid(row=0, column=2, padx=5, pady=5)

        # Whisper settings
        whisper_frame = ctk.CTkFrame(whisper_tab)
        whisper_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ctk.CTkLabel(whisper_frame, text="Model Size:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        whisper_model_size = tk.StringVar(value=self.config.get("models.whisper.model_size", "base"))
        size_combo = ctk.CTkOptionMenu(whisper_frame, variable=whisper_model_size, values=[
                                       "tiny", "base", "small", "medium", "large"])
        size_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)

        ctk.CTkLabel(whisper_frame, text="Use GPU:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        use_gpu = tk.BooleanVar(value=self.config.get("models.whisper.use_gpu", True))
        ctk.CTkSwitch(whisper_frame, text="", variable=use_gpu).grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)

        ctk.CTkLabel(whisper_frame, text="Language:").grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        language = tk.StringVar(value=self.config.get("models.whisper.language", ""))
        ctk.CTkEntry(whisper_frame, textvariable=language, width=100).grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)
        ctk.CTkLabel(whisper_frame, text="(Leave empty for auto-detection)").grid(row=2,
                                                                                  column=2, sticky=tk.W, pady=5, padx=5)

        # Subtitles settings
        subtitles_frame = ctk.CTkFrame(subtitles_tab)
        subtitles_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Default style selection
        ctk.CTkLabel(subtitles_frame, text="Default Style:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        default_style = tk.StringVar(value=self.config.get("subtitles.default_style", "default"))
        style_combo = ctk.CTkOptionMenu(subtitles_frame, variable=default_style, values=["default", "youtube"])
        style_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)

        # Style settings section
        style_settings_frame = ctk.CTkFrame(subtitles_frame)
        style_settings_frame.grid(row=1, column=0, columnspan=3, sticky=tk.NSEW, pady=10, padx=5)
        subtitles_frame.columnconfigure(2, weight=1)

        # Default style settings
        ctk.CTkLabel(style_settings_frame, text="Default Style Settings", font=ctk.CTkFont(
            weight="bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)

        # Font
        ctk.CTkLabel(style_settings_frame, text="Font:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        default_font = tk.StringVar(value=self.config.get("subtitles.styles.default.font", "Arial"))
        ctk.CTkEntry(style_settings_frame, textvariable=default_font, width=150).grid(
            row=1, column=1, sticky=tk.W, pady=5, padx=5)

        # Font size
        ctk.CTkLabel(style_settings_frame, text="Font Size:").grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        default_fontsize = tk.StringVar(value=str(self.config.get("subtitles.styles.default.fontsize", 40)))
        ctk.CTkEntry(style_settings_frame, textvariable=default_fontsize,
                     width=50).grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)

        # Text color
        ctk.CTkLabel(style_settings_frame, text="Text Color:").grid(row=3, column=0, sticky=tk.W, pady=5, padx=5)
        default_color = tk.StringVar(value=self.config.get("subtitles.styles.default.color", "white"))
        ctk.CTkEntry(style_settings_frame, textvariable=default_color, width=100).grid(
            row=3, column=1, sticky=tk.W, pady=5, padx=5)

        # Stroke color
        ctk.CTkLabel(style_settings_frame, text="Stroke Color:").grid(row=4, column=0, sticky=tk.W, pady=5, padx=5)
        default_stroke_color = tk.StringVar(value=self.config.get("subtitles.styles.default.stroke_color", "black"))
        ctk.CTkEntry(style_settings_frame, textvariable=default_stroke_color,
                     width=100).grid(row=4, column=1, sticky=tk.W, pady=5, padx=5)

        # Stroke width
        ctk.CTkLabel(style_settings_frame, text="Stroke Width:").grid(row=5, column=0, sticky=tk.W, pady=5, padx=5)
        default_stroke_width = tk.StringVar(value=str(self.config.get("subtitles.styles.default.stroke_width", 1.5)))
        ctk.CTkEntry(style_settings_frame, textvariable=default_stroke_width,
                     width=50).grid(row=5, column=1, sticky=tk.W, pady=5, padx=5)

        # Background color
        ctk.CTkLabel(style_settings_frame, text="Background Color:").grid(row=6, column=0, sticky=tk.W, pady=5, padx=5)
        default_bg_color = tk.StringVar(value=self.config.get("subtitles.styles.default.bg_color", "transparent"))
        ctk.CTkEntry(style_settings_frame, textvariable=default_bg_color,
                     width=100).grid(row=6, column=1, sticky=tk.W, pady=5, padx=5)

        # YouTube style settings
        youtube_settings_frame = ctk.CTkFrame(subtitles_frame)
        youtube_settings_frame.grid(row=2, column=0, columnspan=3, sticky=tk.NSEW, pady=10, padx=5)

        ctk.CTkLabel(youtube_settings_frame, text="YouTube Style Settings", font=ctk.CTkFont(
            weight="bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)

        # Font
        ctk.CTkLabel(youtube_settings_frame, text="Font:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        youtube_font = tk.StringVar(value=self.config.get("subtitles.styles.youtube.font", "Helvetica-BoldOblique"))
        ctk.CTkEntry(youtube_settings_frame, textvariable=youtube_font,
                     width=150).grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)

        # Font size
        ctk.CTkLabel(youtube_settings_frame, text="Font Size:").grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        youtube_fontsize = tk.StringVar(value=str(self.config.get("subtitles.styles.youtube.fontsize", 45)))
        ctk.CTkEntry(youtube_settings_frame, textvariable=youtube_fontsize,
                     width=50).grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)

        # Text color
        ctk.CTkLabel(youtube_settings_frame, text="Text Color:").grid(row=3, column=0, sticky=tk.W, pady=5, padx=5)
        youtube_color = tk.StringVar(value=self.config.get("subtitles.styles.youtube.color", "white"))
        ctk.CTkEntry(youtube_settings_frame, textvariable=youtube_color,
                     width=100).grid(row=3, column=1, sticky=tk.W, pady=5, padx=5)

        # Stroke color
        ctk.CTkLabel(youtube_settings_frame, text="Stroke Color:").grid(row=4, column=0, sticky=tk.W, pady=5, padx=5)
        youtube_stroke_color = tk.StringVar(value=self.config.get("subtitles.styles.youtube.stroke_color", "black"))
        ctk.CTkEntry(youtube_settings_frame, textvariable=youtube_stroke_color,
                     width=100).grid(row=4, column=1, sticky=tk.W, pady=5, padx=5)

        # Stroke width
        ctk.CTkLabel(youtube_settings_frame, text="Stroke Width:").grid(row=5, column=0, sticky=tk.W, pady=5, padx=5)
        youtube_stroke_width = tk.StringVar(value=str(self.config.get("subtitles.styles.youtube.stroke_width", 1.5)))
        ctk.CTkEntry(youtube_settings_frame, textvariable=youtube_stroke_width,
                     width=50).grid(row=5, column=1, sticky=tk.W, pady=5, padx=5)

        # Background color
        ctk.CTkLabel(youtube_settings_frame, text="Background Color:").grid(
            row=6, column=0, sticky=tk.W, pady=5, padx=5)
        youtube_bg_color = tk.StringVar(value=self.config.get("subtitles.styles.youtube.bg_color", "transparent"))
        ctk.CTkEntry(youtube_settings_frame, textvariable=youtube_bg_color,
                     width=100).grid(row=6, column=1, sticky=tk.W, pady=5, padx=5)

        # Summarizer settings
        summarizer_frame = ctk.CTkFrame(summarizer_tab)
        summarizer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Model name
        ctk.CTkLabel(summarizer_frame, text="Model Name:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        model_name = tk.StringVar(value=self.config.get("summarizer.model_name", "mistral"))
        ctk.CTkEntry(summarizer_frame, textvariable=model_name, width=150).grid(
            row=0, column=1, sticky=tk.W, pady=5, padx=5)

        # Base URL
        ctk.CTkLabel(summarizer_frame, text="Base URL:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        base_url = tk.StringVar(value=self.config.get("summarizer.base_url", "http://localhost:11434"))
        ctk.CTkEntry(summarizer_frame, textvariable=base_url, width=250).grid(
            row=1, column=1, sticky=tk.W, pady=5, padx=5)

        # Buttons
        button_frame = ctk.CTkFrame(settings_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ctk.CTkButton(button_frame, text="Save", command=lambda: self._save_settings(
            engine_var.get(),
            vosk_model_path.get(),
            whisper_model_size.get(),
            use_gpu.get(),
            language.get(),
            default_style.get(),
            default_font.get(),
            default_fontsize.get(),
            default_color.get(),
            default_stroke_color.get(),
            default_stroke_width.get(),
            default_bg_color.get(),
            youtube_font.get(),
            youtube_fontsize.get(),
            youtube_color.get(),
            youtube_stroke_color.get(),
            youtube_stroke_width.get(),
            youtube_bg_color.get(),
            model_name.get(),
            base_url.get(),
            settings_window,
            silence_threshold_var.get()
        )).pack(side=tk.RIGHT, padx=5)

        ctk.CTkButton(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.RIGHT, padx=5)

    def _browse_model_dir(self, path_var):
        """Open a directory dialog to select a model directory."""
        directory = filedialog.askdirectory()
        if directory:
            path_var.set(directory)

    def _save_settings(self, engine, vosk_model_path, whisper_model_size, use_gpu, language,
                       default_style, default_font, default_fontsize, default_color, default_stroke_color,
                       default_stroke_width, default_bg_color, youtube_font, youtube_fontsize,
                       youtube_color, youtube_stroke_color, youtube_stroke_width, youtube_bg_color,
                       model_name, base_url, window, silence_threshold=None):
        """Save the settings and close the dialog."""
        # Update the configuration
        self.config.set("engine", engine)
        self.config.set("models.vosk.model_path", vosk_model_path)
        self.config.set("models.whisper.model_size", whisper_model_size)
        self.config.set("models.whisper.use_gpu", use_gpu)
        self.config.set("models.whisper.language", language)
        
        # Update silence threshold if provided
        if silence_threshold is not None:
            try:
                # Convert to float and ensure it's a valid value
                threshold_value = float(silence_threshold)
                if threshold_value > 0:
                    self.config.set("silence_threshold", threshold_value)
            except ValueError:
                # If conversion fails, keep the existing value
                pass

        # Update subtitle settings
        self.config.set("subtitles.default_style", default_style)
        self.config.set("subtitles.styles.default.font", default_font)
        self.config.set("subtitles.styles.default.fontsize", int(default_fontsize))
        self.config.set("subtitles.styles.default.color", default_color)
        self.config.set("subtitles.styles.default.stroke_color", default_stroke_color)
        self.config.set("subtitles.styles.default.stroke_width", float(default_stroke_width))
        self.config.set("subtitles.styles.default.bg_color", default_bg_color)

        self.config.set("subtitles.styles.youtube.font", youtube_font)
        self.config.set("subtitles.styles.youtube.fontsize", int(youtube_fontsize))
        self.config.set("subtitles.styles.youtube.color", youtube_color)
        self.config.set("subtitles.styles.youtube.stroke_color", youtube_stroke_color)
        self.config.set("subtitles.styles.youtube.stroke_width", float(youtube_stroke_width))
        self.config.set("subtitles.styles.youtube.bg_color", youtube_bg_color)

        # Update summarizer settings
        self.config.set("summarizer.model_name", model_name)
        self.config.set("summarizer.base_url", base_url)

        # Save the configuration
        self.config.save()

        # Close the window
        window.destroy()

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
        about_window = ctk.CTkToplevel(self.root)
        about_window.title("About Audio Transcriber")
        about_window.geometry("400x300")
        about_window.transient(self.root)
        about_window.grab_set()

        # Add about text
        about_frame = ctk.CTkFrame(about_window)
        about_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ctk.CTkLabel(about_frame, text="Audio Transcriber", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        ctk.CTkLabel(about_frame, text="A tool for transcribing audio from MP4 or WAV files\nand capturing live audio for real-time transcription.",
                     font=ctk.CTkFont(size=14)).pack(pady=10)
        ctk.CTkLabel(about_frame, text="Supported engines: Vosk, Whisper", font=ctk.CTkFont(size=14)).pack(pady=10)

        # Close button
        ctk.CTkButton(about_frame, text="Close", command=about_window.destroy).pack(pady=10)

    def start_device_monitoring(self):
        """Start the device monitoring thread to detect device changes."""
        self.device_monitor_active = True
        self.device_monitor_thread = threading.Thread(target=self._monitor_devices, daemon=True)
        self.device_monitor_thread.start()
    
    def stop_device_monitoring(self):
        """Stop the device monitoring thread."""
        self.device_monitor_active = False
        if hasattr(self, 'device_monitor_thread') and self.device_monitor_thread.is_alive():
            self.device_monitor_thread.join(timeout=1.0)
    
    def _monitor_devices(self):
        """Monitor for changes in audio devices."""
        while self.device_monitor_active:
            # Get current devices
            current_mics = set(str(mic) for mic in self.live_audio_capture.get_available_mics())
            current_system_devices = set(str(dev) for dev in self.live_audio_capture.get_available_system_devices())
            
            # Check for changes
            mics_added = current_mics - self.last_known_mics
            mics_removed = self.last_known_mics - current_mics
            system_added = current_system_devices - self.last_known_system_devices
            system_removed = self.last_known_system_devices - current_system_devices
            
            # If changes detected
            if mics_added or mics_removed or system_added or system_removed:
                self.device_change_detected = True
                
                # Update UI from main thread
                self.root.after(0, lambda: self._handle_device_changes(
                    mics_added, mics_removed, system_added, system_removed))
                
                # Update last known devices
                self.last_known_mics = current_mics
                self.last_known_system_devices = current_system_devices
            
            # Check every 2 seconds
            time.sleep(2)
    
    def _handle_device_changes(self, mics_added, mics_removed, system_added, system_removed):
        """Handle device changes by updating UI and notifying user."""
        # Update device dropdowns
        self._update_device_dropdowns()
        
        # Create notification message
        messages = []
        if mics_added:
            messages.append(f"New microphone(s) connected: {', '.join(mics_added)}")
        if mics_removed:
            messages.append(f"Microphone(s) disconnected: {', '.join(mics_removed)}")
        if system_added:
            messages.append(f"New system audio device(s) connected: {', '.join(system_added)}")
        if system_removed:
            messages.append(f"System audio device(s) disconnected: {', '.join(system_removed)}")
        
        # Show notification if recording
        if self.is_recording:
            # Check if the active device was disconnected
            active_mic = self.mic_var.get()
            active_system = self.system_var.get()
            
            if (self.mic_enabled and active_mic in mics_removed) or \
               (self.system_enabled and active_system in system_removed):
                # Stop recording temporarily
                was_recording = self.is_recording
                if was_recording:
                    self.stop_recording()
                
                # Automatically switch to the first available device (index 0)
                if self.mic_enabled and active_mic in mics_removed and self.mic_dropdown._values:
                    self.mic_var.set(self.mic_dropdown._values[0])
                
                if self.system_enabled and active_system in system_removed and self.system_dropdown._values:
                    self.system_var.set(self.system_dropdown._values[0])
                
                # Restart recording with new devices
                if was_recording:
                    self.start_recording()
                    
                # Show notification about automatic device switch
                message = "\n".join(messages)
                message += "\n\nAutomatically switched to default audio devices and resumed recording."
                messagebox.showinfo("Audio Device Change Detected", message)
            else:
                # Just show notification without stopping
                message = "\n".join(messages)
                message += "\n\nYou can continue recording with current devices or stop and select new ones."
                messagebox.showinfo("Audio Device Change Detected", message)
        else:
            # Ask user if they want to use newly connected devices
            if mics_added or system_added:
                self._prompt_for_new_devices(mics_added, system_added)
    
    def refresh_audio_devices(self):
        """Manually refresh the audio device lists."""
        # Update status
        self.device_status_var.set("Refreshing audio devices...")
        self.refresh_button.configure(state="disabled")
        self.root.update()
        
        try:
            # Get current devices
            current_mics = set(str(mic) for mic in self.live_audio_capture.get_available_mics())
            current_system_devices = set(str(dev) for dev in self.live_audio_capture.get_available_system_devices())
            
            # Check for changes
            mics_added = current_mics - self.last_known_mics
            mics_removed = self.last_known_mics - current_mics
            system_added = current_system_devices - self.last_known_system_devices
            system_removed = self.last_known_system_devices - current_system_devices
            
            # Update dropdowns
            self._update_device_dropdowns()
            
            # Update last known devices
            self.last_known_mics = current_mics
            self.last_known_system_devices = current_system_devices
            
            # Update status with changes
            if mics_added or mics_removed or system_added or system_removed:
                changes = []
                if mics_added:
                    changes.append(f"{len(mics_added)} new mic(s)")
                if mics_removed:
                    changes.append(f"{len(mics_removed)} mic(s) removed")
                if system_added:
                    changes.append(f"{len(system_added)} new system device(s)")
                if system_removed:
                    changes.append(f"{len(system_removed)} system device(s) removed")
                
                self.device_status_var.set(f"Devices updated: {', '.join(changes)}")
            else:
                self.device_status_var.set("No device changes detected")
        except Exception as e:
            self.device_status_var.set(f"Error refreshing devices: {e}")
        finally:
            # Re-enable refresh button
            self.refresh_button.configure(state="normal")
    
    def _update_device_dropdowns(self):
        """Update the device dropdown menus with current devices."""
        # Update microphone dropdown
        current_mic = self.mic_var.get() if self.mic_var.get() in self.last_known_mics else ""
        self.mic_dropdown.configure(values=[str(mic) for mic in self.live_audio_capture.get_available_mics()])
        if self.mic_dropdown._values and not current_mic:
            self.mic_var.set(self.mic_dropdown._values[0])
        elif current_mic:
            self.mic_var.set(current_mic)
        
        # Update system audio dropdown
        current_system = self.system_var.get() if self.system_var.get() in self.last_known_system_devices else ""
        self.system_dropdown.configure(values=[str(dev) for dev in self.live_audio_capture.get_available_system_devices()])
        if self.system_dropdown._values and not current_system:
            self.system_var.set(self.system_dropdown._values[0])
        elif current_system:
            self.system_var.set(current_system)
            
    def _prompt_for_new_devices(self, mics_added, system_added):
        """Show a dialog asking if the user wants to use newly connected devices.
        
        Args:
            mics_added: Set of newly added microphones
            system_added: Set of newly added system audio devices
        """
        # Create message about new devices
        message = "New audio devices detected:\n\n"
        
        new_mic = None
        new_system = None
        
        if mics_added:
            message += f"New microphone(s): {', '.join(mics_added)}\n"
            new_mic = list(mics_added)[0]  # Get the first new microphone
            
        if system_added:
            message += f"New system audio device(s): {', '.join(system_added)}\n"
            new_system = list(system_added)[0]  # Get the first new system device
            
        message += "\nWould you like to use the newly connected device(s)?"
        
        # Show confirmation dialog
        if messagebox.askyesno("New Audio Device Detected", message):
            # User confirmed, switch to new devices
            was_recording = False
            
            # Stop recording if active
            if self.is_recording:
                was_recording = True
                self.stop_recording()
            
            # Switch to new devices
            if new_mic and self.mic_enabled:
                self.mic_var.set(new_mic)
                self.device_status_var.set(f"Switched to new microphone: {new_mic}")
                
            if new_system and self.system_enabled:
                self.system_var.set(new_system)
                if new_mic:
                    self.device_status_var.set(f"Switched to new devices")
                else:
                    self.device_status_var.set(f"Switched to new system audio: {new_system}")
            
            # Restart recording if it was active
            if was_recording:
                self.root.after(500, self.toggle_recording)  # Small delay before restarting
    
    def on_close(self):
        """Handle window close event."""
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
            
        # Stop device monitoring
        self.stop_device_monitoring()

        # Clean up resources
        self._cleanup_resources()

        # Destroy the window
        self.root.destroy()


def main():
    """Main function to run the application."""
    root = ctk.CTk()
    app = TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
