import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import queue
import time
import os

# Import our custom modules
from audio_recorder1 import AudioRecorder2
from live_transcriber import LiveTranscriber
from config import get_config

class LiveTranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Audio Transcription")
        self.root.geometry("800x600")
        
        # Get configuration
        self.config = get_config().config
        
        # Initialize our modules
        self.recorder = AudioRecorder2()
        
        # Get available devices
        self.available_mics = self.recorder.get_available_mics()
        self.available_system_devices = self.recorder.get_available_system_devices()
        
        # Create separate transcribers for mic and system audio
        self.mic_queue = queue.Queue()
        self.system_queue = queue.Queue()
        
        # Show loading dialog
        loading_dialog = tk.Toplevel(self.root)
        loading_dialog.title("Loading Models")
        loading_dialog.geometry("300x150")
        loading_dialog.transient(self.root)
        loading_dialog.grab_set()
        
        # Center the dialog
        loading_dialog.geometry("+%d+%d" % (
            self.root.winfo_x() + self.root.winfo_width()/2 - 150,
            self.root.winfo_y() + self.root.winfo_height()/2 - 75
        ))
        
        # Add loading message and progress bar
        ttk.Label(loading_dialog, text="Loading transcription models...", padding=10).pack()
        progress = ttk.Progressbar(loading_dialog, mode='indeterminate')
        progress.pack(padx=20, pady=10, fill=tk.X)
        progress.start()
        
        # Update the dialog
        loading_dialog.update()
        
        # Initialize transcribers
        self.mic_transcriber = LiveTranscriber(
            config=self.config,
            transcription_callback=self.handle_mic_transcription
        )
        
        self.system_transcriber = LiveTranscriber(
            config=self.config,
            transcription_callback=self.handle_system_transcription
        )
        
        # Load models
        self.mic_transcriber.load_model()
        self.system_transcriber.load_model()
        
        # Close the dialog
        loading_dialog.destroy()
        
        # Set up the UI
        self.setup_ui()
        
        # Start the queue processing
        self.process_transcription_queue()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Device selection frame
        device_frame = ttk.LabelFrame(main_frame, text="Audio Device Selection", padding="10")
        device_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Microphone selection
        ttk.Label(device_frame, text="Microphone:").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        self.mic_var = tk.StringVar()
        self.mic_dropdown = ttk.Combobox(device_frame, textvariable=self.mic_var, state="readonly")
        self.mic_dropdown['values'] = [str(mic) for mic in self.available_mics]
        if self.available_mics:
            self.mic_dropdown.current(0)
        self.mic_dropdown.grid(row=0, column=1, sticky=tk.EW)
        
        # System audio selection
        ttk.Label(device_frame, text="System Audio:").grid(row=1, column=0, padx=(0, 5), sticky=tk.W)
        self.system_var = tk.StringVar()
        self.system_dropdown = ttk.Combobox(device_frame, textvariable=self.system_var, state="readonly")
        self.system_dropdown['values'] = [str(dev) for dev in self.available_system_devices]
        if self.available_system_devices:
            self.system_dropdown.current(0)
        self.system_dropdown.grid(row=1, column=1, sticky=tk.EW)
        
        # Configure grid weights
        device_frame.columnconfigure(1, weight=1)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Status: Ready")
        self.status_label.pack(pady=5, anchor=tk.W)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Transcription", command=self.start_transcription)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Transcription", command=self.stop_transcription)
        self.stop_button.pack(side=tk.LEFT)
        self.stop_button.state(["disabled"])
        
        # Clear button
        self.clear_button = ttk.Button(button_frame, text="Clear Transcription", command=self.clear_transcription)
        self.clear_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Transcription display
        ttk.Label(main_frame, text="Transcription:").pack(anchor=tk.W, pady=(10, 5))
        
        self.transcription_text = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=20)
        self.transcription_text.pack(fill=tk.BOTH, expand=True)
        self.transcription_text.config(state=tk.DISABLED)
    
    def start_transcription(self):
        self.start_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self.status_label.config(text="Status: Transcribing...")
        
        # Configure text colors
        self.transcription_text.tag_configure("blue", foreground="blue")
        self.transcription_text.tag_configure("green", foreground="green")
        
        # Set selected devices
        selected_mic = next((mic for mic in self.available_mics if str(mic) == self.mic_var.get()), None)
        selected_system = next((dev for dev in self.available_system_devices if str(dev) == self.system_var.get()), None)
        
        self.recorder.set_mic_device(selected_mic)
        self.recorder.set_system_device(selected_system)
        
        # Start recording
        self.recorder.start_recording()
        
        # Start both transcribers
        self.mic_transcriber.start_transcription()
        self.system_transcriber.start_transcription()
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_transcription(self):
        self.recorder.stop_recording()
        self.mic_transcriber.stop_transcription()
        self.system_transcriber.stop_transcription()
        
        self.stop_button.state(["disabled"])
        self.start_button.state(["!disabled"])
        self.status_label.config(text=f"Status: Ready")
    
    def clear_transcription(self):
        self.transcription_text.config(state=tk.NORMAL)
        self.transcription_text.delete(1.0, tk.END)
        self.transcription_text.config(state=tk.DISABLED)
    
    def handle_mic_transcription(self, text):
        """Callback function for microphone transcription results"""
        if text:
            self.mic_queue.put((text, "blue"))
    
    def handle_system_transcription(self, text):
        """Callback function for system audio transcription results"""
        if text:
            self.system_queue.put((text, "green"))
    
    def process_audio(self):
        """Process audio from both channels and send to respective transcribers"""
        while self.recorder.is_recording or not (self.recorder.mic_queue.empty() and self.recorder.system_queue.empty()):
            # Process microphone audio
            mic_frames = self.recorder.get_mic_audio()
            if mic_frames is not None and len(mic_frames) > 0:
                self.mic_transcriber.add_audio_data(mic_frames)
            
            # Process system audio
            system_frames = self.recorder.get_system_audio()
            if system_frames is not None and len(system_frames) > 0:
                self.system_transcriber.add_audio_data(system_frames)
            
            time.sleep(0.1)
    
    def process_transcription_queue(self):
        """Process transcriptions from both queues and update UI with colored text"""
        # Process microphone transcriptions
        if not self.mic_queue.empty():
            transcription, color = self.mic_queue.get()
            
            self.transcription_text.config(state=tk.NORMAL)
            if self.transcription_text.index('end-1c') != '1.0':
                self.transcription_text.insert(tk.END, " ")
            self.transcription_text.insert(tk.END, transcription, color)
            self.transcription_text.see(tk.END)
            self.transcription_text.config(state=tk.DISABLED)
        
        # Process system transcriptions
        if not self.system_queue.empty():
            transcription, color = self.system_queue.get()
            
            self.transcription_text.config(state=tk.NORMAL)
            if self.transcription_text.index('end-1c') != '1.0':
                self.transcription_text.insert(tk.END, " ")
            self.transcription_text.insert(tk.END, transcription, color)
            self.transcription_text.see(tk.END)
            self.transcription_text.config(state=tk.DISABLED)
        
        # # Also check for transcriptions directly from the transcriber
        # transcription = self.transcriber.get_next_transcription()
        # if transcription:
        #     self.transcription_text.config(state=tk.NORMAL)
        #     if self.transcription_text.index('end-1c') != '1.0':
        #         self.transcription_text.insert(tk.END, " ")
        #     self.transcription_text.insert(tk.END, transcription)
        #     self.transcription_text.see(tk.END)
        #     self.transcription_text.config(state=tk.DISABLED)
        
        # Schedule the next check
        self.root.after(100, self.process_transcription_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveTranscriptionApp(root)
    root.mainloop()