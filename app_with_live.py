import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import queue
import time
import os

# Import our custom modules
from live_audio_capture import AudioRecorder
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
        self.recorder = AudioRecorder()
        
        # Create transcriber with callback for transcriptions
        self.transcription_queue = queue.Queue()
        self.transcriber = LiveTranscriber(
            config=self.config,
            transcription_callback=self.handle_transcription
        )
        
        # Set up the UI
        self.setup_ui()
        
        # Start the queue processing
        self.process_transcription_queue()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
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
        
        # Start recording
        self.recorder.start_recording()
        
        # Start the transcriber
        self.transcriber.start_transcription()
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_transcription(self):
        self.recorder.stop_recording()
        self.transcriber.stop_transcription()
        
        self.stop_button.state(["disabled"])
        self.start_button.state(["!disabled"])
        self.status_label.config(text=f"Status: Ready")
    
    def clear_transcription(self):
        self.transcription_text.config(state=tk.NORMAL)
        self.transcription_text.delete(1.0, tk.END)
        self.transcription_text.config(state=tk.DISABLED)
    
    def handle_transcription(self, text):
        """Callback function for transcription results"""
        if text:
            self.transcription_queue.put(text)
    
    def process_audio(self):
        """Process audio from the recorder and send to transcriber"""
        while self.recorder.is_recording or not self.recorder.audio_queue.empty():
            frames = self.recorder.get_audio()
            if frames is not None and len(frames) > 0:  # Check if frames exist and contain data
                # Send audio to transcriber
                self.transcriber.add_audio_data(frames)
            
            time.sleep(0.1)
    
    def process_transcription_queue(self):
        """Process transcriptions from the queue and update UI"""
        # Check for transcriptions from the callback queue
        if not self.transcription_queue.empty():
            transcription = self.transcription_queue.get()
            
            self.transcription_text.config(state=tk.NORMAL)
            if self.transcription_text.index('end-1c') != '1.0':
                self.transcription_text.insert(tk.END, " ")
            self.transcription_text.insert(tk.END, transcription)
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