import numpy as np
import tkinter as tk
from typing import Optional, Dict, Any, Tuple, List, Callable
import threading
import time

class AudioVisualizer:
    """Class for visualizing audio data in real-time.
    
    This class provides functionality to visualize audio data from system output
    and microphone input in real-time using a tkinter canvas.
    """
    
    def __init__(self, master: tk.Widget, width: int = 400, height: int = 100, 
                 bg_color: str = "#000000", line_color: str = "#00FF00",
                 line_width: int = 2, update_interval_ms: int = 50):
        """Initialize the audio visualizer.
        
        Args:
            master: The tkinter widget to place the visualizer in
            width: Width of the visualizer canvas
            height: Height of the visualizer canvas
            bg_color: Background color of the canvas
            line_color: Color of the visualization line
            line_width: Width of the visualization line
            update_interval_ms: Interval between visualization updates in milliseconds
        """
        self.master = master
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.line_color = line_color
        self.line_width = line_width
        self.update_interval_ms = update_interval_ms
        
        # Create the canvas for visualization
        self.canvas = tk.Canvas(master, width=width, height=height, bg=bg_color, 
                               highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initialize visualization data
        self.audio_data = np.zeros(width)
        self.is_active = False
        self.update_id = None
        
        # Create the visualization line
        self.line_id = None
        self._create_line()
        
        # Bind to window resize events
        self.master.bind('<Configure>', self._on_resize)
    
    def _create_line(self):
        """Create the initial visualization line."""
        # Create points for the line (centered vertically)
        points = []
        mid_y = self.height // 2
        
        for x in range(self.width):
            points.extend([x, mid_y])
        
        # Create the line on the canvas
        self.line_id = self.canvas.create_line(
            *points, fill=self.line_color, width=self.line_width, smooth=True
        )
    
    def update_data(self, audio_data: np.ndarray):
        """Update the audio data for visualization.
        
        Args:
            audio_data: Audio data as numpy array (should be 1D)
        """
        # Ensure we have 1D data
        if audio_data.ndim > 1:
            # If stereo or more channels, take the mean across channels
            audio_data = np.mean(audio_data, axis=1)
        
        # Resize data to match canvas width if needed
        if len(audio_data) != self.width:
            # Resample to match canvas width
            indices = np.linspace(0, len(audio_data) - 1, self.width, dtype=int)
            audio_data = audio_data[indices]
        
        # Normalize the data to fit in the canvas height
        if len(audio_data) > 0:
            # Apply some scaling to make the visualization more visible
            # Increased scale factor for larger wave amplitude
            scale_factor = 200  # Increased from 0.5 to 0.8 for larger waves
            audio_data = audio_data * scale_factor
            
            # Clip to ensure it stays within the canvas
            audio_data = np.clip(audio_data, -self.height/2, self.height/2)
        
        self.audio_data = audio_data
    
    def _update_visualization(self):
        """Update the visualization line with current audio data."""
        if not self.is_active:
            return
        
        # Create points for the line
        points = []
        mid_y = self.height // 2
        
        for x in range(self.width):
            if x < len(self.audio_data):
                # Invert the y-coordinate (canvas y increases downward)
                y = mid_y - self.audio_data[x]
                points.extend([x, y])
            else:
                points.extend([x, mid_y])
        
        # Update the line on the canvas
        self.canvas.coords(self.line_id, *points)
        
        # Schedule the next update
        self.update_id = self.master.after(self.update_interval_ms, self._update_visualization)
    
    def start(self):
        """Start the visualization updates."""
        if not self.is_active:
            self.is_active = True
            self._update_visualization()
    
    def _on_resize(self, event):
        """Handle window resize events to adjust the visualizer.
        
        Args:
            event: The Configure event containing new dimensions
        """
        # Only respond to events from the parent widget
        if event.widget == self.master:
            # Get the new width
            new_width = event.width
            
            # Update canvas width
            if new_width != self.width:
                self.width = new_width
                self.canvas.config(width=new_width)
                
                # Resize the audio data array
                if len(self.audio_data) > 0:
                    # Create new array with the new width
                    new_data = np.zeros(new_width)
                    
                    # Resample existing data to the new width
                    if len(self.audio_data) > 1:
                        indices = np.linspace(0, len(self.audio_data) - 1, min(new_width, len(self.audio_data)), dtype=int)
                        new_data[:min(new_width, len(self.audio_data))] = self.audio_data[indices]
                    
                    self.audio_data = new_data
                else:
                    self.audio_data = np.zeros(new_width)
                
                # Recreate the visualization line
                if self.line_id is not None:
                    self.canvas.delete(self.line_id)
                self._create_line()
    
    def stop(self):
        """Stop the visualization updates."""
        self.is_active = False
        if self.update_id is not None:
            self.master.after_cancel(self.update_id)
            self.update_id = None


class DualAudioVisualizer(tk.Frame):
    """Frame containing two audio visualizers for system and microphone audio."""
    
    def __init__(self, master: tk.Widget, width: int = 400, height: int = 200):
        """Initialize the dual audio visualizer.
        
        Args:
            master: The tkinter widget to place the visualizer in
            width: Width of the visualizer frame
            height: Height of the visualizer frame
        """
        super().__init__(master, width=width, height=height)
        self.width = width
        self.height = height
        
        # Create labels
        self.system_label = tk.Label(self, text="System Audio")
        self.system_label.pack(anchor=tk.W, padx=5)
        
        # Create system audio visualizer with increased height for better visibility
        self.system_visualizer = AudioVisualizer(
            self, width=width, height=height//3,  # Increased from height//4 to height//3
            bg_color="#000000", line_color="#00FF00"
        )
        self.system_visualizer.canvas.pack(fill=tk.X, padx=5, pady=5)
        
        # Create microphone label
        self.mic_label = tk.Label(self, text="Microphone Audio")
        self.mic_label.pack(anchor=tk.W, padx=5, pady=(10, 0))
        
        # Create microphone audio visualizer with increased height
        self.mic_visualizer = AudioVisualizer(
            self, width=width, height=height//3,  # Increased from height//4 to height//3
            bg_color="#000000", line_color="#00FFFF"
        )
        self.mic_visualizer.canvas.pack(fill=tk.X, padx=5, pady=5)
        
        # Bind to parent resize events
        self.bind('<Configure>', self._on_resize)
    
    def update_system_data(self, audio_data: np.ndarray):
        """Update the system audio visualizer data.
        
        Args:
            audio_data: Audio data as numpy array
        """
        self.system_visualizer.update_data(audio_data)
    
    def update_mic_data(self, audio_data: np.ndarray):
        """Update the microphone audio visualizer data.
        
        Args:
            audio_data: Audio data as numpy array
        """
        self.mic_visualizer.update_data(audio_data)
    
    def start(self):
        """Start both visualizers."""
        self.system_visualizer.start()
        self.mic_visualizer.start()
    
    def _on_resize(self, event):
        """Handle resize events for the dual visualizer frame.
        
        Args:
            event: The Configure event containing new dimensions
        """
        # Only respond to events from this frame
        if event.widget == self:
            # Update internal width
            new_width = event.width
            if new_width != self.width:
                self.width = new_width
                
                # No need to manually update the visualizers as they have their own resize handlers
                # The canvas resize will trigger their own <Configure> events
    
    def stop(self):
        """Stop both visualizers."""
        self.system_visualizer.stop()
        self.mic_visualizer.stop()