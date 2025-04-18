import os
import tempfile
import numpy as np
from moviepy.editor import TextClip, CompositeVideoClip, VideoFileClip


class SubtitleAdder:
    """Module for adding subtitles to video clips and video files"""

    def __init__(self, style="default", font="Arial", fontsize=40, color="white", stroke_color="black", stroke_width=1.5):
        """Initialize the subtitle adder

        Args:
            style (str): Style of subtitles (default, tiktok, youtube, instagram)
            font (str): Font to use for subtitles
            fontsize (int): Font size for subtitles
            color (str): Text color
            stroke_color (str): Text stroke color
            stroke_width (float): Text stroke width
        """
        self.font = font
        self.fontsize = fontsize
        self.color = color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width

        # Apply preset styles
        self._apply_style(style)

    def _apply_style(self, style):
        """Apply a preset subtitle style

        Args:
            style (str): Style name (default, tiktok, youtube, instagram)
        """
        if style == "tiktok":
            self.font = "Arial-Bold"
            self.fontsize = 50
            self.color = "white"
            self.stroke_color = "black"
            self.stroke_width = 2.0
        elif style == "youtube":
            self.font = "Arial"
            self.fontsize = 45
            self.color = "white"
            self.stroke_color = "black"
            self.stroke_width = 1.5
        elif style == "instagram":
            self.font = "Helvetica-Bold"
            self.fontsize = 48
            self.color = "white"
            self.stroke_color = "black"
            self.stroke_width = 1.8
        # default style is already set in __init__

    def _create_subtitle_clip(self, text, duration, clip_size):
        """Create a subtitle clip with the given text

        Args:
            text (str): Text for the subtitle
            duration (float): Duration of the subtitle in seconds
            clip_size (tuple): Size of the video clip (width, height)

        Returns:
            TextClip: Subtitle clip
        """
        # Create text clip
        txt_clip = TextClip(
            text,
            font=self.font,
            fontsize=self.fontsize,
            color=self.color,
            stroke_color=self.stroke_color,
            stroke_width=self.stroke_width,
            method='caption',
            align='center',
            size=(clip_size[0] * 0.9, None)  # 90% of video width, auto height
        )

        # Set duration and position (bottom center)
        txt_clip = txt_clip.set_duration(duration)
        txt_clip = txt_clip.set_position(('center', 'bottom'))

        return txt_clip

    def _split_long_text(self, text, max_chars_per_line=30):
        """Split long text into multiple lines for better readability

        Args:
            text (str): Text to split
            max_chars_per_line (int): Maximum characters per line

        Returns:
            str: Text with line breaks
        """
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars_per_line:
                current_line += (" " + word) if current_line else word
            else:
                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return "\n".join(lines)

    def add_subtitles(self, edited_clips):
        """Add subtitles to the edited clips

        Args:
            edited_clips (list): List of edited video clips
            highlights (list): List of highlight segments with text

        Returns:
            list: List of video clips with subtitles
        """
        subtitled_clips = []

        for i, clip_info in enumerate(edited_clips):
            clip = clip_info["clip"]
            highlight = clip_info["highlight"]

            # Get segments from highlight
            segments = highlight.get("segments", [])

            if not segments:
                # If no segments, use the full highlight text
                text = highlight.get("text", "")
                formatted_text = self._split_long_text(text)

                # Create subtitle clip for the full duration
                subtitle = self._create_subtitle_clip(formatted_text, clip.duration, clip.size)

                # Composite video and subtitle
                final_clip = CompositeVideoClip([clip, subtitle])
                subtitled_clips.append(final_clip)
            else:
                # Create subtitle clips for each segment
                subtitle_clips = []

                for segment in segments:
                    # Calculate relative start and end times
                    rel_start = segment["start"] - highlight["start_time"]
                    rel_end = segment["end"] - highlight["start_time"]

                    # Ensure times are within clip duration
                    rel_start = max(0, min(rel_start, clip.duration))
                    rel_end = max(rel_start, min(rel_end, clip.duration))

                    # Format text
                    text = segment["text"]
                    formatted_text = self._split_long_text(text)

                    # Create subtitle clip
                    subtitle = self._create_subtitle_clip(
                        formatted_text,
                        rel_end - rel_start,
                        clip.size
                    )

                    # Set start time relative to clip
                    subtitle = subtitle.set_start(rel_start)

                    subtitle_clips.append(subtitle)

                # Composite video and all subtitle clips
                final_clip = CompositeVideoClip([clip] + subtitle_clips)
                subtitled_clips.append(final_clip)

        return subtitled_clips

    def add_subtitles_to_video(self, video_path, segments, output_path=None):
        """Add subtitles to a video file using transcription segments

        Args:
            video_path (str): Path to the video file
            segments (list): List of transcription segments with timestamps
            output_path (str, optional): Path to save the output video. If None, a temp file is used.

        Returns:
            str: Path to the output video file with subtitles
        """
        # Load the video file
        video = VideoFileClip(video_path)

        # Create subtitle clips for each segment
        subtitle_clips = []

        for segment in segments:
            # Get start and end times
            start_time = segment.get("start", 0)
            end_time = segment.get("end", start_time + 5)  # Default to 5 seconds if no end time
            duration = end_time - start_time

            # Format text
            text = segment.get("text", "")
            formatted_text = self._split_long_text(text)

            # Create subtitle clip
            subtitle = self._create_subtitle_clip(
                formatted_text,
                duration,
                video.size
            )

            # Set start time
            subtitle = subtitle.set_start(start_time)

            subtitle_clips.append(subtitle)

        # Composite video and all subtitle clips
        final_video = CompositeVideoClip([video] + subtitle_clips)

        # Determine output path
        if output_path is None:
            # Create a temporary file with the same extension as the input
            _, ext = os.path.splitext(video_path)
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                output_path = temp_file.name

        # Write the output video
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

        return output_path
