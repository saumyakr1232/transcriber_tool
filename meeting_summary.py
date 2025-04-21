from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MeetingSummary:
    """
    A structured model for meeting summaries with organized information.

    This class provides a structured way to store and access meeting summary data,
    including the main summary text, key points discussed, and action items or tasks
    that were identified during the meeting.
    """

    summary: str
    key_points: List[str]
    tasks: List[str]

    def __str__(self) -> str:
        """Return a formatted string representation of the meeting summary."""
        formatted_output = f"Meeting Summary:\n{self.summary}\n\n"

        if self.key_points:
            formatted_output += "Key Points:\n"
            for i, point in enumerate(self.key_points, 1):
                formatted_output += f"{i}. {point}\n"
            formatted_output += "\n"

        if self.tasks:
            formatted_output += "Action Items / Tasks:\n"
            for i, task in enumerate(self.tasks, 1):
                formatted_output += f"{i}. {task}\n"

        return formatted_output

    @classmethod
    def from_dict(cls, data: dict) -> 'MeetingSummary':
        """Create a MeetingSummary instance from a dictionary."""
        return cls(
            summary=data.get('summary', ''),
            key_points=data.get('key_points', []),
            tasks=data.get('tasks', [])
        )
