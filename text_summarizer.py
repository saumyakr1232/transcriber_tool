import re
from typing import Union

from langchain_ollama import OllamaLLM
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from config import get_config
from models import MeetingSummary

# Define prompt templates for extracting structured meeting summary
MAP_TEMPLATE = """Summarize the following text from a meeting transcript:
{text}

SUMMARY:"""

COMBINE_TEMPLATE = """Below are summaries of chunks from a meeting transcript:
{text}

Your task is to create a comprehensive summary of the entire meeting with the following structure:
1. A concise summary paragraph of the main discussion points and conclusions.
2. A list of key points discussed during the meeting.
3. A list of specific tasks, action items, or follow-ups mentioned during the meeting.

Provide your response in the following format:

SUMMARY: <concise summary paragraph>

KEY POINTS:
- <key point 1>
- <key point 2>
- ...

TASKS:
- <task or action item 1>
- <task or action item 2>
- ...
"""


class TextSummarizer:
    """Class for text summarization using Ollama and LangChain.

    This class provides functionality to summarize text using a local LLM through Ollama
    and LangChain's summarization chain.
    """

    # Add model size configuration at class level
    MODEL_SIZE_CONFIG = {
        "7b": {"context_window": 4096, "chunk_size": 2048},
        "8b": {"context_window": 4096, "chunk_size": 2048},
        "13b": {"context_window": 8192, "chunk_size": 4096},
        "34b": {"context_window": 16384, "chunk_size": 8192},
        "70b": {"context_window": 32768, "chunk_size": 16384}
    }

    def _parse_model_name(self, model_name: str) -> tuple[str, str]:
        """Parse model name to extract base name and size.

        Args:
            model_name: Model name in format 'model_name:size' (e.g., 'llama2:13b')

        Returns:
            Tuple of (base_name, size)
        """
        if ":" not in model_name:
            return model_name, "7b"  # Default size if not specified

        base_name, size = model_name.split(":")
        return base_name, size.lower()

    def _get_chunk_size(self, model_size: str) -> int:
        """Get appropriate chunk size based on model size.

        Args:
            model_size: Size of the model (e.g., '13b')

        Returns:
            Recommended chunk size for the model
        """
        config = self.MODEL_SIZE_CONFIG.get(model_size, self.MODEL_SIZE_CONFIG["7b"])
        return config["chunk_size"]

    def __init__(self, config, chunk_overlap: int = 200):
        """Initialize the text summarizer.

        Args:
            config: Configuration object containing model settings
            chunk_overlap: Overlap between text chunks
        """
        try:
            # Get model configuration
            model_name = config.get("summarizer.model_name", "llama2:13b")
            base_url = config.get("summarizer.base_url", "http://localhost:11434")

            print(f"Using model: {model_name}", f"at base_url: {base_url}")
            # Parse model name and get appropriate chunk size
            _, model_size = self._parse_model_name(model_name)
            chunk_size = self._get_chunk_size(model_size)

            # Initialize Ollama LLM
            self.llm = OllamaLLM(model=model_name, base_url=base_url)

            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            # Initialize prompt templates
            self.map_prompt = PromptTemplate(template=MAP_TEMPLATE, input_variables=["text"])
            self.combine_prompt = PromptTemplate(template=COMBINE_TEMPLATE, input_variables=["text"])

            # Initialize standard summarization chain
            self.chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                verbose=True
            )

            # Initialize structured summarization chain
            self.structured_chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                map_prompt=self.map_prompt,
                combine_prompt=self.combine_prompt,
                verbose=True
            )

        except Exception as e:
            print(f"Error initializing text summarizer: {e}")
            raise

    def summarize(self, text: str, structured: bool = False) -> Union[str, MeetingSummary]:
        """Summarize the given text.

        Args:
            text: The text to summarize
            structured: Whether to return a structured MeetingSummary (True) or plain text (False)

        Returns:
            Either a plain text summary (string) or a structured MeetingSummary object
        """
        try:
            # Split text into chunks
            texts = self.text_splitter.split_text(text)
            docs = [Document(page_content=t) for t in texts]

            if not structured:
                # Generate standard summary
                summary = self.chain.invoke({"input_documents": docs})["output_text"]
                return summary.strip()
            else:
                # Generate structured summary
                result = self.structured_chain.invoke({"input_documents": docs})["output_text"]
                return self._parse_structured_summary(result)

        except Exception as e:
            print(f"Error summarizing text: {e}")
            if structured:
                return MeetingSummary(summary="", key_points=[], tasks=[])
            else:
                return ""

    def summarize_file(self, file_path: str, structured: bool = False) -> Union[str, MeetingSummary]:
        """Summarize text from a file.

        Args:
            file_path: Path to the text file
            structured: Whether to return a structured MeetingSummary (True) or plain text (False)

        Returns:
            Either a plain text summary (string) or a structured MeetingSummary object
        """
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Summarize the text
            return self.summarize(text, structured)

        except Exception as e:
            print(f"Error summarizing file: {e}")
            if structured:
                return MeetingSummary(summary="", key_points=[], tasks=[])
            else:
                return ""

    def _parse_structured_summary(self, text: str) -> MeetingSummary:
        """Parse the structured summary text into a MeetingSummary object.

        Args:
            text: The structured summary text from the LLM

        Returns:
            A MeetingSummary object with extracted summary, key points, and tasks
        """
        # Default values
        summary = ""
        key_points = []
        tasks = []

        # Try to extract the summary
        summary_match = re.search(r"SUMMARY:\s*(.+?)(?:\n\n|\n\s*KEY POINTS:|$)", text, re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip()

        # Try to extract key points
        key_points_match = re.search(r"KEY POINTS:\s*(.+?)(?:\n\n|\n\s*TASKS:|$)", text, re.DOTALL)
        if key_points_match:
            key_points_text = key_points_match.group(1).strip()
            # Extract points that are prefixed with a dash or bullet
            key_points = [point.strip().lstrip('- ').lstrip('• ')
                          for point in key_points_text.split('\n')
                          if point.strip() and not point.strip().startswith('KEY POINTS:')]

        # Try to extract tasks
        tasks_match = re.search(r"TASKS:\s*(.+?)(?:\n\n|$)", text, re.DOTALL)
        if tasks_match:
            tasks_text = tasks_match.group(1).strip()
            # Extract tasks that are prefixed with a dash or bullet
            tasks = [task.strip().lstrip('- ').lstrip('• ')
                     for task in tasks_text.split('\n')
                     if task.strip() and not task.strip().startswith('TASKS:')]

        # Create and return the MeetingSummary object
        return MeetingSummary(summary=summary, key_points=key_points, tasks=tasks)


if __name__ == "__main__":
    # Sample text for testing a meeting transcript
    sample_text = """
    Meeting Transcript:
    
    John: Let's begin our weekly product development meeting. Today, we need to discuss the upcoming launch of version 2.0, address the bugs found in testing, and assign tasks for the next sprint.
    
    Sarah: I've been working on the UI improvements. Most of the design elements are ready, but we still need to finalize the color scheme and get approval from marketing.
    
    David: The backend integration tests are showing some issues with the payment processing module. I think we need to prioritize fixing those before release.
    
    John: I agree. Let's put that at the top of our priority list. Sara, can you work with the marketing team to get that approval by Friday?
    
    Sarah: Yes, I'll schedule a meeting with them tomorrow.
    
    David: Another thing - we need to update the documentation to reflect the new features. Who should handle that?
    
    John: Let's assign that to Michael, but I'll need to check his availability first. I'll also reach out to the QA team to conduct another round of testing once the payment bugs are fixed.
    
    Sarah: What about the customer feedback on the beta version? Should we address any of those issues before launch?
    
    John: Good point. Let's review the most common feedback and see if there's anything critical we should include in this release. Otherwise, we'll add them to the backlog for version 2.1.
    
    David: I'll prepare a summary of the feedback by Thursday so we can make those decisions.
    
    John: Perfect. So our action items are: David fixes the payment processing bugs, Sarah finalizes the UI with marketing by Friday, I'll assign the documentation to Michael, and David will provide the feedback summary by Thursday. Anything else?
    
    Sarah: We should also send an update to the stakeholders about our progress and the expected release date.
    
    John: Good idea. I'll draft an email by the end of the week. If there's nothing else, let's adjourn and get to work on these tasks.
    """

    # Create an instance of TextSummarizer
    summarizer = TextSummarizer(get_config())

    # Print original text
    print("\nOriginal Text:")
    print(sample_text)

    # Generate and print standard summary
    print("\nStandard Summary:")
    standard_summary = summarizer.summarize(sample_text)
    print(standard_summary)

    # Generate and print structured summary
    print("\nStructured Summary:")
    meeting_summary = summarizer.summarize(sample_text, structured=True)
    print(meeting_summary)
