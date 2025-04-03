#!/usr/bin/env python3

import os
import sys
from typing import Optional, Dict, Any


from langchain_ollama import OllamaLLM
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from config import get_config


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
            base_name, model_size = self._parse_model_name(model_name)
            chunk_size = self._get_chunk_size(model_size)
            
            # Initialize Ollama LLM
            self.llm = OllamaLLM(model=base_name, base_url=base_url)
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Initialize summarization chain
            self.chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                verbose=True
            )
            
        except Exception as e:
            print(f"Error initializing text summarizer: {e}")
            raise
    
    def summarize(self, text: str) -> str:
        """Summarize the given text.
        
        Args:
            text: The text to summarize
            
        Returns:
            Summarized text
        """
        try:
            # Split text into chunks
            texts = self.text_splitter.split_text(text)
            docs = [Document(page_content=t) for t in texts]
            
            # Generate summary
            summary = self.chain.run(docs)
            
            return summary.strip()
            
        except Exception as e:
            print(f"Error summarizing text: {e}")
            return ""
    
    def summarize_file(self, file_path: str) -> str:
        """Summarize text from a file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Summarized text
        """
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Summarize the text
            return self.summarize(text)
            
        except Exception as e:
            print(f"Error summarizing file: {e}")
            return ""

if __name__ == "__main__":
    # Sample text for testing
    sample_text = """
    Artificial Intelligence (AI) has emerged as one of the most transformative technologies of our time. 
    It encompasses machine learning, deep learning, natural language processing, and robotics. 
    AI systems can now perform tasks that traditionally required human intelligence, such as visual 
    perception, speech recognition, decision-making, and language translation. The technology has 
    found applications across various sectors, including healthcare, finance, transportation, and 
    education. However, the rapid advancement of AI also raises important ethical considerations 
    regarding privacy, bias, accountability, and the future of human work. As AI continues to evolve, 
    it's crucial to ensure its development aligns with human values and benefits society as a whole.
    """
    
    # Create an instance of TextSummarizer
    summarizer = TextSummarizer(get_config())
    
    # Print original text
    print("\nOriginal Text:")
    print(sample_text)
    
    # Generate and print summary
    print("\nSummary:")
    summary = summarizer.summarize(sample_text)
    print(summary)