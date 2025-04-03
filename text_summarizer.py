#!/usr/bin/env python3

import os
import sys
from typing import Optional, Dict, Any


from langchain.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document



class TextSummarizer:
    """Class for text summarization using Ollama and LangChain.
    
    This class provides functionality to summarize text using a local LLM through Ollama
    and LangChain's summarization chain.
    """
    
    def __init__(self, model_name: str = "mistral", chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the text summarizer.
        
        Args:
            model_name: Name of the Ollama model to use
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between text chunks
        """
        try:
            # Initialize Ollama LLM
            self.llm = Ollama(model=model_name)
            
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
    summarizer = TextSummarizer()
    
    # Print original text
    print("\nOriginal Text:")
    print(sample_text)
    
    # Generate and print summary
    print("\nSummary:")
    summary = summarizer.summarize(sample_text)
    print(summary)