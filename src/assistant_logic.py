import logging
import os
import tempfile
from pathlib import Path
import asyncio

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.document_processing import DocumentProcessor, ImageProcessor
from src.retrieval import MultimodalRetriever
from src.scraper import WebScraper
from src.utils import run_async_in_thread

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class MultimodalResearchAssistant:
    """Main class orchestrating the multimodal research assistant"""

    def __init__(self):
        self.image_processor = ImageProcessor()
        self.doc_processor = DocumentProcessor(self.image_processor)
        self.retriever = MultimodalRetriever()
        self.web_scraper = WebScraper()
        self.processed_docs = []
        self.temp_files = []
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.7)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {e}")
            self.llm = None

    def _build_full_index(self):
        """Builds the index with all added documents, showing a spinner in the UI."""
        with st.spinner("🧠 Building knowledge index..."):
            self.retriever.build_index()

    def add_pdf(self, file_path: str):
        """Add a PDF document to the system"""
        with st.spinner(f"Processing {Path(file_path).name}..."):
            processed_doc = self.doc_processor.process_pdf(file_path)
            if processed_doc:
                self.processed_docs.append(processed_doc)
                self.retriever.add_documents(processed_doc)
                self._build_full_index()
                st.success(f"✅ Successfully processed and indexed {Path(file_path).name}")
            else:
                st.error(f"Failed to process {Path(file_path).name}")

    async def process_url_async(self, url: str):
        """Process a URL asynchronously and add its content to the knowledge base"""
        if not url or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
            logger.error(f"Invalid URL: {url}")
            return False
            
        try:
            # Use the async web scraper
            content = await self.web_scraper.scrape(url)
            if not content:
                logger.error(f"Failed to scrape content from {url}")
                return False
                
            # Process the scraped content directly
            processed_doc = await asyncio.to_thread(
                self.doc_processor.process_html,
                content,
                url
            )
            
            if processed_doc:
                self.processed_docs.append(processed_doc)
                self.retriever.add_documents(processed_doc)
                return True
            else:
                logger.error(f"Failed to process HTML content from {url}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return False
                    
        return False

    def add_url(self, url: str):
        """Add a web URL or a direct PDF link to the system."""
        with st.spinner(f"Scraping and processing {url}..."):
            try:
                # Check if it's a PDF URL
                if url.lower().endswith('.pdf'):
                    # Download the PDF to a temporary file
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                temp_file.write(chunk)
                        temp_file_path = temp_file.name
                    
                    # Add the downloaded PDF
                    self.add_pdf(temp_file_path)
                else:
                    # It's a web page
                    st.write("Processing URL asynchronously...")
                    try:
                        # Use asyncio.run() to run the async function
                        result = asyncio.run(self.process_url_async(url))
                        if result:
                            self._build_full_index()
                            st.success(f"✅ Successfully scraped and indexed {url}")
                        else:
                            st.error(f"❌ Failed to process {url}")
                    except Exception as e:
                        logger.error(f"Error processing URL {url}: {e}")
                        st.error(f"Error processing URL: {str(e)}")
                        
            except requests.RequestException as e:
                logger.error(f"Error downloading {url}: {e}")
                st.error(f"Error downloading {url}: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error processing {url}: {e}")
                st.error(f"An unexpected error occurred while processing {url}")

    def query(self, question: str):
        """Query the research assistant"""
        if not self.retriever or not self.retriever.retriever:
            st.warning("Please add documents before asking a question.")
            return None

        with st.spinner("🔍 Searching for answers..."):
            retrieved_docs = self.retriever.retrieve(question, k=10)
            
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            
            prompt = f"""
            You are a helpful research assistant. Based on the following context, please answer the user's question.
            The context is a mix of text chunks and image captions from multiple documents.
            
            CONTEXT:
            {context}
            
            QUESTION: {question}
            
            Provide a comprehensive answer. If the context contains tables or data, synthesize it in your answer.
            If the context is insufficient, say so. Do not make up information.
            Cite the sources used in your answer, referencing the 'source' from the metadata.
            """
            
            if not self.llm:
                st.error("LLM not initialized. Cannot answer questions.")
                return None

            try:
                response = self.llm.invoke(prompt)
                answer = response.content
                
                # Convert Document objects to dictionaries for serialization
                relevant_docs = []
                for doc in retrieved_docs:
                    relevant_docs.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    })

                return {"answer": answer, "relevant_docs": relevant_docs, "context": context}
            except Exception as e:
                logger.error(f"Error during LLM query: {e}")
                st.error(f"An error occurred while generating the answer: {e}")
                return None
