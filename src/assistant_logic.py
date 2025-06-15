import logging
import os
import tempfile
from pathlib import Path

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

    def add_url(self, url: str):
        """Add a web URL or a direct PDF link to the system."""
        with st.spinner(f"Scraping and processing {url}..."):
            try:
                response = requests.head(url, allow_redirects=True, timeout=10)
                content_type = response.headers.get('Content-Type', '')

                if 'application/pdf' in content_type:
                    # It's a direct link to a PDF
                    pdf_response = requests.get(url, timeout=30)
                    pdf_response.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="url_") as temp_file:
                        temp_file.write(pdf_response.content)
                        temp_file_path = temp_file.name
                    self.temp_files.append(temp_file_path)
                    self.add_pdf(temp_file_path)
                else:
                    # It's a web page
                    html_content = run_async_in_thread(self.web_scraper.scrape(url))
                    if html_content:
                        processed_doc = self.doc_processor.process_html(html_content, url)
                        self.processed_docs.append(processed_doc)
                        self.retriever.add_documents(processed_doc)
                        self._build_full_index()
                        st.success(f"✅ Successfully scraped and indexed {url}")
                    else:
                        st.error(f"Failed to scrape content from {url}")
            except requests.RequestException as e:
                st.error(f"Error fetching URL: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

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
