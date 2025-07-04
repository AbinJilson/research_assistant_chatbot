# High-Level Design (HLD) for Multimodal Research Assistant Chatbot

## Overview
The Multimodal Research Assistant is a Streamlit-based web application that enables users to interact with and query information from both PDF documents and web pages. It leverages Google's Gemini models for advanced language and vision capabilities, providing a seamless, multimodal research experience.

---

## Architecture Diagram

```
+-------------------+         +-------------------+         +-------------------+
|                   |         |                   |         |                   |
|   User Interface  +-------->+   Assistant Logic  +-------->+   Document & Web  |
|    (Streamlit)    |         | (Orchestrator)    |         |   Processing      |
|                   |         |                   |         |                   |
+-------------------+         +-------------------+         +-------------------+
        |                              |                              |
        |                              v                              v
        |                    +-------------------+         +-------------------+
        |                    |   Retrieval &     |         |   Web Scraper     |
        |                    |   Indexing        |         |                   |
        |                    +-------------------+         +-------------------+
        |                              |                              |
        +------------------------------+------------------------------+
                                       |
                                       v
                             +-------------------+
                             |   Gemini LLM API  |
                             +-------------------+
```

---

## Component Breakdown

### 1. User Interface (`app.py`)
- Built with Streamlit.
- Handles file uploads, user queries, and displays results.
- Manages session state and initializes the assistant.

### 2. Assistant Logic (`src/assistant_logic.py`)
- Central orchestrator (`MultimodalResearchAssistant` class).
- Coordinates document processing, retrieval, and LLM-based answering.
- Manages the flow between UI, processing, and retrieval.

### 3. Document & Image Processing (`src/document_processing.py`)
- Extracts text and images from PDFs using PyMuPDF.
- Processes images and generates captions/descriptions using Gemini Vision.
- Splits text into manageable chunks for retrieval.
- Defines the `ProcessedDocument` and `ImageProcessor` classes.

### 4. Retrieval & Indexing (`src/retrieval.py`)
- Uses LangChain's `MultiVectorRetriever` and FAISS for similarity search.
- Embeds both text and image data for true multimodal retrieval.
- Handles efficient search and ranking of relevant content.

### 5. Web Scraping (`src/scraper.py`)
- Fetches and parses web content using `requests` and `BeautifulSoup`.
- Optionally uses `crawl4ai` for advanced async crawling.
- Provides content for indexing and answering queries.

### 6. Utilities (`src/utils.py`)
- Helper functions for async execution, error handling, and image library imports.
- Ensures smooth operation across platforms (e.g., Windows event loop policy).

---

## Data Flow
1. **User uploads PDF(s) or enters a web URL via the UI.**
2. **Assistant Logic** triggers document or web processing:
    - PDFs: Text and images are extracted, processed, and chunked.
    - Web: Content is scraped and parsed.
3. **Processed content** is indexed by the Retrieval module (text and image embeddings).
4. **User asks a question.**
5. **Retriever** finds the most relevant chunks/images.
6. **Assistant Logic** sends context and question to Gemini LLM for answer generation.
7. **UI displays** the answer and sources to the user.

---

## Key Technologies
- **Streamlit**: UI framework.
- **LangChain**: RAG pipeline, chunking, and retrieval.
- **FAISS**: Vector store for similarity search.
- **Google Gemini API**: LLM and vision capabilities.
- **PyMuPDF, BeautifulSoup, Requests**: PDF and web processing.
- **OpenCV, Pillow**: Image processing.

---

## Security & Limitations
- Requires a valid Gemini API key (stored in `.env`).
- All LLM and vision processing is cloud-based (data sent to Google).
- Performance may be limited by document size and API rate limits.

---

## Extensibility
- Modular design allows for easy addition of new document types, retrievers, or LLMs.
- Future improvements: more file formats, persistent chat history, advanced caching, batch processing.

---

## Authors & Credits
- See `README.md` for full credits and open-source dependencies.
