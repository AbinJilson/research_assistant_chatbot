# Multimodal Research Assistant

This project is a powerful, multimodal research assistant built with Streamlit and powered by Google's Gemini models. It allows you to chat with your documents (PDFs and web pages), ask questions, and get answers with sources, all through an intuitive web interface.

## Features

- **Multimodal Input:** Process and analyze both PDF documents and web URLs.
- **Image Analysis:** Automatically extracts images from PDFs and generates descriptive captions.
- **Advanced Question Answering:** Uses a Retrieval-Augmented Generation (RAG) pipeline to provide answers based on the content of your documents.
- **Source Citation:** Displays the sources used to generate an answer, allowing you to verify the information.
- **Interactive UI:** A user-friendly interface built with Streamlit that allows you to upload documents, ask questions, and view results in real-time.
- **Component-Based Architecture:** The project is structured into modular components, making it easy to understand, maintain, and extend.

## How to Run the Code

To get the Research Assistant up and running on your local machine, follow these steps:

### Prerequisites

- Python 3.8 or higher
- An API key for the Google Gemini models.

### Installation

1. **Clone the repository (or download the source code):**
   ```bash
   git clone https://github.com/AbinJilson/research_assistant_chatbot/
   cd research_assistant_chatbot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables:**
   - Create a file named `.env` in the root of the project directory.
   - Add your Google Gemini API key to the `.env` file as follows:
     ```
     GEMINI_API_KEY="your-api-key-here"
     ```

### Running the Application

Once you have completed the installation steps, you can run the Streamlit application with the following command:

```bash
streamlit run app.py
```

This will start the web server and open the application in your default web browser.

## Design Choices and Architecture

The Research Assistant is built on a modular, component-based architecture that separates concerns and promotes code reusability. The key components are:

- **`app.py` (User Interface):** The main entry point of the application, built with Streamlit. It handles user interactions, file uploads, and the overall layout of the web interface.

- **`src/assistant_logic.py` (Orchestrator):** The `MultimodalResearchAssistant` class acts as the central orchestrator. It initializes and coordinates all the other components, managing the flow of data from document processing to question answering.

- **`src/document_processing.py` (Content Processing):** This module is responsible for parsing and processing input documents. It uses `PyMuPDF` to extract text and images from PDFs and `BeautifulSoup` to scrape content from web pages. It also uses LangChain's `RecursiveCharacterTextSplitter` to break down large texts into manageable chunks.

- **`src/retrieval.py` (Information Retrieval):** The `MultimodalRetriever` class manages the information retrieval pipeline. It uses a `MultiVectorRetriever` from LangChain, with a FAISS vector store for efficient similarity searches and an in-memory store for the original documents. It creates embeddings for both text chunks and image captions, enabling true multimodal retrieval.

- **`src/scraper.py` (Web Scraping):** This module handles the scraping of web pages. It uses `BeautifulSoup` and `requests` to fetch and parse HTML content from URLs.

- **`src/utils.py` (Utility Functions):** This file contains helper functions and utilities that are used across the application, such as running asynchronous tasks in a separate thread.

## Limitations

- **Dependency on Gemini API:** The application requires a valid API key for Google's Gemini models to function. All language and vision capabilities are tied to this service.
- **Potential for Hallucinations:** Like all large language models, the answers generated by the assistant may occasionally contain inaccuracies or "hallucinations." The inclusion of sources helps mitigate this by allowing users to verify the information.
- **Performance:** Processing very large or image-heavy documents can be time-consuming and memory-intensive.

## Dependencies

The project relies on a number of open-source libraries, including:

- `streamlit`: For the web interface.
- `langchain` and `langchain-google-genai`: For the RAG pipeline and Gemini integration.
- `faiss-cpu`: For the vector store.
- `PyMuPDF`: For PDF processing.
- `beautifulsoup4` and `requests`: For web scraping.
- `opencv-python` and `pillow`: For image processing.

For a complete list of dependencies, please see the `requirements.txt` file.

## Future Work

- **Support for More Document Types:** Add support for other document formats, such as `.docx`, `.pptx`, and `.txt`.
- **Improved Caching:** Implement more sophisticated caching mechanisms to speed up the processing of previously seen documents.
- **Advanced Chat History:** Allow users to have persistent conversations and refer back to previous questions and answers.
- **Batch Processing:** Enable users to upload and process multiple documents at once in a more streamlined manner.
