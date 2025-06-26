import base64
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
import requests
from urllib.parse import urljoin
import mimetypes

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.schema.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image

from src.utils import _import_image_dependencies, on_rm_error, cv2, np, custom_clean_text, split_into_sentences

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


@dataclass
class ProcessedDocument:
    doc_id: str
    text_chunks: List[Document]
    images: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    full_text: str


class ImageProcessor:
    def __init__(self):
        _import_image_dependencies()
        try:
            self.gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Vision model: {e}")
            self.gemini_llm = None

    def generate_image_caption(self, image_path: str, context: str = "") -> str:
        if not self.gemini_llm:
            return "Image captioning not available"
        try:
            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
            encoded_image = base64.b64encode(img_bytes).decode("utf-8")
            prompt = (
                "You are an expert research assistant. Analyze the following image from a research paper.\n\n"
                "1. **Summarize the main finding:** What is the key takeaway or conclusion presented in this image?\n"
                "2. **Extract key data:** If it's a graph or chart, what are the labels on the axes? What are the units? What are the most important data points or trends?\n"
                "3. **Describe the components:** If it's a diagram or flowchart, describe the different parts and how they are connected.\n"
                f"4. **Consider the context:** {context} Use this context to provide a more in-depth analysis of the image's significance.\n"
                "5. **Infer the researcher's intent:** What message or point is the researcher trying to convey with this image? How does it support the paper's main arguments?"
                "6.Provide the analysis directly, without any introductory phrases (e.g., 'Here's an analysis...').\n\n"
            )
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
                ]
            )
            response = self.gemini_llm.invoke([message])
            response_text = response.content
            generic_phrases = ["please provide the image", "i need the image", "cannot analyze the image"]
            if any(phrase in response_text.lower() for phrase in generic_phrases):
                return "This is likely a decorative or non-informative image."
            return response_text
        except Exception as e:
            logger.error(f"Error generating image caption for {image_path}: {e}")
            return "Error generating caption."


class DocumentProcessor:
    def __init__(self, image_processor: ImageProcessor, tokenizer=None, nlp=None):
        self.image_processor = image_processor
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.tokenizer = tokenizer
        self.nlp = nlp

    def process_pdf(self, file_path: str) -> ProcessedDocument:
        doc_id = str(uuid.uuid4())
        temp_dir = Path(tempfile.mkdtemp(prefix=f"pdf_{doc_id}_"))
        images = []
        full_text = ""

        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += page.get_text() + "\n\n"
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_path = temp_dir / f"img_{page_num+1}_{img_index+1}.{image_ext}"
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    images.append({"path": str(image_path), "page": page_num + 1, "caption": ""})
            doc.close()
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            shutil.rmtree(temp_dir, onerror=on_rm_error)
            return ProcessedDocument(doc_id=doc_id, text_chunks=[], images=[], metadata={}, full_text="")

        # Clean text before chunking
        cleaned_text = custom_clean_text(full_text)
        # Optionally use semantic chunking if tokenizer and nlp are provided
        if self.tokenizer and self.nlp:
            chunks = split_into_sentences(cleaned_text, self.tokenizer, nlp=self.nlp)
            text_chunks = [Document(page_content=chunk, metadata={"doc_id": doc_id, "source": file_path}) for chunk in chunks]
        else:
            text_chunks = [Document(page_content=chunk, metadata={"doc_id": doc_id, "source": file_path}) for chunk in self.text_splitter.split_text(cleaned_text)]

        for img in images:
            context = "\n".join([chunk.page_content for chunk in text_chunks if Path(chunk.metadata['source']).name == Path(file_path).name])
            img["caption"] = self.image_processor.generate_image_caption(img["path"], context=f"This image is from page {img['page']} of the document {Path(file_path).name}. Context: {context[:1000]}")

        return ProcessedDocument(
            doc_id=doc_id,
            text_chunks=text_chunks,
            images=images,
            metadata={"source": file_path, "type": "pdf"},
            full_text=cleaned_text
        )

    def process_html(self, content: str, url: str) -> ProcessedDocument:
        """Processes HTML content, extracting text and images."""
        soup = BeautifulSoup(content, 'html.parser')
        doc_id = str(uuid.uuid4())
        temp_dir = Path(tempfile.mkdtemp(prefix=f"html_{doc_id}_"))
        images = []

        # Find, download, and save images
        for img_tag in soup.find_all('img'):
            img_src = img_tag.get('src')
            if not img_src:
                continue
            try:
                img_url = urljoin(url, img_src)
                response = requests.get(img_url, stream=True, timeout=15)
                response.raise_for_status()
                content_type = response.headers.get('content-type')
                ext = mimetypes.guess_extension(content_type) if content_type else Path(img_url).suffix
                if not ext:
                    ext = '.jpg'  # Default extension
                image_path = temp_dir / f"img_{len(images) + 1}{ext}"
                with open(image_path, "wb") as f:
                    for chunk in response.iter_content(8192):
                        f.write(chunk)
                images.append({"path": str(image_path), "page": 1, "caption": ""})
            except requests.RequestException as e:
                logger.warning(f"Could not download image {img_src} from {url}: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing an image from {url}: {e}")

        # Extract and clean text
        text = soup.get_text(separator='\n', strip=True)
        cleaned_text = custom_clean_text(text)
        # Optionally use semantic chunking if tokenizer and nlp are provided
        if self.tokenizer and self.nlp:
            chunks = split_into_sentences(cleaned_text, self.tokenizer, nlp=self.nlp)
            text_chunks = [Document(page_content=chunk, metadata={"doc_id": doc_id, "source": url}) for chunk in chunks]
        else:
            text_chunks = [Document(page_content=chunk, metadata={"doc_id": doc_id, "source": url}) for chunk in self.text_splitter.split_text(cleaned_text)]

        # Generate captions for downloaded images
        for img in images:
            context = cleaned_text[:1500]  # Use the beginning of the article as context
            img["caption"] = self.image_processor.generate_image_caption(
                img["path"], 
                context=f"This image is from the webpage {url}. Context from the page: {context}"
            )

        return ProcessedDocument(
            doc_id=doc_id,
            text_chunks=text_chunks,
            images=images,
            metadata={"source": url, "type": "html"},
            full_text=cleaned_text
        )
