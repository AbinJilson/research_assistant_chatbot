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

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.schema.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image

from src.utils import _import_image_dependencies, on_rm_error, cv2, np

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
            prompt = f"Analyze this image and provide a detailed description. {context}"
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
    def __init__(self, image_processor: ImageProcessor):
        self.image_processor = image_processor
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

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

        text_chunks = [Document(page_content=chunk, metadata={"doc_id": doc_id, "source": file_path}) for chunk in self.text_splitter.split_text(full_text)]

        for img in images:
            context = "\n".join([chunk.page_content for chunk in text_chunks if Path(chunk.metadata['source']).name == Path(file_path).name])
            img["caption"] = self.image_processor.generate_image_caption(img["path"], context=f"This image is from page {img['page']} of the document {Path(file_path).name}. Context: {context[:1000]}")

        return ProcessedDocument(
            doc_id=doc_id,
            text_chunks=text_chunks,
            images=images,
            metadata={"source": file_path, "type": "pdf"},
            full_text=full_text
        )

    def process_html(self, content: str, url: str) -> ProcessedDocument:
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        doc_id = str(uuid.uuid4())
        text_chunks = [Document(page_content=chunk, metadata={"doc_id": doc_id, "source": url}) for chunk in self.text_splitter.split_text(text)]
        return ProcessedDocument(
            doc_id=doc_id,
            text_chunks=text_chunks,
            images=[],
            metadata={"source": url, "type": "html"},
            full_text=text
        )
