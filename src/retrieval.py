import logging
import os
import uuid
import warnings
from typing import List, Optional

import torch
import numpy as np
from dotenv import load_dotenv
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.document_processing import ProcessedDocument

# Suppress GPU Faiss warning - we're using CPU version with GPU acceleration
warnings.filterwarnings("ignore", message="Failed to load GPU Faiss")

logger = logging.getLogger(__name__)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class MultimodalRetriever:
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the multimodal retriever with optional GPU acceleration.
        
        Args:
            use_gpu: Whether to use GPU acceleration for similarity search
        """
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        self.retriever = None
        self.child_docs = []
        self.parent_docs = {}
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if self.use_gpu:
            logger.info(f"GPU acceleration enabled. Using device: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("GPU acceleration disabled. Using CPU.")
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device for tensor operations."""
        return torch.device('cuda' if self.use_gpu else 'cpu')
    
    def _compute_similarity_gpu(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity using GPU acceleration.
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Document embedding vectors
            
        Returns:
            Similarity scores
        """
        device = self._get_device()
        
        # Convert to PyTorch tensors and move to GPU
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=device)
        doc_tensor = torch.tensor(doc_embeddings, dtype=torch.float32, device=device)
        
        # Normalize vectors for cosine similarity
        query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=0)
        doc_norm = torch.nn.functional.normalize(doc_tensor, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(doc_norm, query_norm.unsqueeze(1)).squeeze(1)
        
        # Move result back to CPU and convert to numpy
        return similarity.cpu().numpy()

    def add_documents(self, processed_doc: ProcessedDocument):
        """Adds a processed document, creating parent-child links."""
        parent_doc_id = processed_doc.doc_id
        parent_content = processed_doc.full_text if processed_doc.full_text else " ".join([c.page_content for c in processed_doc.text_chunks])
        parent_document = Document(
            page_content=parent_content,
            metadata={'doc_id': parent_doc_id, 'source': processed_doc.metadata.get('source', 'Unknown')}
        )
        self.parent_docs[parent_doc_id] = parent_document

        for chunk in processed_doc.text_chunks:
            chunk.metadata['doc_id'] = parent_doc_id
            self.child_docs.append(chunk)

        for img in processed_doc.images:
            summary_doc = Document(
                page_content=img['caption'],
                metadata={'doc_id': parent_doc_id, 'source': processed_doc.metadata.get('source', 'Unknown'), 'image_path': img['path']}
            )
            self.child_docs.append(summary_doc)

    def build_index(self):
        """Builds the FAISS index and MultiVectorRetriever from all added documents."""
        if not self.child_docs:
            logger.warning("No documents to index.")
            return

        vectorstore = FAISS.from_documents(documents=self.child_docs, embedding=self.embeddings)
        store = InMemoryStore()
        store.mset(list(self.parent_docs.items()))

        self.retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key="doc_id",
        )

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents using the multi-vector retriever."""
        if not self.retriever:
            logger.warning("Retriever not built yet. Call build_index() first.")
            return []
        
        # The `get_relevant_documents` method is deprecated. The new `invoke` method
        # is recommended but doesn't accept `k` directly. We pass it via `search_kwargs`.
        # This also fixes a likely bug where the `k` argument was previously ignored.
        self.retriever.search_kwargs = {'k': k}
        return self.retriever.invoke(query)
    
    def retrieve_with_gpu_acceleration(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents with GPU-accelerated similarity search.
        This method provides GPU acceleration for the similarity computation.
        """
        if not self.retriever:
            logger.warning("Retriever not built yet. Call build_index() first.")
            return []
        
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Get all document embeddings
        doc_texts = [doc.page_content for doc in self.child_docs]
        doc_embeddings = self.embeddings.embed_documents(doc_texts)
        
        # Compute similarities using GPU acceleration
        similarities = self._compute_similarity_gpu(query_embedding, np.array(doc_embeddings))
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Retrieve documents
        results = []
        for idx in top_indices:
            doc = self.child_docs[idx]
            # Get parent document
            parent_doc_id = doc.metadata.get('doc_id')
            if parent_doc_id in self.parent_docs:
                results.append(self.parent_docs[parent_doc_id])
        
        return results
