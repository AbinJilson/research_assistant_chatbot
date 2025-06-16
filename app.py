import logging
import os
import tempfile
import asyncio
import platform
from pathlib import Path

# Set Windows event loop policy if on Windows
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from dotenv import load_dotenv

from src.assistant_logic import MultimodalResearchAssistant
from src.utils import run_async_in_thread

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    """The main Streamlit application."""
    st.set_page_config(page_title="Multimodal Research Assistant", layout="wide")



    # Initialize assistant in session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = MultimodalResearchAssistant()
        logger.info("Initialized new MultimodalResearchAssistant object.")

    # --- Sidebar for document upload and management ---
    with st.sidebar:
        st.header("📚 Add to Knowledge Base")

        # PDF Upload
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to add them to the knowledge base."
        )
        if uploaded_files:
            for file in uploaded_files:
                file_key = f"processed_{file.file_id}_{file.size}"
                if not st.session_state.get(file_key, False):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                    
                    st.session_state.assistant.add_pdf(tmp_path)
                    os.unlink(tmp_path)
                    st.session_state[file_key] = True
                    st.success(f"✅ Added {file.name}")

        # URL Input
        url_input = st.text_input(
            "Add from URL (Article or PDF)",
            placeholder="https://example.com/article",
            help="Enter a URL to a web page or a direct link to a PDF."
        )
        if st.button("Add URL") and url_input:
            url_key = f"processed_{url_input}"
            if not st.session_state.get(url_key, False):
                st.session_state.assistant.add_url(url_input)
                st.session_state[url_key] = True
                st.success(f"✅ Added content from URL")

    # --- Main content area ---
    st.title("🧠 Multimodal Research Assistant")
    st.markdown("Ask questions about your documents and web pages. The assistant can analyze text, charts, and images.")



    st.header("💬 Ask Questions")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.write("""
        - What are the main findings in the uploaded documents?
        - Summarize the key points from all sources
        - What charts or data visualizations are mentioned?
        - Compare information across different documents
        - What are the conclusions or recommendations?
        """)

    query_input = st.text_area(
        "Enter your question:", 
        placeholder="What would you like to know about your documents?",
        height=100
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("🔍 Search", type="primary", disabled=not query_input)

    if search_button and query_input:
        if not st.session_state.assistant.processed_docs:
            st.warning("⚠️ Please add some documents first before asking questions.")
        else:
            with st.spinner("Searching and analyzing..."):
                try:
                    result = st.session_state.assistant.query(query_input)
                    st.session_state.last_result = result
                    st.session_state.last_query = query_input
                except Exception as e:
                    st.error(f"❌ Search failed: {str(e)}")

    # Display results
    if 'last_result' in st.session_state:
        result = st.session_state.last_result
        query = st.session_state.get('last_query', 'Previous query')
        
        st.header("📝 Results")
        st.subheader(f"Question: {query}")
        
        # Answer
        st.markdown("### Answer")
        st.write(result["answer"])

        # Sources
        if result["relevant_docs"]:
            st.markdown("### 📚 Sources")
            
            for i, doc_data in enumerate(result["relevant_docs"], 1):
                try:
                    metadata = doc_data.get("metadata", {})
                    content = doc_data.get("page_content", "No content available")
                    
                    source_name = metadata.get('source', 'Unknown Source')
                    doc_type = metadata.get('type', 'unknown')
                    
                    # Format source display
                    if doc_type == 'image_caption':
                        source_title = f"🖼️ Image from {Path(source_name).name}"
                        if 'page' in metadata:
                            source_title += f" (Page {metadata['page']})"
                    else:
                        source_title = f"📄 {Path(source_name).name}"
                    
                    with st.expander(f"Source {i}: {source_title}"):
                        st.write("**Content:**")
                        # Truncate long content
                        display_content = content[:800] + "..." if len(content) > 800 else content
                        st.write(display_content)
                        
                        if st.checkbox(f"Show metadata for source {i}", key=f"meta_{i}"):
                            st.json(metadata)
                            
                except Exception as e:
                    st.error(f"Error displaying source {i}: {e}")

        # Option to show full context
        if st.checkbox("Show full retrieved context"):
            st.subheader("📄 Retrieved Context")
            st.text_area("Full Context", result["context"], height=300, key="context_display")

    # Document details section
    if st.session_state.assistant.processed_docs:
        st.header("📋 Document Library")
        
        with st.expander("View Document Details"):
            for i, doc in enumerate(st.session_state.assistant.processed_docs):
                source_name = Path(doc.metadata.get('source', 'Unknown')).name
                doc_type = doc.metadata.get('type', 'Unknown').upper()
                
                st.write(f"**📄 Document {i+1}: {source_name}** ({doc_type})")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Text Chunks", len(doc.text_chunks))
                with col2:
                    st.metric("Images", len(doc.images))
                with col3:
                    st.metric("Total Content", 
                             f"{sum(len(chunk.page_content) for chunk in doc.text_chunks):,} chars")
                
                if doc.images:
                    st.write("**🖼️ Image Summaries:**")
                    for j, img in enumerate(doc.images[:3], 1):  # Show first 3 images
                        caption = img['caption']
                        truncated_caption = caption[:150] + "..." if len(caption) > 150 else caption
                        st.write(f"  {j}. {truncated_caption}")
                    
                    if len(doc.images) > 3:
                        st.write(f"  ... and {len(doc.images) - 3} more images")
                
                st.write("---")

    # Tips and help
    with st.sidebar:
        st.markdown("---")
        st.subheader("💡 Tips")
        st.markdown("""
        **For best results:**
        - Upload clear, text-rich PDFs
        - Use specific questions
        - Ask about comparisons across documents
        - Reference charts, tables, or images
        
        **Supported formats:**
        - PDF files with text and images
        - Web pages and articles
        - Research papers and reports
        """)
        
        if st.button("🗑️ Clear All Documents"):
            if st.session_state.get('confirm_clear', False):
                # Reset the assistant
                st.session_state.assistant = MultimodalResearchAssistant()
                # Clear processed file tracking
                keys_to_remove = [k for k in st.session_state.keys() if k.startswith('processed_')]
                for key in keys_to_remove:
                    del st.session_state[key]
                # Clear results
                if 'last_result' in st.session_state:
                    del st.session_state['last_result']
                if 'last_query' in st.session_state:
                    del st.session_state['last_query']
                st.session_state.confirm_clear = False
                st.success("🗑️ All documents cleared!")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm clearing all documents")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}", exc_info=True)