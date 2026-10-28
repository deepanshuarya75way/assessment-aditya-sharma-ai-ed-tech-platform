from ..load_documents.text_splitting import text_splitter
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
try:
    converter = DocumentConverter()
except Exception as e:
    converter = None
    print(f"Failed to initialize DocumentConverter: {e}")

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "load_documents" / "uploads"




def process_document(file_path_str: str, current_user: int, filename: str):
    ext = Path(file_path_str).suffix.lower()
    
    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path_str)
        docs = loader.load()
        markdown = "\n\n".join([d.page_content for d in docs])
    else:
        if not converter:
            raise RuntimeError("DocumentConverter is not initialized")
        
        result = converter.convert(file_path_str)
        markdown = result.document.export_to_markdown()

    if not markdown.strip():
        raise ValueError("Uploaded document did not contain readable text")

    doc = Document(
        page_content=markdown,
        metadata={
            "id": current_user,
            "source": filename,
            "title": filename
        }
    )

    chunks = text_splitter.split_documents([doc])

    if not chunks:
        raise ValueError("No text chunks were created from this document")

    return chunks