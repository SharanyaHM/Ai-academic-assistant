import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

class PDFChunk:
    def __init__(self, text: str, page: int, chunk_id: int, source_doc: str, doc_id: Optional[str] = None, section: Optional[str] = None):
        self.text = text
        self.page = page
        self.chunk_id = chunk_id

        
        self.source_doc = source_doc
        self.doc_id = doc_id or "default_doc"
        self.section = section or f"Page {page}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "source_doc": self.source_doc,
            "doc_id": self.doc_id,
            "section": self.section
        }

class AcademicPDFProcessor:
    """
    Academic PDF Processing Engine
    Extracts text, preserves page numbers, identifies headings/topics,
    and breaks content into semantic chunks for RAG.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text page-by-page.
        Uses PyMuPDF (fitz) if available, with pypdf and pdfminer as fallbacks.
        """
        pages_data = []
        pdf_path = str(pdf_path)

        # Method 1: Try PyMuPDF (pymupdf)
        try:
            import pymupdf as fitz
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": self._clean_text(text)
                })
            doc.close()
            if pages_data:
                return pages_data
        except ImportError:
            pass
        except Exception as e:
            print(f"PyMuPDF failed: {e}")

        # Method 2: Try pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages_data.append({
                    "page_number": idx + 1,
                    "text": self._clean_text(text)
                })
            if pages_data:
                return pages_data
        except ImportError:
            pass
        except Exception as e:
            print(f"pypdf failed: {e}")

        # Method 3: Fallback plain text reading if text file uploaded
        if pdf_path.endswith(".txt") or pdf_path.endswith(".md"):
            with open(pdf_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                # Split roughly into pages of ~2500 chars
                parts = [content[i:i+2500] for i in range(0, len(content), 2500)]
                for i, part in enumerate(parts):
                    pages_data.append({
                        "page_number": i + 1,
                        "text": self._clean_text(part)
                    })
                return pages_data

        if not pages_data:
            # If all else fails, return empty list
            return []

        return pages_data

    def _clean_text(self, text: str) -> str:
        """Cleans artifacts, hyphenated line wraps, and irregular spacing."""
        if not text:
            return ""
        # Remove hyphenated linebreaks (e.g. compu-\nter -> computer)
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Normalize non-breaking spaces
        text = text.replace('\xa0', ' ')
        # Remove repeated horizontal whitespaces
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def detect_sections(self, text: str) -> List[str]:
        """Detects headings like '1. Introduction', 'Module 3: Pointers', 'Algorithm 4.1'"""
        heading_patterns = [
            r'^(?:Module|Unit|Chapter)\s+\d+[:\s\w]+$',
            r'^\d+\.\d+\s+[A-Z][\w\s]+$',
            r'^[A-Z\s]{4,}$'
        ]
        headings = []
        for line in text.splitlines():
            line_str = line.strip()
            for pattern in heading_patterns:
                if re.match(pattern, line_str, re.MULTILINE) and len(line_str) < 80:
                    headings.append(line_str)
                    break
        return headings

    def create_chunks(self, pages_data: List[Dict[str, Any]], source_doc: str, doc_id: Optional[str] = None) -> List[PDFChunk]:
        """
        Segments page texts into academic chunks with overlap,
        preserving page numbers and section context.
        """
        if not doc_id:
            import hashlib
            doc_id = hashlib.sha256(source_doc.encode("utf-8")).hexdigest()[:12]

        all_chunks = []
        chunk_counter = 1

        for p_info in pages_data:
            page_num = p_info["page_number"]
            page_text = p_info["text"]
            if not page_text or len(page_text.strip()) < 15:
                continue

            # Split into paragraphs first to avoid cutting sentences mid-way
            paragraphs = page_text.split("\n\n")
            current_buffer = ""

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                if len(current_buffer) + len(para) + 2 <= self.chunk_size:
                    current_buffer = f"{current_buffer}\n\n{para}".strip()
                else:
                    if current_buffer:
                        chunk = PDFChunk(
                            text=current_buffer,
                            page=page_num,
                            chunk_id=chunk_counter,
                            source_doc=source_doc,
                            doc_id=doc_id
                        )
                        all_chunks.append(chunk)
                        chunk_counter += 1

                    # If the paragraph itself exceeds chunk_size, split by sentences/fixed size
                    if len(para) > self.chunk_size:
                        sentences = re.split(r'(?<=[.?!])\s+', para)
                        sub_buffer = ""
                        for sent in sentences:
                            if len(sub_buffer) + len(sent) + 1 <= self.chunk_size:
                                sub_buffer = f"{sub_buffer} {sent}".strip()
                            else:
                                if sub_buffer:
                                    chunk = PDFChunk(
                                        text=sub_buffer,
                                        page=page_num,
                                        chunk_id=chunk_counter,
                                        source_doc=source_doc,
                                        doc_id=doc_id
                                    )
                                    all_chunks.append(chunk)
                                    chunk_counter += 1
                                sub_buffer = sent
                        current_buffer = sub_buffer
                    else:
                        current_buffer = para

            if current_buffer:
                chunk = PDFChunk(
                    text=current_buffer,
                    page=page_num,
                    chunk_id=chunk_counter,
                    source_doc=source_doc,
                    doc_id=doc_id
                )
                all_chunks.append(chunk)
                chunk_counter += 1

        return all_chunks

    def process_file(self, file_path: str, source_doc: Optional[str] = None, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Complete ingestion pipeline for a single file."""
        if not source_doc:
            source_doc = Path(file_path).name

        if not doc_id:
            import hashlib
            file_sig = f"{source_doc}"
            try:
                p = Path(file_path)
                file_sig = f"{source_doc}_{p.stat().st_size}_{p.stat().st_mtime}"
            except Exception:
                pass
            doc_id = hashlib.sha256(file_sig.encode("utf-8")).hexdigest()[:12]

        pages = self.extract_text_from_pdf(file_path)
        chunks = self.create_chunks(pages, source_doc, doc_id)

        return {
            "doc_id": doc_id,
            "source_doc": source_doc,
            "file_path": str(file_path),
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "pages": pages,
            "chunks": [c.to_dict() for c in chunks]
        }
