# file_loader.py

from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader, CSVLoader, JSONLoader, Docx2txtLoader, UnstructuredPowerPointLoader


def pdf_loader(pdf_path):
    """Ajoute un PDF à la base de données ChromaDB avec des IDs uniques pour chaque chunk."""
    loader = PyPDFLoader(pdf_path)
    return loader

def txt_loader(txt_path):
    """Ajoute un fichier TXT à la base de données ChromaDB avec des IDs uniques pour chaque chunk."""
    loader = TextLoader(txt_path)
    return loader

def docx_loader(docx_path):
    """Ajoute un fichier DOCX à la base de données ChromaDB avec des IDs uniques pour chaque chunk."""
    loader = Docx2txtLoader(docx_path)
    return loader

def html_loader(html_path):
    """Ajoute un fichier HTML à la base de données ChromaDB avec des IDs uniques pour chaque chunk."""
    loader = UnstructuredHTMLLoader(html_path)
    return loader

def csv_loader(csv_path):
    """Ajoute un fichier CSV à la base de données ChromaDB avec des IDs uniques pour chaque chunk."""
    loader = CSVLoader(csv_path)
    return loader

def json_loader(json_path):
    """Ajoute un fichier JSON à la base de données ChromaDB avec des IDs uniques pour chaque chunk."""
    loader = JSONLoader(json_path)
    return loader

def pptx_loader(pptx_path):
    """Ajoute un fichier PowerPoint (PPTX) à la base de données ChromaDB avec des IDs uniques pour chaque chunk."""
    loader = UnstructuredPowerPointLoader(pptx_path)
    return loader