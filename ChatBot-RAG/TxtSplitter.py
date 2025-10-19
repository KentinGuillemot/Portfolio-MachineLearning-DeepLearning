from langchain.text_splitter import TextSplitter
from langchain.schema import Document
import tiktoken
from ocr import ImageTextExtractor 
from io import BytesIO  # Pour gérer les images en flux binaire
from config import MODEL

def count_tokens(text, model=MODEL):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

class TokenBasedSentenceSplitter(TextSplitter):
    def __init__(self, chunk_size=800, chunk_overlap=50, model=MODEL):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunk_size = chunk_size
        self.model = model
        self.image_text_extractor = ImageTextExtractor() 

    def split_text(self, text):
        sentences = text.split('.')
        chunks = []
        current_chunk = []

        for sentence in sentences:
            if sentence.strip():
                temp_chunk = ' '.join(current_chunk + [sentence.strip() + '.'])
                if count_tokens(temp_chunk, self.model) > self.chunk_size:
                    chunks.append(' '.join(current_chunk).strip())
                    current_chunk = [sentence.strip() + '.']
                else:
                    current_chunk.append(sentence.strip() + '.')

        if current_chunk:
            chunks.append(' '.join(current_chunk).strip())

        return chunks

    def split_documents(self, documents):
        split_docs = []
        for doc in documents:
            images = doc.metadata.get("images", [])
            
            # Vérifier la liste des images
            if not images:
                print("Aucune image trouvée dans les métadonnées.")
            else:
                print(f"{len(images)} image(s) trouvée(s) dans les métadonnées.")
            
            image_text = ""

            # Extrait le texte de chaque image en utilisant ImageTextExtractor
            for idx, image_stream in enumerate(images):
                if isinstance(image_stream, BytesIO):
                    print(f"Image {idx+1} est de type BytesIO, extraction du texte...")
                    extracted_text = self.image_text_extractor.extract_text(image_stream)
                    image_text += " " + extracted_text 
                    print(f"Texte extrait de l'image {idx+1} : {extracted_text}")
                else:
                    print(f"L'image {idx+1} n'est pas un flux BytesIO, type actuel : {type(image_stream)}. Ignorée.")

            # Ajoute le texte extrait des images au contenu de la page, s'il existe
            if image_text.strip():
                print(f"Texte extrait des images : {image_text}")
                doc.page_content += " " + image_text
            else:
                print("Aucun texte extrait des images pour ce document.")
            
            # Divise le texte en chunks
            text_chunks = self.split_text(doc.page_content)
            
            # Vérifie que text_chunks n'est pas vide avant l'insertion
            if text_chunks:
                for chunk in text_chunks:
                    split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
            else:
                print("Aucun contenu textuel valide trouvé dans ce document (texte ou images).")

        return split_docs
