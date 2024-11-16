from langchain.text_splitter import TextSplitter
from langchain.schema import Document
import tiktoken
from config import MODEL

def count_tokens(text, model=MODEL):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

class TokenBasedSentenceSplitter(TextSplitter):
    def __init__(self, chunk_size=800, chunk_overlap=50, model=MODEL):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunk_size = chunk_size
        self.model = model

    def split_text(self, text):
        sentences = text.split('.')
        chunks = []
        current_chunk = []

        for sentence in sentences:
            if sentence.strip():  # Ignore les phrases vides
                temp_chunk = ' '.join(current_chunk + [sentence.strip() + '.'])
                if count_tokens(temp_chunk, self.model) > self.chunk_size:
                    # Si le chunk dépasse la taille en tokens, ajoute le chunk actuel et recommence
                    chunks.append(' '.join(current_chunk).strip())
                    current_chunk = [sentence.strip() + '.']
                else:
                    # Sinon, ajoute la phrase au chunk en cours
                    current_chunk.append(sentence.strip() + '.')

        # Ajoute le dernier chunk s'il reste du texte
        if current_chunk:
            chunks.append(' '.join(current_chunk).strip())

        return chunks

    def split_documents(self, documents):
        split_docs = []
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            for chunk in text_chunks:
                # Transforme chaque chunk en objet Document avec les métadonnées de la page d'origine
                split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
        return split_docs