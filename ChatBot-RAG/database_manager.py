from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from config import API_KEY, DATA_BASE_DIR

class DatabaseManager:
    def __init__(self, api_key=API_KEY, persist_directory=DATA_BASE_DIR):
        self.api_key = api_key
        self.persist_directory = persist_directory
        self.db = None

        self.initialize_database()

    def initialize_database(self):
        
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialise les embeddings et ChromaDB
        embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.db = Chroma(embedding_function=embeddings, persist_directory=self.persist_directory)

    def get_database(self):
        # Retourne l'instance de la base de données
        return self.db
