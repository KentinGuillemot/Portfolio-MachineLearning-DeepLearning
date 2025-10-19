from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI 
from langchain.memory import ConversationBufferMemory 
from langchain.chains import LLMChain 
from uuid import uuid4
from database_manager import DatabaseManager

from config import MODEL, API_KEY, PROMPT


class Chatbot:
    def __init__(self, key= API_KEY, max_results=5):
        """
        Initialise la classe Chatbot avec un UUID unique pour l'instance, une base de données et la mémoire.

        Args:
            db: Instance de la base de données ChromaDB.
            key: Clé API pour accéder au modèle OpenAI GPT-4o.
            max_results (int): Nombre maximum de résultats pour chaque recherche.
        """
        db_manager = DatabaseManager()
        self.db = db_manager.get_database()
        self.key = key
        self.max_results = max_results
        self.memory = ConversationBufferMemory(memory_key="history")
        self.query_id = str(uuid4())

        # Définir un template de prompt avec une seule variable d'entrée
        self.prompt_template = PromptTemplate(
            input_variables=["combined_input"],
            template=f"""
                Tu es un assistant intelligent spécialisé dans la gestion de documents et des informations stockées en mémoire. Il est impératif que tu répondes uniquement en utilisant les informations fournies dans les documents associés ou celles mémorisées au cours des interactions.

                **Instructions strictes :**
                1. **Réponse obligatoire basée sur les documents ou la mémoire** : Si tu ne trouves aucune information pertinente dans les documents fournis ou dans la mémoire pour répondre à la question, **n'essaie pas de deviner** ou de formuler une réponse basée sur des connaissances externes. Réponds uniquement par : "Je suis désolé mais cette question sort de mon domaine de compétence."
                2. **Utilisation de la mémoire** : Si des informations pertinentes ont été stockées en mémoire lors des échanges précédents, tu peux les utiliser pour formuler une réponse. Cependant, ne complète jamais avec des connaissances externes non mémorisées ou non présentes dans les documents.
                3. **Langue de réponse adaptée** : Réponds dans la langue de la question posée par l'utilisateur. Si la question est en français, réponds en français. Si la question est en anglais, réponds en anglais, etc.
                4. **Format de réponse :**
                - Réponds de manière concise et directe.

                **IMPORTANT :**
                - Si les informations demandées ne se trouvent ni dans les documents ni dans la mémoire, ne donne aucune réponse. Utilise uniquement la phrase : "Je suis désolé mais cette question sort de mon domaine de compétence."

                {{"combined_input"}}
            """
        )



    def search_context(self, query_text):
        """
        Effectue une recherche de texte dans la base de données pour obtenir le contexte.

        Args:
            query_text (str): Texte de la requête.

        Returns:
            str: Contexte obtenu à partir de la base de données.
        """
        # Récupération des résultats de la base de données via similarity_search
        results = self.db.similarity_search(
            query=query_text,
            k=self.max_results
        )
        # Concaténer le contenu des résultats pour former le contexte
        context = "\n".join(result.page_content for result in results)
        return context

    def ask(self, question):
        """
        Pose une question au chatbot avec un prompt strict et envoie le prompt à OpenAI.

        Args:
            question (str): La question à poser au chatbot.

        Returns:
            dict: Dictionnaire contenant l'identifiant de la requête, la question, et la réponse complète.
        """
        # Initialiser le modèle avec la clé API et GPT-4o
        llm = ChatOpenAI(api_key=self.key, model=MODEL)

        # Récupérer le contexte via une recherche dans la base de données
        context = self.search_context(query_text=question)

        # Limiter la mémoire aux 10 derniers échanges
        memory_buffer = self.memory.chat_memory.messages[-10:]

        # Créer une seule chaîne de caractères combinée pour l'entrée
        combined_input = f"""
        Historique de la conversation :
        {memory_buffer}

        Contexte :
        {context}

        Question :
        {question}
        """

        # Définir le prompt template
        prompt_template = PromptTemplate(
            input_variables=["combined_input"],
            template="""
                Tu es un assistant spécialisé dans l'extraction d'informations des documents fournis. Réponds uniquement à partir des informations présentes dans le contexte ci-dessous :

                {combined_input}
            """
        )

        # Exécuter le pipeline avec le modèle et le prompt
        answer = (prompt_template | llm).invoke({"combined_input": combined_input})

        # Mettre à jour la mémoire avec la nouvelle question et réponse
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(answer)

        # Vérifier et limiter la mémoire aux 10 derniers messages en supprimant les anciens
        if len(self.memory.chat_memory.messages) > 10:
            self.memory.chat_memory.messages = self.memory.chat_memory.messages[-10:]

        return {
            "query_id": self.query_id,
            "question": question,
            "answer": answer
        }