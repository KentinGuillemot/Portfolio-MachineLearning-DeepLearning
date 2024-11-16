import os  # Le module os est utilisé pour manipuler les chemins de fichiers de manière dynamique
import getpass

# Définir la clé API OpenAI si elle n'est pas déjà dans les variables d'environnement
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# Maintenant, elle peut être utilisée dans le reste du projet
API_KEY = os.getenv("OPENAI_API_KEY")

# Modèle OpenAI
# Ce paramètre indique le modèle que tu veux utiliser avec l'API OpenAI.
# Par exemple, 'gpt-4o' peut être modifié si tu veux tester d'autres modèles comme 'gpt-3.5'.

MODEL="gpt-4o"

# Chemins vers les données
# Le chemin de base du projet est défini à partir du répertoire actuel où se trouve ce fichier.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ce dossier "data" contient toutes les ressources de données telles que les documents, les segments, etc.
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Chemins spécifiques vers différents types de données dans le dossier 'data'.
# DOCUMENTS_DIR : Répertoire où seront placés les documents bruts (PDF, DOCX, TXT).
DOCUMENTS_DIR = os.path.join(DATA_DIR, 'Documents - Tests')

# PROCESSED_DIR : Répertoire où seront stockés les documents déjà traités ou segmentés.
PROCESSED_DIR = os.path.join(DATA_DIR, 'Documents traités')

# DOC_IDS_PATH : Dossier où est stocké la database Chroma.
DATA_BASE_DIR = os.path.join(DATA_DIR, 'data_base_folder')

# Langues 
LANGUAGES = ['fr']

# Définir le chemin du fichier prompt et charger son contenu
PROMPT_FILE_PATH = os.path.join(BASE_DIR, "prompt.txt")
with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as file:
    PROMPT = file.read()