#file_process_manager

import os
import shutil
from uuid import uuid4
from file_loader import pdf_loader, txt_loader, docx_loader, html_loader, csv_loader, json_loader, pptx_loader
from database_manager import DatabaseManager
from TxtSplitter import *
from ocr import ImageTextExtractor  # Classe pour l'extraction d'images et de texte
from config import LANGUAGES, MODEL, PROCESSED_DIR, API_KEY


def process_and_add_to_db(db, loader, file_path, treated_folder= PROCESSED_DIR):
    # Charger le document
    documents = loader.load()
    source_name = os.path.basename(file_path)
    ''' 
    # Initialiser l'extracteur d'images et de texte pour les images
    image_extractor = ImageTextExtractor()
    
    # Extraire les images du PDF et obtenir le texte
    images = image_extractor.extract_images_from_pdf(file_path)
    images_text = ""
    for image_stream in images:
        images_text += image_extractor.extract_text(image_stream) + " "
    # Ajouter uniquement le texte extrait des images au contenu principal du document
    for doc in documents:
        doc.metadata["source"] = source_name
        doc.metadata["images_text"] = images_text.strip()  # Ajouter le texte extrait des images comme chaîne de caractères
        
        # Fusionner `images_text` avec le texte principal du document
        if images_text.strip():
            doc.page_content += " " + images_text.strip()
'''
    # Segmenter le texte en chunks basés sur les tokens
    splitter = TokenBasedSentenceSplitter(chunk_size=800, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    
    texts = []
    metadatas = []
    for chunk in split_docs:
        chunk.metadata["id"] = str(uuid4())
        chunk.metadata["original_source"] = source_name
        texts.append(chunk.page_content)
        metadatas.append(chunk.metadata)
    
    # Vérifier et insérer les textes et métadonnées dans la base de données
    if texts and metadatas:
        db.add_texts(texts=texts, metadatas=metadatas)
    else:
        print("Pas de textes ou de métadonnées valides à insérer dans la base de données.")
    
    # Copier le document dans le dossier 'Documents traités'
    destination_path = os.path.join(treated_folder, source_name)
    if not os.path.exists(treated_folder):
        os.makedirs(treated_folder)
    shutil.copy(file_path, destination_path)
    
    print(f"Fichier '{file_path}' ajouté à la base de données et copié dans '{treated_folder}'.")




def add_documents(file_paths, treated_folder=PROCESSED_DIR):
    """
    Ajoute un ou plusieurs documents dans la base de données en vérifiant qu'ils ne sont pas déjà dans 'Documents traités'.
    
    Args:
        db: L'objet ChromaDB.
        file_paths (list): Liste des chemins des fichiers à ajouter.
        treated_folder (str): Dossier où les fichiers traités sont stockés.
    """
    db_manager = DatabaseManager()
    db = db_manager.get_database()

    for file_path in file_paths:
        source_name = os.path.basename(file_path)
        treated_path = os.path.join(treated_folder, source_name)
        
        # Vérifie si le document est déjà présent dans 'Documents traités'
        if os.path.exists(treated_path):
            print(f"Le fichier '{source_name}' est déjà présent dans '{treated_folder}' et ne sera pas ajouté.")
            continue

        # Détermine le type de fichier et appelle la fonction appropriée
        if file_path.endswith(".pdf"):
            loader = pdf_loader(file_path)
            process_and_add_to_db(db, loader, file_path, treated_folder)
        elif file_path.endswith(".txt"):
            loader = txt_loader(file_path)
            process_and_add_to_db(db, loader, file_path, treated_folder)
        elif file_path.endswith(".docx"):
            loader = docx_loader(file_path)
            process_and_add_to_db(db, loader, file_path, treated_folder)
        elif file_path.endswith(".html"):
            loader = html_loader(file_path)
            process_and_add_to_db(db, loader, file_path, treated_folder)
        elif file_path.endswith(".csv"):
            loader = csv_loader(file_path)
            process_and_add_to_db(db, loader, file_path, treated_folder)
        elif file_path.endswith(".json"):
            loader = json_loader(file_path)
            process_and_add_to_db(db, loader, file_path, treated_folder)
        elif file_path.endswith(".pptx"):
            loader = pptx_loader(file_path)
            process_and_add_to_db(db, loader, file_path, treated_folder)
        else:
            print(f"Type de fichier non pris en charge pour '{file_path}'.")


def delete_documents(source_values, treated_folder=PROCESSED_DIR):
    """
    Supprime tous les documents dans la base de données `db` ayant une certaine valeur pour la métadonnée 'source'.
    
    Args:
        db: L'objet ChromaDB (base de données).
        source_values (list): La valeur de la métadonnée 'source' pour filtrer les documents à supprimer.
    """
    
    db_manager = DatabaseManager()
    db = db_manager.get_database()

    for source_value in source_values:
        try:
            # Récupérer les IDs des documents à supprimer
            ids = db.get(where={'source' : source_value})['ids']
            db.delete(ids)
            db.persist()  # Sauvegarder la collection de manière persistante
            print(f"PDF '{source_value}' supprimé de la base de données avec succès.")
            
            # Supprimer le document du dossier 'Documents traités'
            treated_file_path = os.path.join(treated_folder, source_value)
            if os.path.exists(treated_file_path):
                os.remove(treated_file_path)
                print(f"PDF '{source_value}' supprimé de '{treated_folder}' avec succès.")
            else:
                print(f"Le fichier '{source_value}' n'existe pas dans le dossier '{treated_folder}'.")
        
        except:
            print(f"Aucun document trouvé avec la source '{source_value}'")  