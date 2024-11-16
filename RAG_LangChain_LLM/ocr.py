import easyocr
import numpy as np
from PIL import Image
import fitz  # PyMuPDF pour l'extraction d'images
from io import BytesIO
from config import LANGUAGES

class ImageTextExtractor:
    def __init__(self):
        self.reader = easyocr.Reader(lang_list= LANGUAGES, gpu=True)

    def extract_text(self, image_stream):
        """
        Extrait le texte d'une image en utilisant EasyOCR.

        :param image_stream: Flux d'image (BytesIO)
        :return: Texte extrait de l'image
        """
        try:
            image = Image.open(image_stream)  # Ouvrir l'image à partir du flux binaire
            image_np = np.array(image)  # Convertir l'image PIL en tableau numpy compatible avec EasyOCR
            print(f"Extrait l'image avec taille : {image_np.shape}")  # Vérifier les dimensions de l'image
            result = self.reader.readtext(image_np, detail=0)  # Utiliser EasyOCR pour extraire le texte
            extracted_text = " ".join(result)
            print(f"Texte extrait de l'image : {extracted_text}")
            return extracted_text
        except Exception as e:
            print(f"Erreur lors de l'extraction de texte : {e}")
            return ""

    def extract_images_from_pdf(self, pdf_path):
        """
        Extrait les images d'un fichier PDF et les renvoie sous forme de flux BytesIO.

        :param pdf_path: Chemin du fichier PDF
        :return: Liste des images sous forme de flux BytesIO
        """
        images = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_stream = BytesIO(image_bytes)
                images.append(image_stream)
                print(f"Image extraite de la page {page_num + 1}, index {img_index + 1}")
        
        return images
