import os
import pandas as pd
import spacy
from langdetect import detect

# Charger les modèles spaCy pour chaque langue
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")
nlp_it = spacy.load("it_core_news_sm")

def preprocess_text(text, lang):
    if not isinstance(text, str):
        return ""  # Retourner une chaîne vide si le texte n'est pas une chaîne de caractères
    
    # Sélectionner le modèle spaCy en fonction de la langue
    if lang == 'en':
        nlp = nlp_en
    elif lang == 'fr':
        nlp = nlp_fr
    elif lang == 'it':
        nlp = nlp_it
    else:
        return text  # Retourner le texte non modifié si la langue n'est pas supportée

    # Créer un document spaCy
    doc = nlp(text)

    # Tokenisation et nettoyage
    cleaned_text = " ".join(
        [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space])

    return cleaned_text


# Dossier contenant les fichiers CSV (à modifier)
directory = "./corpus/corpus_extraits/test/"

# Itérer sur tous les fichiers CSV dans le dossier
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)

        # Charger le fichier CSV
        df = pd.read_csv(file_path)

        # Détecter la langue du premier texte non vide dans le fichier
        lang = detect(next(text for text in df['Text'] if text.strip()))

        # Prétraiter chaque texte dans la colonne 'text'
        df['text_cleaned'] = df['Text'].apply(lambda x: preprocess_text(x, lang))

        # Sauvegarder le DataFrame prétraité (à modifier)
        df.to_csv(os.path.join(directory, f"prétraite_test_{lang}.csv"), index=False)
