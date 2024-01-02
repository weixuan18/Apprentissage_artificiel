import xml.etree.ElementTree as ET
import glob
import csv
import os

# Chemin vers le dossier contenant les fichiers XML
chemin_dossier = './corpus/parlement_multilingue/*xml'

# Parcourir tous les fichiers XML dans le dossier
for fichier_xml in glob.glob(chemin_dossier):
    tree = ET.parse(fichier_xml)
    root = tree.getroot()

    # Nom du fichier CSV basé sur le nom du fichier XML
    nom_base = os.path.basename(fichier_xml)
    chemin_csv = f'./corpus/corpus_extraits/{os.path.splitext(nom_base)[0]}.csv'


    # Structure pour stocker les données extraites du fichier XML courant
    data_fichier = []

    for doc in root.findall('.//doc'):
        doc_id = doc.get('id')
        for eval_parti in doc.findall('.//EVAL_PARTI'):
            parti = eval_parti.find('.//PARTI').get('valeur')
            confiance = eval_parti.find('.//PARTI').get('confiance')
            text = " ".join([p.text for p in doc.findall('.//texte/p') if p.text])

            data_fichier.append({
                #'fichier': fichier_xml,
                #'doc_id': doc_id,
                'parti': parti,
                #'confiance': confiance,
                'text': text
            })

    # Enregistrer les données dans un fichier CSV pour le fichier XML courant
    with open(chemin_csv, mode='w', newline='', encoding='utf-8') as fichier_csv:
        #writer = csv.DictWriter(fichier_csv, fieldnames=['fichier', 'doc_id', 'parti', 'confiance', 'text'])
        writer = csv.DictWriter(fichier_csv, fieldnames=['parti', 'text'])
        writer.writeheader()
        for data in data_fichier:
            writer.writerow(data)
