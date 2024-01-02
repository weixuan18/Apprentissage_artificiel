from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

chemin_dossier_données_prétraitées='./corpus/corpus_prétraité/'

# Supposons que 'texts' et 'labels' sont les données prétraitées et les étiquettes de parti politique
texts = []  # Les textes prétraités
labels = [] # Les étiquettes

for fichier in os.listdir(chemin_dossier_données_prétraitées):
    if fichier.endswith('.csv'):
        chemin_complet = os.path.join(chemin_dossier_données_prétraitées, fichier)
        # Lire le contenu du fichier CSV
        df = pd.read_csv(chemin_complet)

        # Assurez-vous que les colonnes 'text' et 'parti' existent
        if 'text' in df.columns and 'parti' in df.columns:
            # Remplacer les NaN par des chaînes vides dans la colonne 'text'
            df['text'] = df['text'].fillna('')

            texts.extend(df['text'].tolist())
            labels.extend(df['parti'].tolist())

# Vectorisation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Choix et entraînement du modèle
model = LogisticRegression()
model.fit(X_train, y_train)

######### Évaluation du modèle ##########
# Prédiction sur l'ensemble de test
predictions = model.predict(X_test)

# Affichage du rapport de classification
report = classification_report(y_test, predictions)
print(report)
# Stocker le rapport de classification dans un fichier
with open('./résultats/rapport_classification.txt', 'w') as file:
    file.write(report)

# Création de la matrice de confusion
conf_matrix = confusion_matrix(y_test, predictions)

# Affichage de la matrice de confusion avec Seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Prédictions')
plt.ylabel('Valeurs Réelles')
plt.title('Matrice de Confusion_Logistic Regression')
# Sauvegarder la figure en PNG
plt.savefig('./résultats/matrice_de_confusion_lg.png', bbox_inches='tight', dpi=300)

plt.show()

