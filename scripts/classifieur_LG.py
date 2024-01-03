from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_and_evaluate(X_train, X_test, y_train, y_test, description):
    # Entraînement du modèle
    model = LogisticRegression()
    model.fit(X_train, y_train)

    #################### Évaluation du modèle ####################
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)

    # Enregistrement du rapport de classification
    with open(f'./résultats/Régression_logistique/rapport_classification_{description}_LG.txt', 'w') as file:
        file.write(report)

    # Création et affichage de la matrice de confusion
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs Réelles')
    plt.title(f'Matrice de Confusion - {description}')
    plt.savefig(f'./résultats/Régression_logistique/matrice_de_confusion_{description}_LG.png', bbox_inches='tight', dpi=300)
    plt.show()

def load_data(language):
    # Charger les données d'entraînement et de test pour une langue spécifique
    train_df = pd.read_csv(f'./corpus/corpus_prétraité/train/prétraite_train_{language}.csv')
    test_df = pd.read_csv(f'./corpus/corpus_prétraité/test/prétraite_test_{language}.csv')

    # Remplacer les NaN par des chaînes vides dans les colonnes de texte
    train_df['text_cleaned'].fillna('', inplace=True)
    test_df['text_cleaned'].fillna('', inplace=True)

    # Vectorisation
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_df['text_cleaned'])
    X_test = vectorizer.transform(test_df['text_cleaned'])

    y_train = train_df['Label']
    y_test = test_df['Label']

    return X_train, X_test, y_train, y_test

def load_combined_data():
    # Charger et combiner les données pour toutes les langues
    train_combined = pd.DataFrame()
    test_combined = pd.DataFrame()

    for lang in ['en', 'fr', 'it']:
        train_df = pd.read_csv(f'./corpus/corpus_prétraité/train/prétraite_train_{lang}.csv')
        test_df = pd.read_csv(f'./corpus/corpus_prétraité/test/prétraite_test_{lang}.csv')
        train_combined = pd.concat([train_combined, train_df], ignore_index=True)
        test_combined = pd.concat([test_combined, test_df], ignore_index=True)

    # Gérer les NaN et vectoriser pour l'ensemble combiné
    train_combined['text_cleaned'].fillna('', inplace=True)
    test_combined['text_cleaned'].fillna('', inplace=True)

    # Vectorisation
    vectorizer = TfidfVectorizer()
    X_train_combined = vectorizer.fit_transform(train_combined['text_cleaned'])
    X_test_combined = vectorizer.transform(test_combined['text_cleaned'])

    y_train_combined = train_combined['Label']
    y_test_combined = test_combined['Label']

    return X_train_combined, X_test_combined, y_train_combined, y_test_combined


# # Pour chaque langue
# for lang in ['en', 'fr', 'it']:
#     X_train, X_test, y_train, y_test = load_data(lang)
#     process_and_evaluate(X_train, X_test, y_train, y_test, lang)

# Traiter l'ensemble combiné de toutes les langues
X_train_combined, X_test_combined, y_train_combined, y_test_combined = load_combined_data()
process_and_evaluate(X_train_combined, X_test_combined, y_train_combined, y_test_combined, "ensemble_combiné")