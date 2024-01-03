from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

####################### entraînement du modèle régression logistique #####################################
def process_and_evaluate(X_train, X_test, y_train, y_test, description):
    # Entraînement du modèle
    model = LogisticRegression(penalty='l2', solver='sag', multi_class='multinomial', max_iter=1000, verbose=1, n_jobs=-1, random_state=42, tol=0.001, C=100.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None)
    model.fit(X_train, y_train)

    #################### Évaluation du modèle ####################
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    print(report)
    # Enregistrement du rapport de classification
    with open(f'./résultats/Régression_logistique/rapport_classification_{description}_RL.txt', 'w') as file:
        file.write(report)

    #Création et affichage de la matrice de confusion
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs Réelles')
    plt.title(f'Matrice de Confusion - {description}')
    plt.savefig(f'./résultats/Régression_logistique/matrice_de_confusion_{description}_RL.png', bbox_inches='tight', dpi=300)
    plt.show()

########################## Fonction pour tracer les courbes d'apprentissage ################################
def plot_learning_curve(X_train, y_train, X_test, y_test, description):
    # Définir les valeurs de C à tester
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 200, 1000]
    # Créer des listes pour stocker les scores pour les deux régularisations
    l2_train=[]
    l2_test=[]

    for C in C_values:
        # Créer un modèle de régression logistique avec régularisation L2
        model_l2 = LogisticRegression(penalty='l2', C=C, solver='sag', max_iter=1000)
        l2= model_l2.fit(X_train, y_train)

        # Calculer la précision sur l'ensemble d'entraînement
        l2_train.append(f1_score(y_train, model_l2.predict(X_train), average='weighted'))
        # Calculer la précision sur l'ensemble de test
        l2_test.append(f1_score(y_test, model_l2.predict(X_test), average='weighted'))


    # Tracer les courbes d'apprentissage
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, l2_train, label='Training F1 Score', marker='o')
    plt.plot(C_values, l2_test, label='Testing F1 Score', marker='s')
    plt.xscale('log')
    plt.xlabel('Value of C')
    plt.ylabel('F1 Score')
    plt.title('Learning Curve')
    plt.savefig(f'./résultats/Régression_logistique/courbe_d_apprentissage_{description}_RL.png', bbox_inches='tight', dpi=300)
    plt.legend(loc=2)
    plt.show()

########################## Chargement des données pour chaque langue #######################################
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

########################## Chargement des données combinées ###############################################
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


########################### Appel des fonctions ############################################################
# Pour chaque langue
# for lang in ['en', 'fr', 'it']:
#     X_train, X_test, y_train, y_test = load_data(lang)
#     process_and_evaluate(X_train, X_test, y_train, y_test, lang)

# Traiter l'ensemble combiné de toutes les langues
X_train_combined, X_test_combined, y_train_combined, y_test_combined = load_combined_data()
process_and_evaluate(X_train_combined, X_test_combined, y_train_combined, y_test_combined, "ensemble_combiné")
plot_learning_curve(X_train_combined, y_train_combined, X_test_combined, y_test_combined, "ensemble_combiné")