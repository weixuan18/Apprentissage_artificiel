# Apprentissage-artificiel
Projet pour *Apprentissage artificiel*

### Contribueurs du groupe
> CHENG Weixuan, GUI Kexin, Huang Yidi


## Objectif  
Proposer un (des) classifieur(s) pour cette tâche, étudier ses (leurs) performances sur cette tâche. Comparer aux informations données dans les actes.

## Comment ça marche ?

1. Extraction : 
    - `Scripts/extraction.py`
    - Données : `Corpus/`
        - Test : `Corpus/test`
        - Train : `Corpus/apprend`
2. Prétraitement des données : 
    - Outils : **spaCy** 
    - Script : `Scripts/Prétraitement.py`
    - Données : `corpus_prétraité/`
3. Entraînement et évaluation des modèles (classifieur)
    1. **Modèle 1** : Régression logistique
        - Script : `Scripts/classifieur_RL`
        - Résultat d'évaluation (image) : `résultats/Régression_logistique/martice_de_confusion_*` <br>
    Résultat d'évaluation (Métriques) : `résultats/Régression_logistique/rapport_classification_*`
    2. **Modèle 2** : Random forest
    3. **Modèle 3** : SVM
