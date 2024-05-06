# Competence 12-13 du référentiel devia
Répo dédié au test du modèle et du dataset puis de son déploiement : \n

## 1. Situation
Deux évènements peuvent se produire : 
1. Nouveau dataset 
2. Nouveau modèle
Ainsi le répertoire compte 3 branches:
- branche dataset (modification concernant le dataset)
- branche model (modification concernant le modèle)
- branche main (production)

### 1.1 Nouveau dataset
Le changement de dataset se fait sur la branche dédiée du même nom. \n
Cela implique une modification de la référence du dataset (s3_datasets.json) \n
Lors de la PR sur la branche main , le workflow va executer : 
- entrainement du modèle de prod
- évaluation du modèle de prod entrainé (bleu score)
- si les scores sont supérieurs à ceux attendus : sauvegarde sur une plateforme de stockage de celui-ci  et enregistrement des scores
- Push de l'image contenant le modèle et le tokenizer vers un repo de conteneur
### 1.2 Nouveau modèle
Un nouveau modèle doit avoir de meilleures performances que celui précédent. \
Pour changer le modèle : modifier la référence de celui-ci dans le fichier s3_model.json
Ainsi lors de la PR de la branche model vers la main : 
- évaluation du modèle 
- si il est meilleur : sauvegarde du modèle en tant que production
- Push de l'image contenant le modèle et le tokenizer vers un repo de conteneur

## 2. Déploiement
2 volumes 'green' et 'blue' au sein du cluster. \n
Sur le volume qui ne sert pas en production : \n
Lancer le container pour copier le contenu du modèle dans celui du volume. \n
``` docker run --volume xxxxxxxxxxxxxxxxxx```


Puis sur le conteneur de l'api exposant le modèle effectué un déploiement bleu/vert en modifiant le montage du volume tokenizer et model à celui de 'blue' ou 'green'.

## 3.Défauts
- Sur un nouveau dataset : entrainement obligatoire.
- Actuellement les scores ne sont pas mis à jours ( il faudrait pour voir effectuer un commit au sein du workflow tout en maintenant le gitignore ou alors toujours avec le principe de upload / download le fichier de scoring ?)
