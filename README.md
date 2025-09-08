# Projet-son

Prend un son, le découpe en paquets de n millissecondes, et arrondit ses fréquences aux notes de musique les plus proches afin de le modifier.

## Prérequis

* Python (https://www.python.org/downloads/)
* Git (https://git-scm.com/downloads)

## Installation du programme et des dépendances
Ouvrez un terminal et exécutez 

`git clone https://github.com/varmule/Sound2note.git`

`cd Sound2note`

`pip3 install -r requirements.txt` (Sous windows, si cette commande retourne une erreur, essayez `py3 -m pip install -r requirements.txt`)



## Execution

*Attention  : sous windows, la commande `python3` (qui fonctionne sous linux et mac) est parfois remplacée par la commande `py3`. Veillez donc à remplacer le début de toutes les commandes si  `python3` ne fonctionne pas.*

Après les commandes précédentes,  exécutez : `python3 gui.py`

## Rajouter des sons

Les fichiers doivent être dans le dossier sounds du projet et au format .wav