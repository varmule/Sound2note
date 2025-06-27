# Projet-son

Prend un son, le découpe en paquets de n millissecondes, et arrondit ses fréquences aux notes de musique les plus proches afin de le modifier.

## Prérequis

* Python (https://www.python.org/downloads/)
* Les modules :
    * numpy
    * scipy
    * librosa
    * matplotlib
    * argparse


Pour installer les modules, simplement éxecuter :
* Sous mac/linux et parfois windows:
    `pip install numpy scipy librosa matplotlib argparse`  
* Sous windows, si la commande précédente ne fonctionne pas :
    `py -m pip install numpy scipy librosa matplotlib argparse`

## Installation
Cliquez sur code->Download zip pour télecharger le programme python et les fichiers test.
Extrayez ensuite le fichier .zip

## Execution
Pour lancer le programme, ouvrez tout d'abord un terminal au dossier qui contient le programme en allant sur le dossier, puis faites un clic droit sur ledit dossier et cliquer sur 'ouvrir un terminal au dossier' (ou similaire).

*Attention  : sous windows, la commande `python` (qui fonctionne sous linux et mac) est peut-être à remplacer par la commande `py`. Veillez donc à remplacer le début de toutes les commandes.*

Ensuite,  éxecutez :
`python compression.py [nom du fichier]` sous mac/linux (ou `py compression.py [nom du fichier]` si cela ne fonctionne pas sous windows)

Si vous voulez utiliser des arguments (comme le nombre de fréquences retenues, le nombre de millisecondes par paquet ou désactiver les graphiques), éxecutez `python compression.py -h` pour voir les paramètres et rajoutez les à la fin de la commande. Par exemple, `python compression.py SNCF.wav --ms 100 --freq 5 --graphique`.

Après éxecution, un fichier reconstructed.wav sera créé. Il correspond au fichier son modifié selon vos paramètres.

## En cas d'erreur
Le fichier son doit être au format .wav, au format mono (et non stéréo) et dans le même dossier que le programme python.  
Vérifiez bien que le terminal est ouvert dans le dossier comportant le programme python.

## A propos des notes
Les notes écrites dans le terminal sont approximatives : des erreurs d'un demi-ton peuvent apparaitre et il y a très souvent des harmoniques qui se glissent ici et là. Chaque ligne correspond aux nombre de millisecondes indiqué.
