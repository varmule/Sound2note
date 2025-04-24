# Projet-son

Prend un son, le découpe en paquets de n millissecondes, et arrondit ses fréquences aux notes de musique les plus proches afin de le modifier.

## Prérequis
-Python


Les modules :

-numpy

-scipy

-librosa

-soundfile

-matplotlib (fourni avec python)

-argparse (fourni avec python)


Pour installer les modules, simplement éxecuter :
`pip install [nom du module]` ou `pip3 install [nom du module]` selon votre version du python (si vous ne savez pas, essayez les 2)

## Execution
Pour lancer le programme, ouvrez un terminal au dossier qui contient le programme et éxecutez :
`python 3 compression.py [nom du fichier]`.

Si vous voulez utiliser des arguments (comme le nombre de fréquences retenues, le nombre de millisecondes par paquet ou désactiver les graphiques), éxecutez `python 3 compression.py -h` pour voir les paramètres et rajoutez les à la fin de la commende. Par exemple, `python3 compression.py SNCF.wav --ms 100 --freq 5 --graphique`.

Après éxecution, un fichier reconstructed.wav sera créé. Il correspond au fichier son modifié selon vos paramètres.

## En cas d'erreur
Le fichier son doit être au format .wav, au format mono (et non stéréo) et dans le même dossier que le programme python.

## A propos des notes
Les notes écrites dans le terminal sont approximatives : des erreurs d'un demi-ton peuvent apparaitre et il y a très souvent des harmoniques qui se glissent ici et là. Chaque ligne correspond aux nombre de millisecondes indiqué.