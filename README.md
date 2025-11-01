# Sound2note

Application multi-plateformes permettant de convertir un son en notes de musiques.

## Lancer l'application

Téléchargez le fichier correspondant à votre plaleforme dans les [releases](https://github.com/varmule/Sound2note/releases/latest), puis lancez l'application

## Fonctionnalités

### Pour charger le signal sonore

- Possibilité de charger un fichier wav (quelques fichiers fournis)
- Possibilité de s'enregistrer

### Pour traiter le signal

3 paramètres pour la stft:

- frame_size : Nombre de points sur lesquels la fft est calculée. Plus il est élevé, meilleure est la qualité. Il doit être pair
- hop size : Nombre de points sur lesquels les fft se superposent. Il est **recommendé** que ce soit un diviseur de la frame size
- Le nombre de fréquences qui sont conservées. Voir ci dessous

Lorsque le programme "traite" le signal, il effectue ces étapes:

1. Récupère les paramètres de traitement (taille de la fenêtre, taille du saut, nombre de fréquences).
2. Calcule la STFT du signal.
3. Ne garde qu'un certain nombre de fréquences
4. Arrondit ces fréquences pour obtenir des notes de musique.
5. Reconstruit le signal audio à partir des notes de musique.
6. Calcule et affiche le PSNR entre le signal original et le signal reconstruit.

### Lecture sonore

Il est possible de jouer les signaux, que ce soit celui original ou celui reconstruit

### Graphiques

Après avoir traiter le signal, l'application propose 3 graphiques :

- Un graphique comparatif des signaux
- Un graphique comparatif des spectrogrammes (qui montrent l'évolution des fréquences dans le temps)
- Un graphique montrant vers quelles notes ont été arrondies les fréquences

### Sauvegarde

Il est possible de sauvegarder le signal reconstruit dans un fichier wav

## Code source

Le code source est contenu dans le dossier src/sound2note.

compression_stft.py permet surtout de traiter le signal, tandis que app.py intègre les fonctionnalités qui vont avec l'interface graphique.
Les deux fichiers sont commentés, de sorte qu'ils soient (plutôt) compréhensible pour ceux qui ont une base en python

## Compiler l'application

Pour compiler le projet pour votre plateforme, il faut utiliser [briefcase](https://briefcase.readthedocs.io/)

1. Commencez par télécharger le code source (`git clone https://github.com/varmule/Sound2note.git` si vous avez [git](https://git-scm.com/))
2. Créez ensuite un environnement virtuel (`python3 -m venv venv` sur Macos/Linux ou `py3 -m venv venv` sur Windows) et activez le (`source venv/bin/activate` pour mac/linux et `venv\Scripts\activate.bat` pour Windows)
3. Allez dans le dossier contenant le code source (`cd Sound2note`)
4. Téléchargez briefcase dans votre environnement virtuel : `pip install briefcase`
5. Pour compiler l'application, éxécutez ces commandes :

```bash
# Pour que briefcase télécharge les dépendances, etc 
briefcase create

# Pour compiler l'application
briefcase build

# Pour créer un fichier permettant d'installer l'application
briefcase package
```
