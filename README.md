# Sound2note

Application multi-plateformes permettant de convertir un son en notes de musique.

## Lancer l'application

Téléchargez le fichier correspondant à votre plateforme dans les [releases](https://github.com/varmule/Sound2note/releases/latest), puis lancez l'application.

## Fonctionnalités

### Chargement du signal sonore

- Chargement d'un fichier `.wav` (quelques fichiers fournis)
- Possibilité de s'enregistrer directement

### Pour traiter le signal

3 paramètres pour la stft:

- frame_size : Nombre de points sur lesquels la fft est calculée. Plus il est élevé, meilleure est la qualité. Il doit être pair
- hop size : Nombre de points sur lesquels les fft se superposent. Il est **recommandé** que ce soit un diviseur de la frame size
- Le nombre de fréquences qui sont conservées. Voir ci-dessous

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

Après avoir traité le signal, l'application propose 3 graphiques :

- Un graphique comparatif des signaux
- Un graphique comparatif des spectrogrammes (qui montrent l'évolution des fréquences dans le temps)
- Un graphique montrant vers quelles notes ont été arrondies les fréquences

### Sauvegarde

Il est possible de sauvegarder le signal reconstruit dans un fichier `.wav`

## Code source

Le code source est contenu dans le dossier src/sound2note.

- `compression_stft.py` : traitement du signal et autre
- `app.py` : interface graphique et fonctionnalités associées

Les deux fichiers sont commentés afin d’être (plutôt) compréhensibles pour toute personne disposant de bases en Python.

## Si aucun binaire n'est fourni pour votre plateforme

Deux possibilités s'offrent à vous :

- Compiler l'application
- Utiliser `compression_stft.py` en ligne de commande

### Compiler l'application

Pour compiler le projet pour votre plateforme, il faut utiliser [briefcase](https://briefcase.readthedocs.io/)

1. Commencez par télécharger le code source (`git clone https://github.com/varmule/Sound2note.git` si vous avez [git](https://git-scm.com/))
2. Créez ensuite un environnement virtuel (`python3 -m venv venv` sur macOS / Linux ou `py3 -m venv venv` sur Windows) et activez le (`source venv/bin/activate` pour macOS / Linux et `venv\Scripts\activate.bat` pour Windows)
3. Allez dans le dossier contenant le code source (`cd Sound2note`)
4. Téléchargez briefcase dans votre environnement virtuel : `pip install briefcase`
5. Pour compiler l'application, exécutez ces commandes :

```bash
# Téléchargement des dépendances
briefcase create

# Compilation de l'application
briefcase build

# Création du fichier d’installation
briefcase package
```

### Lancer `compression_stft.py` en tant qu'outil en ligne de commande

Allez dans le répertoire contenant `compression_stft.py` puis exécutez

```bash
python compression_stft.py --help
```

Vous obtiendrez la liste des paramètres disponibles.

Exemple de commande :

```bash
python compression_stft.py resources/SNCF.wav --frame_size 2048  --hop_size 512 --nombre_de_freq 3
```
