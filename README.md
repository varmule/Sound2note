# Sound2note

Application multi-plateformes permettant de convertir un son en notes de musiques.

## Lancer l'application

Téléchargez le fichier correspondant à votre plaleforme dans les [releases](https://github.com/varmule/Sound2note/releases/latest)

## Compiler

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
