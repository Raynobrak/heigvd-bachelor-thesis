# Travail de Bachelor

Pilotage de drones basé sur de l'apprentissage par renforcement

## Exécution de simplified-drone-sim

### Installation de BreezySLAM

1. disposer du plugin "shell" de poetry : `poetry self add poetry-plugin-shell`
2. ouvrir un shell poetry : `poetry shell`
3. installation : `pip install -vvv --no-binary :all: ./simplified-drone-sim/libraries/BreezySLAM/python`

### Installation de PyRoboViz

Pareil que pour breezyslam mais il faut télécharger le repo de PyRoboViz par le même auteur.

### Mise en place et lancement du programme

1. Télécharger (ou cloner) le code source
2. Installer Poetry (https://python-poetry.org/docs/#installing-with-the-official-installer)
3. Installer les dépendances : `poetry install --no-root`
4. S'assurer que la création de venv est activée : `poetry config virtualenvs.create true --local`
5. `poetry shell`
6. Exécuter le programme (depuis le dossier contenant le .toml) : `poetry run python simplified-drone-sim/main.py`

### Programme

- Déplacements : W,A,S,D -> accélère le drone en haut, à gauche, en bas ou à droite
- Touche P : active/désactive l'estimation de la position avec l'accéléromètre
- Touche Espace : passe en vue drone ou vue normale. En vue drone, rien n'est visible à part les données des capteurs.