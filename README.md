# Travail de Bachelor

Pilotage autonome de drone basé sur de l'apprentissage par renforcement.

Ce repository contient deux projets :
1. Une simulation 2D servant de "proof of concept" pour se familiariser avec l'apprentissage par renforcement.
2. Une simulation 3D relativement réaliste basée sur le projet `gym-pybullet-drones` de utiasDSL : https://github.com/utiasDSL/gym-pybullet-drones

Les sections ci-dessous indique comment installer les dépendances et lancer ces deux projets.

## Exécution de la simulation simplifiée `simplified-drone-sim`

### Installation de BreezySLAM

1. Cloner le repository https://github.com/simondlevy/BreezySLAM.git dans le dossier libraries/
2. disposer du plugin "shell" de poetry : `poetry self add poetry-plugin-shell`
3. ouvrir un shell poetry : `poetry shell`
4. Installation : `pip install -vvv --no-binary :all: ./simplified-drone-sim/libraries/BreezySLAM/python`

### Installation de PyRoboViz

1. Cloner le repository https://github.com/simondlevy/PyRoboViz.git dans le dossier libraries/
2. commenter la ligne 65 du fichier PyRoboViz/roboviz/__init__.py :
3. Ouvrir un shell poetry : `poetry shell`
4. Installation : `pip install -vvv --no-binary :all: ./simplified-drone-sim/libraries/PyRoboViz`

Note : Voici la ligne à décommenter dans le fichier `__init__.py` :
```
#fig.canvas.set_window_title('SLAM')         <--- ICI
plt.title(title)
```

### Installation des autres dépendances
1. Télécharger (ou cloner) ce repository
2. Installer Poetry (https://python-poetry.org/docs/#installing-with-the-official-installer)
3. Installer les dépendances : `poetry install --no-root`
4. S'assurer que la création de venv est activée : `poetry config virtualenvs.create true --local`
5. `poetry shell`
6. Exécuter un script (depuis le dossier contenant le .toml) : `poetry run python simplified-drone-sim/main.py`

### Lancement du programme

Il y a plusieurs scripts à disposition :
- `main_interactive_sim.py` permet d'ouvrir une simulation interactive
  - W,A,S,D pour accélérer le drone dans une direction
  - Les touches 1,2,3 et 4 permettent de changer le mode d'affichage.
    - 1 est le mode normal
    - 2 est le mode "vision du drone"
    - 3 est le mode normal mais sans les obstacles, on ne voit que les projections des rayons LIDAR
    - 4 est le mode SLAM qui utilise BreezySLAM pour créer une carte. 
- `main_rl.py` permet d'entraîner un modèle. Les paramètres sont modifiables directement dans le script
- `visualize_trained_model.py` permet de visualiser un modèle déjà entraîné. Il prend le chemin menant au .zip du modèle comme unique paramètre.
- `open3d_slam_test.py` est une tentative de faire fonctionner la bibliothèque Open3D SLAM. Il n'y a pas de carte affichée mais la position estimée est représentée par un point rouge.

## Simulation 3D

Pré-requis :
- Python 3.10
- Les build tools Microsoft Visual C++ 14.0 minimum

### Installation de gym-pybullet-drones

1. Cloner le repository https://github.com/utiasDSL/gym-pybullet-drones.git dans le dossier drone-rl-environment/libraries/ 
2. cd dans le dossier `drone-rl-environment` contenant le fichier `pyproject.toml`
3. Exécuter `poetry install`

## Exécution d'un script

Les scripts suivants sont disponibles :
- `drone_actions_demo.py` : lance une démo d'un drone avec des actions pré-définies pour tester et visualiser le vol
- `train_model.py` : permet d'entraîner un modèle dans l'environnement 3D
  - NOTE : L'entraînement d'un modèle prend entre 10-15 millions de "steps" avant d'arriver à un résultat acceptable. Suivant la puissance de votre machine, cela peut prendre plusieurs heures.
- `visualize_trained_model.py` : Permet de visualiser un drone controlé par le modèle (fichier .zip) donné en paramètre
  - NOTE : le paramètre n'a pas besoin de spécifier le chemin d'accès complet, **seulement le nom du fichier** dans le dossier drone-rl-environment/models/
- `evaluate_model.py` : permet d'évaluer un modèle un grand nombre de fois pour obtenir un score de performance robuste ainsi que des statistiques. Les résultats sont sauvegardés dans un fichier .CSV et il est possible de simplement visualiser les graphiques en appellant directement la fonction de visualisation en spécifiant le paramètre.