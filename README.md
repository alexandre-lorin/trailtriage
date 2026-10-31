# TrailTriage

TrailTriage analyse localement des dossiers de photos de caméra de chasse, détecte
personnes, véhicules et animaux, regroupe les images presque identiques et produit
une sélection plus petite à vérifier. **Les images ne quittent jamais la machine.**
**Les originaux ne sont jamais supprimés ni modifiés.**

Les sept phases du pipeline sont implémentées. Le fonctionnement CPU est testé ;
la validation CUDA et sur un véritable jeu de caméra de chasse reste à faire.
L'objectif 50 000 → 500–2 000 dépend de la scène et des réglages ; il n'est pas garanti.

## Installation Linux / Arch

Python 3.12 recommandé, `uv` ou `pip`, environnement isolé :

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,detection]'
```

Alternative :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[detection]'
```

La découverte seule nécessite uniquement `pip install -e .`.
Pour télécharger explicitement le modèle initial (une seule fois, avant le scan) :

```bash
mkdir -p ~/.local/share/trailtriage/models
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt \
  -o ~/.local/share/trailtriage/models/yolo11s.pt
```

L'installation réalisée sur cette machine inclut le modèle YOLO11s et un lanceur
`~/.local/bin/trailtriage` ; la commande fonctionne sans activation du venv.
Le modèle YOLO11n de test reste dans `models/` (ignoré par Git).
Le scan ne télécharge aucun modèle ni image. Télémetrie et installation automatique
Ultralytics désactivées. Le paquet de détection est un extra optionnel.

## CUDA

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Le choix `auto` utilise CUDA si PyTorch le rend disponible, sinon CPU.
Cette installation de validation utilise PyTorch CPU : quand le pilote NVIDIA est
fonctionnel, installer les roues CUDA adaptées à ce pilote dans `.venv` selon
[le sélecteur officiel PyTorch](https://pytorch.org/get-started/locally/), puis refaire
la vérification. Aucun pilote système n'a été modifié par TrailTriage.
`--device cpu` force CPU ; `--device 0` demande CUDA et échoue clairement si indisponible.
Lot automatique : 8 images GPU, 2 CPU. Un manque de VRAM divise le lot récursivement.

## Premier scan

Commencer sur un échantillon, puis utiliser le même dossier de sortie pour reprendre :

```bash
trailtriage --help
trailtriage scan ~/Pictures/camera --dry-run
trailtriage scan ~/Pictures/camera --output ~/Pictures/camera-review
trailtriage serve ~/Pictures/camera-review
```

Ouvrir `http://127.0.0.1:8765`. Les décisions sont enregistrées en SQLite.
Arrêter le serveur avant de relancer un scan dans le même dossier de sortie.
Les filtres, le tri par score/date et les labels sont disponibles ; les boîtes et
raccourcis clavier ne sont pas inclus. La galerie statique `gallery/index.html`
fonctionne aussi sans serveur, mais sans enregistrement des décisions.

```bash
trailtriage scan photos --output output --metadata-only
trailtriage scan photos --output output --model /chemin/modele.pt --batch-size 4
trailtriage scan photos --output output --no-dedup --device cpu
trailtriage scan photos --output output --config config.example.yaml
trailtriage stats output
trailtriage gallery output
```

Sortie par défaut : `trailtriage-output` dans le répertoire courant.
Un dossier de sortie existant doit être vide ou appartenir à ce même scan.
Configuration YAML stricte : une clé inconnue provoque une erreur explicite.
Un fichier modifié est réanalysé ; les images inchangées reprennent leur cache.
Changer les scores ou la déduplication ne relance pas YOLO. Changer le modèle ou
son seuil de confiance relance l'inférence. Les labels manuels sont conservés.

## Sorties et sécurité

```text
output/
├── index.sqlite
├── review/{humans,vehicles,animals,mixed,uncertain}/
├── boring/empty/
├── duplicates/
├── reports/{results.csv,results.jsonl,summary.json,errors.csv,run.log}
├── gallery/{index.html,thumbnails/}
└── .history/     # précédentes générations conservées
```

Les fichiers de revue sont des liens symboliques vers les originaux ; choisir
`output.mode: copy` pour des copies. Aucun lien physique susceptible de permettre
la modification des originaux n'est créé. Ne pas déplacer les sources sans refaire
le scan. Les dossiers de sortie ne sont jamais rescannés et les liens en entrée
sont ignorés. Les liens suspects dans les chemins de rapports sont refusés.
Aucune commande de suppression n'est proposée.
Les anciennes sorties sont conservées sous `.history/` : en mode copie elles
consomment de l'espace supplémentaire à chaque reconstruction.

Le serveur se lie uniquement à `127.0.0.1`. Les vignettes et les images servies sont
réencodées sans EXIF/GPS. La galerie statique ouvre directement les originaux,
qui conservent leurs EXIF. Les rapports contiennent des chemins et dates locaux :
ne pas publier ces dossiers. Les images et modèles sont ignorés par Git.

## Limites et réglages

**« Empty » signifie aucune détection, pas absence certaine d'événement.** Vérifier
`errors.csv` et un échantillon de `boring/empty` pour évaluer les faux négatifs.
Les petites silhouettes, scènes nocturnes et animaux absents des classes du modèle
peuvent être manqués. On peut fournir un modèle local compatible et modifier
`interest.category_groups`, sans hypothèse COCO dans le reste du pipeline.
Toute détection faible ou inconnue est gardée dans `uncertain`, même sous le seuil
incertain. Le seuil définit la bande de score, pas une autorisation de masquer.

Mettre chaque caméra dans un dossier distinct. Les bursts utilisent EXIF local,
puis mtime, puis le nom pour départager. Les dates EXIF sans fuseau sont interprétées
comme locales. La déduplication compare au plus 16 groupes récents dans le même
burst, à une ancre fixe, avec contenu détecté compatible. Les scènes presque
uniformes restent distinctes. Pour les longs clusters à revoir, conserver aussi
le premier et le dernier. Tous les doublons restent consultables dans `duplicates`.

## Tests et benchmark

```bash
pytest
ruff check .
ruff format --check .
mypy src/trailtriage
python scripts/make_sample.py
python scripts/offline_smoke.py
python scripts/benchmark.py /chemin/echantillon --model models/yolo11n.pt
```

Le jeu synthétique contient des scènes dessinées, des bursts et un fichier corrompu.
Les dessins ne mesurent pas la précision de YOLO. Le test réel a également utilisé
[l'image publique de bus Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg).
Le test hors ligne bloque et compte les connexions sortantes. Le benchmark mesure
les étapes, débit, RAM et pic VRAM (si disponible). Utiliser une sortie neuve pour
mesurer un scan sans cache. Voir `TODO.md` pour les validations terrain restantes.
