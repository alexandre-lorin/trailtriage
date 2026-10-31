# Plan TrailTriage

- [x] Phase 1 : CLI, configuration validée, découverte, EXIF, SQLite, reprise, logs.
- [x] Phase 2 : YOLO local, lots bornés, CPU/CUDA, scores et cache des détections.
- [x] Phase 3 : regroupement temporel par caméra/dossier.
- [x] Phase 4 : pHash, comparaison bornée, sélection conservatrice.
- [x] Phase 5 : liens sûrs, aucune écriture dans les originaux.
- [x] Phase 6 : CSV, JSONL, erreurs, résumé et mesures.
- [x] Phase 7 : galerie locale filtrable, labels persistants.

## Décisions
- Python 3.12 isolé : Python système 3.14 ne doit pas imposer une compatibilité aux bibliothèques GPU.
- Détection en extra optionnel : découverte et reprise restent utilisables sans PyTorch.
- Poids YOLO exclusivement locaux : téléchargement initial explicite, aucune connexion nécessaire au scan.
- Un résultat « vide » signifie aucune détection, pas absence garantie d'événement.
- Les faibles détections restent en revue ; les seuils proposés ne doivent pas masquer les animaux.
- Pas de garantie de 500–2 000 images sans calibration sur un échantillon réel.
- Pas de reconnaissance faciale, VLM ni entraînement en V1.

- Déduplication conservatrice : même dossier, même burst, contenu détecté compatible et ancre fixe ; comparaisons bornées à 16 groupes. Les scènes plates restent distinctes.

## État de validation

- 23 tests : configuration, EXIF, reprise, interruption, liens, catégories, pHash,
  comparaison bornée et serveur/labels.
- Ruff et vérification de types Mypy sur le paquet.
- YOLO11n réel sur CPU : 31 images valides + 1 corrompue, reprise sans inférence.
- Test hors ligne : toute connexion sortante est refusée et comptée ; 0 tentative.
- Un exemple public Ultralytics valide personnes + bus ; les dessins synthétiques
  servent au développement, pas à mesurer précision/rappel.

## Simplifications justifiées

- SQLite stocke les détections et clusters comme JSON par image : pas de jointures
  inutiles, les labels sont séparés et persistent lors d'une réanalyse.
- Galerie statique + petit serveur Python standard : évite FastAPI pour trois routes.
  Pagination de 120 images pour garder le navigateur réactif.
- Seuil incertain utilisé pour la bande de score, mais toute détection faible reste
  en revue même sous ce seuil (priorité au rappel).
- Groupes de caméra déduits du dossier parent. Mettre chaque caméra dans son dossier.
- Configurations de score/déduplication modifiées : recalcul sans nouvelle inférence.
- Sorties précédentes déplacées sous `.history/`, jamais effacées automatiquement.
- Aucun VLM, reconnaissance faciale ni entraînement inclus.

## À valider sur le terrain

- [ ] Pilote NVIDIA fonctionnel et version CUDA de PyTorch : cette installation
  utilise PyTorch CPU et `nvidia-smi` ne communique pas avec le pilote.
- [ ] Benchmark GPU et échantillon réel (100–500 images) : précision, rappel,
  animaux non COCO, qualité des timestamps et réglages par caméra.
- [ ] Scan complet de 50 000 photos après validation de l'échantillon.
- [ ] Objectif 500–2 000 images à mesurer, sans promettre une réduction universelle.
- [ ] Calibration assistée, boîtes de détection et raccourcis : améliorations futures.
