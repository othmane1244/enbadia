# Phase 5 — Génération des Métriques et Rapport PFE

## Objectif
Collecter les métriques de performance du pipeline sur 2000+ frames et générer les graphiques pour le rapport PFE.

## Composants créés

### 1. `metrics_collector.py`
- Classe `MetricsCollector` pour enregistrer :
  - Chaque frame envoyée (timestamp, latence)
  - Chaque alerte générée (type, confiance, timestamp)
- Calcul des statistiques :
  - FPS moyen
  - Latence min/max/avg
  - Taux d'alertes par type
  - Confiance moyenne par type

### 2. `generate_report.py`
- Charge les métriques depuis `metrics.json`
- Génère 5 graphiques matplotlib :
  1. **01_latence_timeline.png** — Latence dans le temps avec moyenne
  2. **02_latence_distribution.png** — Histogramme distribution latence
  3. **03_alertes_par_type.png** — Camembert répartition alertes
  4. **04_fps_et_alertes.png** — Barres FPS + Ratio alertes/frames
  5. **05_latence_boxplot.png** — Boîte à moustaches latence
- Génère le rapport HTML interactif : `rapport_pfe.html`

### 3. `run_full_test.py`
- Script orchestrateur pour :
  1. Vérifier que l'API est accessible
  2. Lancer le simulateur sur N frames
  3. Générer les rapports et graphiques
  4. Afficher le résumé des métriques

### 4. Modifications `model/simulator.py`
- Intégration de `MetricsCollector`
- Enregistrement de chaque frame envoyée
- Enregistrement de chaque alerte générée
- Sauvegarde automatique des métriques en JSON

## Mode d'emploi

### Prérequis
```bash
pip install matplotlib numpy httpx
```

### Lancement du test complet

#### Étape 1 : Lancer l'API (terminal 1)
```bash
cd Embedded-IA
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

#### Étape 2 : Lancer le test (terminal 2)
```bash
cd Embedded-IA
python run_full_test.py
```

Le script va :
- ✅ Vérifier que l'API est accessible
- 📹 Lancer le simulateur
- ⏹️ Vous devez arrêter manuellement quand vous avez assez de frames (Ctrl+C)
- 📊 Générer automatiquement les rapports et graphiques

### Durée estimée
- **2000 frames @ 30 fps** ≈ 66 secondes ≈ 1 minute 6 secondes

## Fichiers générés

### Sortie : `reports/`
```
reports/
├── metrics.json              # Données brutes (frames + alertes)
├── rapport_pfe.html         # Rapport interactif
├── 01_latence_timeline.png  # Latence dans le temps
├── 02_latence_distribution.png
├── 03_alertes_par_type.png  # Camembert
├── 04_fps_et_alertes.png    # Performance
└── 05_latence_boxplot.png   # Analyse Q1/Médiane/Q3
```

## Métriques collectées

### Performance
- **FPS moyen** — Nombre de frames traitées par seconde
- **Latence frame → alerte** (ms) :
  - Min, Max, Avg
  - Distribution et quartiles
- **Taux d'alertes** — Alertes / Frames

### Qualité
- **Alertes par type** — Intrusion, Chute, Objet_Abandonné, Attroupement
- **Confiance moyenne** par type d'alerte
- **Nombre total** d'alertes

## Exemple de résumé généré
```
======================================================================
📊 RÉSUMÉ DES MÉTRIQUES
======================================================================
Durée totale:        67.23s
Frames traitées:     2010
Alertes générées:    47
FPS moyen:           29.92
Latence (ms):
  - Min:             18.45
  - Avg:             32.18
  - Max:             156.32
Taux alertes/frames: 2.34%

Alertes par type:
  - INTRUSION      :   32 (confiance avg: 0.852)
  - CHUTE          :   12 (confiance avg: 0.756)
  - ATTROUPEMENT   :    3 (confiance avg: 0.891)
  - OBJET_ABANDONNE:    0 (confiance avg: 0.000)
======================================================================
```

## Visualisation du rapport HTML

Ouvrir `reports/rapport_pfe.html` dans un navigateur pour voir :
- Tableau récapitulatif avec métriques clés
- Graphiques interactifs
- Détail des alertes par type
- Courbes et distributions

## Dépannage

### API non accessible
```
❌ API non accessible sur http://127.0.0.1:8000
```
→ Vérifier que l'API est lancée dans terminal 1

### Pas de métriques générées
```
❌ Fichier de métriques non trouvé
```
→ Vérifier que le simulateur a bien démarré et envoyé des frames
→ Vérifier les logs du simulateur

### Erreur lors de la génération des graphiques
```
❌ Erreur lors de la génération des rapports
```
→ Vérifier que `matplotlib` et `numpy` sont installés
→ Regarder les logs complets pour plus de détails

## Prochaines étapes

1. **Analyser les résultats** du rapport HTML
2. **Calculer les métriques de qualité** (FP, TP, F1-score)
3. **Comparer avec les baselines** existantes
4. **Générer les courbes de training** si nécessaire
5. **Rédiger le rapport PFE** avec les graphiques

---
**Phase 5 — Métriques et Rapports** ✅
