# Bank Churn Prediction

Projet de prédiction du risque de départ client dans le secteur bancaire.  
Master 1 ISADS - Université Paris-Saclay

---

## Objectif

Développer un modèle capable d'identifier les clients susceptibles de quitter la banque pour permettre des actions de rétention ciblées.

**Métrique principale :** Maximiser le Recall (détecter un maximum de churners) tout en gardant une Precision acceptable.

---

## Dataset

- Source : Kaggle - Bank Customer Churn
- Taille : 10,000 clients, 13 variables
- Target : Churn (20% déséquilibre)
- Variables : CreditScore, Age, Tenure, Balance, NumOfProducts, Geography, Gender, IsActiveMember, etc.

---

## Résultats - Analyse exploratoire

### Facteurs de risque identifiés

| Variable | Impact | Observation |
|----------|--------|-------------|
| Age | Élevé | Clients 40-60 ans : risque x2 |
| Geography | Très élevé | Allemagne : 32% churn vs 16% France/Espagne |
| IsActiveMember | Très élevé | Inactifs : 27% churn vs 14% actifs |
| NumOfProducts | Élevé | 3-4 produits : 83% churn |
| Gender | Modéré | Femmes : 25% vs Hommes : 16% |
| Balance | Modéré | Soldes élevés corrélés au churn |

**Profil à très haut risque :** Femme allemande 45-55 ans, inactive, avec 3+ produits.

---

## Preprocessing & Feature Engineering

### Features créées

```python
Age_Group              # Segmentation par tranches d'âge
Balance_Salary_Ratio   # Ratio solde/salaire
Engagement_Score       # Score composite (activité + produits + ancienneté)
High_Risk             # Flag profil démographique à risque
Zero_Balance          # Indicateur compte dormant
```

### Pipeline

- One-Hot Encoding pour Geography
- Label Encoding pour Gender
- StandardScaler sur variables numériques
- SMOTE pour gérer le déséquilibre des classes

---

## Modélisation

### Algorithmes testés

```
Logistic Regression (baseline)
Random Forest
Gradient Boosting
XGBoost
LightGBM  ← modèle retenu
```

### Optimisation

- GridSearchCV avec StratifiedKFold (5 folds)
- Threshold tuning via courbe Precision-Recall
- Cross-validation pour validation robuste

---

## Performances

### LightGBM Optimisé (modèle final)

```
ROC-AUC    : 0.8615
Accuracy   : 0.8640
Precision  : 0.5514
Recall     : 0.6731  (détecte 67% des churners)
F1-Score   : 0.6058

Seuil optimal : 0.3847 (vs 0.5 par défaut)
```

**Avec threshold optimization :**
```
Recall     : 0.7250  (+5.2 points)
F1-Score   : 0.6324  (+2.7 points)
```

### Top 5 features importantes

```
1. Age (26.3%)
2. NumOfProducts (18.7%)
3. IsActiveMember (15.2%)
4. Geography_Germany (12.8%)
5. Balance (9.4%)
```

---

## Application Streamlit

Interface web permettant de :
- Scorer en temps réel le risque de churn d'un client
- Obtenir des recommandations personnalisées
- Comparer le profil client à la médiane du portfolio
- Exporter les résultats (CSV/JSON)

### Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Accessible sur `http://localhost:8501`

---

## Structure du projet

```
.
├── 01_exploration.ipynb              # EDA + statistiques
├── 02_preprocessing_modeling.ipynb   # Feature engineering + training
├── app.py                            # Application Streamlit
├── models/
│   ├── lightgbm_churn_final.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── model_metadata.pkl
├── data/
│   └── Churn_Modelling.csv
└── requirements.txt
```

---

## Technologies

**Data Science :** pandas, numpy, scikit-learn, imbalanced-learn, xgboost, lightgbm  
**Visualisation :** matplotlib, seaborn, plotly  
**Deployment :** streamlit, joblib

---


## Équipe

Ibrahim DABRE  : [LinkedIn](https://linkedin.com/in/ibrahim-konate)
Aaron THEVA : [LinkedIn](https://linkedin.com/in/ibrahim-konate)
Momar FALL  : [LinkedIn](https://linkedin.com/in/ibrahim-konate)

---

