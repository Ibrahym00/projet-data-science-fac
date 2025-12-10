"""
Bank Churn Prediction App
Application Streamlit pour prédire le risque de churn des clients bancaires
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("Bank Churn Prediction")
st.markdown("Prédisez le risque de départ de vos clients bancaires")
st.markdown("---")

# Chargement du modèle
@st.cache_resource
def load_model():
    """Charge le modèle et ses informations"""
    import os
    
    # Déterminer le chemin de base
    if os.path.exists('../models/xgboost_churn_model.pkl'):
        # Développement local
        model_path = '../models/xgboost_churn_model.pkl'
        info_path = '../models/model_info.pkl'
    else:
        # Production (Streamlit Cloud)
        model_path = 'models/xgboost_churn_model.pkl'
        info_path = 'models/model_info.pkl'
    
    model = joblib.load(model_path)
    model_info = joblib.load(info_path)
    return model, model_info

# Charger le modèle
try:
    model, model_info = load_model()
    st.success("Modèle chargé avec succès")
except Exception as e:
    st.error(f"Erreur lors du chargement: {e}")
    st.stop()

# Afficher les performances du modèle
st.markdown("## Performance du Modèle")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", f"{model_info['accuracy']*100:.1f}%")
with col2:
    st.metric("ROC-AUC", f"{model_info['roc_auc']:.3f}")
with col3:
    st.metric("Recall", f"{model_info['recall_class_1']*100:.0f}%")

st.markdown("---")

# ========================================
# SIDEBAR - Formulaire de saisie
# ========================================

st.sidebar.header("Informations Client")

# Credit Score
credit_score = st.sidebar.slider(
    "Credit Score",
    min_value=350,
    max_value=850,
    value=650,
    step=10,
    help="Score de crédit du client (350-850)"
)

# Genre
gender = st.sidebar.radio(
    "Genre",
    options=["Homme", "Femme"],
    help="Genre du client"
)

# Age
age = st.sidebar.slider(
    "Âge",
    min_value=18,
    max_value=100,
    value=40,
    step=1,
    help="Âge du client en années"
)

# Pays
geography = st.sidebar.selectbox(
    "Pays",
    options=["France", "Germany", "Spain"],
    help="Pays de résidence du client"
)

# Années client (Tenure)
tenure = st.sidebar.slider(
    "Années client",
    min_value=0,
    max_value=10,
    value=5,
    step=1,
    help="Nombre d'années en tant que client"
)

# Solde du compte
balance = st.sidebar.number_input(
    "Solde du compte (€)",
    min_value=0.0,
    max_value=300000.0,
    value=50000.0,
    step=1000.0,
    help="Solde actuel du compte"
)

# Nombre de produits
num_products = st.sidebar.selectbox(
    "Nombre de produits",
    options=[1, 2, 3, 4],
    index=1,
    help="Nombre de produits bancaires possédés"
)

# Carte de crédit
has_credit_card = st.sidebar.radio(
    "Carte de crédit",
    options=["Oui", "Non"],
    help="Le client possède-t-il une carte de crédit ?"
)

# Membre actif
is_active_member = st.sidebar.radio(
    "Membre actif",
    options=["Oui", "Non"],
    help="Le client est-il actif ?"
)

# Salaire estimé
estimated_salary = st.sidebar.number_input(
    "Salaire estimé (€)",
    min_value=0.0,
    max_value=200000.0,
    value=50000.0,
    step=1000.0,
    help="Salaire annuel estimé"
)

st.sidebar.markdown("---")

# Bouton de prédiction
predict_button = st.sidebar.button("Prédire le Churn", use_container_width=True, type="primary")

# ========================================
# ONGLETS
# ========================================

tab1, tab2, tab3 = st.tabs(["Prédiction", "Analytics", "À propos"])

# ========================================
# ONGLET 1: PRÉDICTION
# ========================================

with tab1:
    st.markdown("## Prédiction Client")
    st.info("Modèle entraîné sur: France, Germany, Spain uniquement")
    
    if predict_button:
        
        # 1. Encoder les variables
        gender_encoded = 1 if gender == "Femme" else 0
        has_card_encoded = 1 if has_credit_card == "Oui" else 0
        is_active_encoded = 1 if is_active_member == "Oui" else 0
        
        # Encoder Geography (One-Hot)
        geo_france = 1 if geography == "France" else 0
        geo_germany = 1 if geography == "Germany" else 0
        geo_spain = 1 if geography == "Spain" else 0
        
        # 2. Créer le dictionnaire de données
        client_data = {
            'CreditScore': credit_score,
            'Gender': gender_encoded,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'Num Of Products': num_products,
            'Has Credit Card': has_card_encoded,
            'Is Active Member': is_active_encoded,
            'Estimated Salary': estimated_salary,
            'Geography_France': geo_france,
            'Geography_Germany': geo_germany,
            'Geography_Spain': geo_spain
        }
        
        # 3. Convertir en DataFrame
        client_df = pd.DataFrame([client_data])
        
        # 4. Faire la prédiction
        prediction = model.predict(client_df)[0]
        probability = model.predict_proba(client_df)[0, 1]
        
        # 5. Déterminer le niveau de risque
        if probability < 0.3:
            risk_level = "Faible"
            risk_color = "green"
        elif probability < 0.6:
            risk_level = "Moyen"
            risk_color = "orange"
        else:
            risk_level = "Élevé"
            risk_color = "red"
        
        # 6. Afficher les résultats
        st.markdown("---")
        st.markdown("## Résultat de la Prédiction")
        
        # Affichage selon le risque
        if risk_level == "Élevé":
            st.error(f"ALERTE: Risque {risk_level} de churn")
        elif risk_level == "Moyen":
            st.warning(f"Attention: Risque {risk_level} de churn")
        else:
            st.success(f"Risque {risk_level} de churn")
        
        # Afficher la probabilité
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Prédiction",
                "CHURN" if prediction == 1 else "RESTE",
                delta=None
            )
        
        with col2:
            st.metric(
                "Probabilité de départ",
                f"{probability*100:.1f}%",
                delta=None
            )
        
        # Barre de progression
        st.markdown("### Score de risque")
        st.progress(float(probability))
        
        # Gauge de risque
        st.markdown("---")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risque de Churn (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "lightyellow"},
                    {'range': [60, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Recommandations
        st.markdown("---")
        st.markdown("### Recommandations Business")
        
        recommendations = []
        
        if is_active_encoded == 0:
            recommendations.append("• **Client inactif** → Lancer une campagne de réactivation")
        
        if geography == "Germany":
            recommendations.append("• **Marché allemand** → Appliquer l'offre de rétention spéciale Germany")
        
        if age >= 50 and age <= 60:
            recommendations.append("• **Tranche d'âge 50-60 ans** → Proposer le programme seniors")
        
        if num_products >= 3:
            recommendations.append("• **Trop de produits** → Risque d'over-selling, simplifier l'offre")
        
        if balance > 100000:
            recommendations.append("• **Client à forte valeur** → Assigner un conseiller dédié")
        
        if len(recommendations) > 0:
            for rec in recommendations:
                st.markdown(rec)
        else:
            st.info("Aucune recommandation spécifique. Continuer le suivi standard.")
        
        # Profil client
        st.markdown("---")
        st.markdown("### Profil du Client")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Informations personnelles:**
            - Genre: {gender}
            - Âge: {age} ans
            - Pays: {geography}
            - Années client: {tenure} ans
            """)
        
        with col2:
            st.markdown(f"""
            **Informations financières:**
            - Credit Score: {credit_score}
            - Solde: {balance:,.0f} €
            - Salaire: {estimated_salary:,.0f} €
            - Produits: {num_products}
            """)
    
    else:
        st.info("Remplissez le formulaire dans la barre latérale et cliquez sur 'Prédire le Churn'")

# ========================================
# ONGLET 2: ANALYTICS
# ========================================

with tab2:
    st.markdown("## Analyse du Modèle")
    
    # Feature Importance
    st.markdown("### Variables les Plus Importantes")
    
    feature_names = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
        'Num Of Products', 'Has Credit Card', 'Is Active Member',
        'Estimated Salary', 'Geography_France', 'Geography_Germany', 'Geography_Spain'
    ]
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    feature_importance['Feature_FR'] = feature_importance['Feature'].map({
        'CreditScore': 'Score de Crédit',
        'Gender': 'Genre',
        'Age': 'Âge',
        'Tenure': 'Ancienneté',
        'Balance': 'Solde',
        'Num Of Products': 'Nb Produits',
        'Has Credit Card': 'Carte de Crédit',
        'Is Active Member': 'Membre Actif',
        'Estimated Salary': 'Salaire',
        'Geography_France': 'France',
        'Geography_Germany': 'Allemagne',
        'Geography_Spain': 'Espagne'
    })
    
    fig_importance = px.bar(
        feature_importance,
        x='Importance',
        y='Feature_FR',
        orientation='h',
        title="Importance des Variables (XGBoost)",
        labels={'Importance': 'Importance', 'Feature_FR': 'Variable'},
        color='Importance',
        color_continuous_scale='Blues'
    )
    
    fig_importance.update_layout(height=500)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("""
    **Interprétation:**
    - Plus la barre est longue, plus la variable est importante pour prédire le churn
    - Les 3 variables clés sont: Membre Actif (27%), Allemagne (20%), et Âge (11%)
    - Le Score de Crédit et la Carte de Crédit ont peu d'impact (<5%)
    """)
    
    # Statistiques du modèle
    st.markdown("---")
    st.markdown("### Performance Détaillée")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Métriques globales:**
        - Dataset: 10,000 clients
        - Train: 8,000 clients (80%)
        - Test: 2,000 clients (20%)
        - SMOTE appliqué (équilibrage)
        """)
    
    with col2:
        st.markdown("""
        **Classe 1 (Churn):**
        - Precision: 59%
        - Recall: 62%
        - F1-Score: 61%
        """)

# ========================================
# ONGLET 3: À PROPOS
# ========================================

with tab3:
    # Liens GitHub et LinkedIn en haut à droite
    col1, col2, col3 = st.columns([6, 1, 1])
    
   
    st.markdown("## À propos du projet")
    
    st.markdown("""
    ### Objectif
    Cette application prédit le risque de départ (churn) des clients bancaires 
    en utilisant un modèle de Machine Learning XGBoost.
    
    ### Données
    - **Source**: Dataset de 10,000 clients bancaires
    - **Variables**: 12 features (âge, solde, pays, etc.)
    - **Cible**: Churn (0 = reste, 1 = part)
    - **Taux de churn**: 20.37%
    
    ### Modèle
    - **Algorithme**: XGBoost (Gradient Boosting)
    - **Performance**: ROC-AUC de 0.857
    - **Recall**: 62% (détecte 62% des clients qui vont partir)
    
    ### Insights clés
    - Les clients allemands ont 2x plus de risque de churn (32% vs 16%)
    - Les femmes partent plus que les hommes (25% vs 16%)
    - Les 50-60 ans ont un risque très élevé (56%)
    - Les membres inactifs ont 2x plus de risque (27% vs 14%)
    
    ### Limitations
    - Modèle entraîné uniquement sur France, Germany, Spain
    - Ne fonctionne pas pour d'autres pays
    - Basé sur des données historiques (peut nécessiter une mise à jour)
    """)
    

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        Développé par <strong>Ibrahim - Aaron - Momar</strong> | Tous droits réservés
    </div>
    """, unsafe_allow_html=True)