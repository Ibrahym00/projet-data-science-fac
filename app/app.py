"""
Streamlit App - Prédiction Churn Bancaire
Modèle: LightGBM | ROC-AUC: 0.86
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Bank Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Variables */
    :root {
        --primary: #1a1a2e;
        --primary-light: #16213e;
        --accent: #0f3460;
        --accent-bright: #533483;
        --success: #00b894;
        --warning: #fdcb6e;
        --danger: #d63031;
        --text-primary: #2d3436;
        --text-secondary: #636e72;
        --bg-light: #f8f9fa;
        --border: #e1e8ed;
    }
    
    /* Reset & Base */
    .main {
        background: #ffffff;
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Header Navigation */
    .main-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        padding: 0;
        margin: -2rem -3rem 3rem -3rem;
        box-shadow: 0 2px 20px rgba(0,0,0,0.1);
    }
    
    .header-content {
        padding: 2rem 3rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .header-left h1 {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-left p {
        color: rgba(255,255,255,0.85);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
        font-weight: 400;
    }
    
    .header-right {
        display: flex;
        gap: 2rem;
        align-items: center;
    }
    
    .metric-badge {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.15);
    }
    
    .metric-badge span {
        color: rgba(255,255,255,0.7);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: block;
        margin-bottom: 0.2rem;
    }
    
    .metric-badge strong {
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
    }
    
    /* Two Column Layout */
    .main-grid {
        display: grid;
        grid-template-columns: 380px 1fr;
        gap: 2rem;
        margin-top: 2rem;
    }
    
    /* Left Panel - Configuration */
    .config-panel {
        background: var(--bg-light);
        border-radius: 12px;
        padding: 2rem;
        height: fit-content;
        position: sticky;
        top: 2rem;
        border: 1px solid var(--border);
    }
    
    .config-panel h2 {
        color: var(--text-primary);
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--border);
    }
    
    .form-section {
        margin-bottom: 2rem;
    }
    
    .form-section h3 {
        color: var(--text-primary);
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0 0 1rem 0;
    }
    
    .form-group {
        margin-bottom: 1.2rem;
    }
    
    .form-group label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        font-weight: 500;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    /* Right Panel - Results */
    .results-panel {
        background: white;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid var(--border);
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .card h3 {
        color: var(--text-primary);
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0 0 1.5rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid var(--bg-light);
    }
    
    /* Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s;
    }
    
    .metric-card:hover {
        border-color: var(--accent);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--primary);
        margin: 0;
    }
    
    .metric-card .label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Alert Boxes */
    .alert {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        border-left: 4px solid;
        background: white;
    }
    
    .alert-success {
        background: #f0fff4;
        border-left-color: var(--success);
        color: #22543d;
    }
    
    .alert-warning {
        background: #fffbeb;
        border-left-color: var(--warning);
        color: #744210;
    }
    
    .alert-danger {
        background: #fff5f5;
        border-left-color: var(--danger);
        color: #742a2a;
    }
    
    .alert strong {
        font-weight: 700;
    }
    
    /* Risk Badge */
    .risk-badge {
        display: inline-block;
        padding: 0.6rem 1.5rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-low {
        background: var(--success);
        color: white;
    }
    
    .risk-medium {
        background: var(--warning);
        color: white;
    }
    
    .risk-high {
        background: var(--danger);
        color: white;
    }
    
    /* Recommendations */
    .recommendation {
        background: white;
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s;
    }
    
    .recommendation:hover {
        border-left-width: 6px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .recommendation h4 {
        color: var(--text-primary);
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    
    .recommendation p {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }
    
    .recommendation.priority-high {
        border-left-color: var(--danger);
    }
    
    .recommendation .priority-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    
    .priority-high-badge {
        background: var(--danger);
        color: white;
    }
    
    .priority-medium-badge {
        background: var(--warning);
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: white;
        border-bottom: 2px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background: transparent;
        color: var(--text-secondary);
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary);
        background: var(--bg-light);
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--primary);
        border-bottom: 3px solid var(--primary);
        background: transparent;
    }
    
    /* Buttons */
    .stButton>button {
        background: var(--primary);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.9rem 2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s;
        width: 100%;
        box-shadow: 0 4px 12px rgba(26,26,46,0.3);
    }
    
    .stButton>button:hover {
        background: var(--primary-light);
        box-shadow: 0 6px 20px rgba(26,26,46,0.4);
        transform: translateY(-2px);
    }
    
    /* Form Elements */
    .stSelectbox, .stSlider, .stNumberInput, .stRadio {
        margin-bottom: 0;
    }
    
    .stSelectbox label, .stNumberInput label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
    }
    
    /* Slider improvements */
    .stSlider > label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    
    .stSlider [data-testid="stTickBar"] div {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    .stSlider [data-testid="stThumbValue"] {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        background: white !important;
        border: 2px solid var(--primary) !important;
        padding: 0.3rem 0.6rem !important;
        border-radius: 4px !important;
    }
    
    /* Progress */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--success) 0%, var(--warning) 50%, var(--danger) 100%);
        height: 8px;
        border-radius: 4px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: var(--primary);
        font-weight: 800;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    
    /* Info Box */
    .info-box {
        background: var(--bg-light);
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid var(--border);
    }
    
    .info-box h4 {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
    }
    
    .info-box ul {
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .info-box li {
        color: var(--text-secondary);
        margin-bottom: 0.8rem;
        line-height: 1.6;
    }
    
    /* Table */
    .dataframe {
        border: 1px solid var(--border) !important;
        border-radius: 8px;
    }
    
    .dataframe th {
        background: var(--bg-light) !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
    }
    
    /* Footer */
    .footer {
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 2px solid var(--border);
        text-align: center;
        color: var(--text-primary);
        font-size: 0.9rem;
    }
    
    .footer strong {
        color: var(--text-primary);
        font-weight: 700;
    }
    
    .footer small {
        color: var(--text-secondary);
        font-size: 0.85rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-light);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
</style>
""", unsafe_allow_html=True)

# ==================== CHARGEMENT MODÈLE ====================
@st.cache_resource
def load_model():
    """Charge le modèle LightGBM et ses composants"""
    
    import os
    
    # Si on est dans app/, remonte d'un niveau
    if os.path.exists('../models/lightgbm_churn_final.pkl'):
        model_path = '../models/'

    elif os.path.exists('models/lightgbm_churn_final.pkl'):
        model_path = 'models/'
    else:
        raise FileNotFoundError("Dossier models/ introuvable")
    
    model = joblib.load(f'{model_path}lightgbm_churn_final.pkl')
    metadata = joblib.load(f'{model_path}model_metadata.pkl')
    scaler = joblib.load(f'{model_path}scaler.pkl')
    
    return model, metadata, scaler
try:
    model, metadata, scaler = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_message = str(e)

# ==================== HEADER ====================
perf = metadata['performance'] if model_loaded else {}
perf_opt = metadata['performance_optimal_threshold'] if model_loaded else {}

st.markdown(f"""
<div class="main-header">
    <div class="header-content">
        <div class="header-left">
            <h1>Prédiction du Churn Bancaire</h1>
            <p>Analyse prédictive - LightGBM</p>
        </div>
        <div class="header-right">
            <div class="metric-badge">
                <span>Modèle</span>
                <strong>LightGBM</strong>
            </div>
            <div class="metric-badge">
                <span>ROC-AUC</span>
                <strong>{perf.get('roc_auc', 0):.3f}</strong>
            </div>
            <div class="metric-badge">
                <span>Accuracy</span>
                <strong>{perf.get('accuracy', 0)*100:.1f}%</strong>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Vérification chargement
if not model_loaded:
    st.markdown(f"""
    <div class="alert alert-danger">
        <strong>Erreur système</strong><br>
        Impossible de charger le modèle prédictif. Veuillez vérifier l'installation.<br>
        Détails : {error_message}
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ==================== LAYOUT PRINCIPAL ====================

# Création des colonnes principales
col_config, col_results = st.columns([380, 1020], gap="large")

# ==================== COLONNE GAUCHE : CONFIGURATION ====================
with col_config:
    st.markdown('<div class="config-panel">', unsafe_allow_html=True)
    
    st.markdown("<h2>Configuration Client</h2>", unsafe_allow_html=True)
    
    # Section Profil
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown("<h3>Profil Client</h3>", unsafe_allow_html=True)
    
    gender = st.selectbox("Genre", ["Homme", "Femme"], key="gender")
    age = st.slider("Âge", 18, 100, 42, key="age")
    geography = st.selectbox("Localisation", ["France", "Germany", "Spain"], key="geo")
    tenure = st.slider("Ancienneté (années)", 0, 10, 5, key="tenure")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section Financière
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown("<h3>Données Financières</h3>", unsafe_allow_html=True)
    
    credit_score = st.slider("Credit Score", 350, 850, 650, 10, key="credit")
    balance = st.number_input("Solde du compte (€)", 0.0, 300000.0, 80000.0, 5000.0, key="balance")
    estimated_salary = st.number_input("Salaire estimé (€)", 0.0, 200000.0, 100000.0, 5000.0, key="salary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section Produits
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown("<h3>Produits & Services</h3>", unsafe_allow_html=True)
    
    num_products = st.selectbox("Nombre de produits", [1, 2, 3, 4], index=1, key="products")
    has_credit_card = st.radio("Carte de crédit", ["Oui", "Non"], horizontal=True, key="card")
    is_active_member = st.radio("Membre actif", ["Oui", "Non"], horizontal=True, key="active")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bouton Analyse
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("ANALYSER LE RISQUE", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== COLONNE DROITE : RÉSULTATS ====================
with col_results:
    
    if predict_button:
        
        # Encodage
        gender_encoded = 1 if gender == "Femme" else 0
        has_card_encoded = 1 if has_credit_card == "Oui" else 0
        is_active_encoded = 1 if is_active_member == "Oui" else 0
        
        geo_france = 1 if geography == "France" else 0
        geo_germany = 1 if geography == "Germany" else 0
        geo_spain = 1 if geography == "Spain" else 0
        
        # Feature Engineering
        balance_salary_ratio = balance / (estimated_salary + 1)
        
        if age < 30:
            age_group_young, age_group_middle, age_group_senior = 1, 0, 0
        elif age < 50:
            age_group_young, age_group_middle, age_group_senior = 0, 1, 0
        else:
            age_group_young, age_group_middle, age_group_senior = 0, 0, 1
        
        zero_balance = 1 if balance == 0 else 0
        is_premium = 1 if balance > 100000 else 0
        high_risk = 1 if (age >= 40 and age <= 60 and gender_encoded == 1 and geo_germany == 1) else 0
        
        geogender_france_female = geo_france * gender_encoded
        geogender_germany_female = geo_germany * gender_encoded
        geogender_spain_female = geo_spain * gender_encoded
        
        if tenure <= 2:
            tenure_short, tenure_medium, tenure_long = 1, 0, 0
        elif tenure <= 5:
            tenure_short, tenure_medium, tenure_long = 0, 1, 0
        else:
            tenure_short, tenure_medium, tenure_long = 0, 0, 1
        
        engagement_score = (is_active_encoded * 2) + has_card_encoded + min(num_products, 2)
        
        # Créer le DataFrame avec les features dans l'ordre exact du modèle
        feature_order = metadata['features']
        
        # Créer le dictionnaire avec TOUTES les features dans le bon ordre
        client_data = {}
        
        # Remplir avec les valeurs numériques brutes (avant normalisation)
        for feature in feature_order:
            if feature == 'CreditScore':
                client_data[feature] = credit_score
            elif feature == 'Age':
                client_data[feature] = age
            elif feature == 'Tenure':
                client_data[feature] = tenure
            elif feature == 'Balance':
                client_data[feature] = balance
            elif feature == 'Num Of Products':
                client_data[feature] = num_products
            elif feature == 'Has Credit Card':
                client_data[feature] = has_card_encoded
            elif feature == 'Is Active Member':
                client_data[feature] = is_active_encoded
            elif feature == 'Estimated Salary':
                client_data[feature] = estimated_salary
            elif feature == 'Gender':
                client_data[feature] = gender_encoded
            elif feature == 'Geography_Germany':
                client_data[feature] = geo_germany
            elif feature == 'Geography_Spain':
                client_data[feature] = geo_spain
            elif feature == 'Balance_Salary_Ratio':
                client_data[feature] = balance_salary_ratio
            elif feature == 'Age_Group_Middle-aged':
                client_data[feature] = age_group_middle
            elif feature == 'Age_Group_Senior':
                client_data[feature] = age_group_senior
            elif feature == 'Age_Group_Young':
                client_data[feature] = age_group_young
            elif feature == 'Zero_Balance':
                client_data[feature] = zero_balance
            elif feature == 'Is_Premium':
                client_data[feature] = is_premium
            elif feature == 'High_Risk':
                client_data[feature] = high_risk
            elif feature == 'GeoGender_France_Female':
                client_data[feature] = geogender_france_female
            elif feature == 'GeoGender_Germany_Female':
                client_data[feature] = geogender_germany_female
            elif feature == 'GeoGender_Spain_Female':
                client_data[feature] = geogender_spain_female
            elif feature == 'Tenure_Group_Medium':
                client_data[feature] = tenure_medium
            elif feature == 'Tenure_Group_Short':
                client_data[feature] = tenure_short
            elif feature == 'Engagement_Score':
                client_data[feature] = engagement_score
            else:
                client_data[feature] = 0
        
        # Créer le DataFrame avec les colonnes dans le bon ordre
        client_df = pd.DataFrame([client_data])[feature_order]
        
        # Normalisation avec le scaler
        client_df_scaled = pd.DataFrame(
            scaler.transform(client_df),
            columns=feature_order
        )
        
        # Prédiction
        probability = model.predict_proba(client_df_scaled)[0, 1]
        optimal_threshold = metadata['optimal_threshold']
        prediction = 1 if probability >= optimal_threshold else 0
        
        # Classification risque
        if probability < 0.3:
            risk_level = "Faible"
            risk_class = "risk-low"
            risk_color = "#00b894"
            alert_class = "alert-success"
        elif probability < 0.6:
            risk_level = "Modéré"
            risk_class = "risk-medium"
            risk_color = "#fdcb6e"
            alert_class = "alert-warning"
        else:
            risk_level = "Élevé"
            risk_class = "risk-high"
            risk_color = "#d63031"
            alert_class = "alert-danger"
        
        # ========== AFFICHAGE RÉSULTATS ==========
        
        # Alert principale
        st.markdown(f"""
        <div class="alert {alert_class}">
            <strong>Résultat de l'Analyse</strong><br>
            Classification : <strong>{'CHURN' if prediction == 1 else 'RÉTENTION'}</strong> • 
            Niveau de risque : <span class="risk-badge {risk_class}">{risk_level}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Métriques clés
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Probabilité de Churn",
                f"{probability*100:.1f}%",
                delta=f"{(probability - optimal_threshold)*100:+.1f}% vs seuil"
            )
        
        with col2:
            confidence = abs(probability - 0.5) * 2
            st.metric(
                "Niveau de Confiance",
                f"{confidence*100:.0f}%",
                delta="Élevé" if confidence > 0.7 else "Modéré"
            )
        
        with col3:
            st.metric(
                "Classification",
                "CHURN" if prediction == 1 else "STABLE",
                delta="À risque" if prediction == 1 else "Fidèle",
                delta_color="inverse"
            )
        
        # Jauge de risque
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Évaluation du Risque</h3>", unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Score de Churn (Seuil optimal: {optimal_threshold:.1%})", 'font': {'size': 18}},
            delta={'reference': optimal_threshold * 100, 'increasing': {'color': risk_color}},
            number={'suffix': "%", 'font': {'size': 48}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2},
                'bar': {'color': risk_color, 'thickness': 0.7},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e1e8ed",
                'steps': [
                    {'range': [0, 30], 'color': "#f0fff4"},
                    {'range': [30, 60], 'color': "#fffbeb"},
                    {'range': [60, 100], 'color': "#fff5f5"}
                ],
                'threshold': {
                    'line': {'color': "#1a1a2e", 'width': 3},
                    'thickness': 0.75,
                    'value': optimal_threshold * 100
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': "Inter", 'size': 14},
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommandations
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Plan d'Action Recommandé</h3>", unsafe_allow_html=True)
        
        recommendations = []
        
        if is_active_encoded == 0:
            recommendations.append({
                'title': 'Réactivation Client Prioritaire',
                'description': 'Client inactif - risque de churn multiplié par 2. Contact personnalisé requis sous 48h.',
                'priority': 'high'
            })
        
        if geography == "Germany":
            recommendations.append({
                'title': 'Marché Allemand à Risque',
                'description': 'Le marché allemand présente un taux de churn 2× supérieur (32%). Mesures de rétention spécifiques.',
                'priority': 'high'
            })
        
        if age >= 40 and age <= 60 and gender_encoded == 1:
            recommendations.append({
                'title': 'Segment Critique Femmes 40-60 ans',
                'description': 'Ce segment affiche un taux de churn de 56%. Programme privilège avec gestionnaire dédié.',
                'priority': 'high'
            })
        
        if num_products >= 3:
            recommendations.append({
                'title': 'Optimisation Portefeuille Produits',
                'description': f'{num_products} produits détenus. Configuration 3-4 produits corrélée au churn.',
                'priority': 'high'
            })
        
        if balance > 100000:
            recommendations.append({
                'title': 'Service Private Banking',
                'description': f'Solde élevé ({balance:,.0f}€). Éligibilité au service Private Banking.',
                'priority': 'medium'
            })
        
        if credit_score < 500:
            recommendations.append({
                'title': 'Accompagnement Credit Score',
                'description': f'Credit score faible ({credit_score}). Plan d\'amélioration avec conseiller financier.',
                'priority': 'medium'
            })
        
        if tenure < 2:
            recommendations.append({
                'title': 'Programme Onboarding',
                'description': f'Client récent ({tenure} an). Phase critique - suivi renforcé 12 premiers mois.',
                'priority': 'medium'
            })
        
        if balance == 0:
            recommendations.append({
                'title': 'Alerte Compte Dormant',
                'description': 'Solde nul - forte probabilité de dormance. Contact immédiat requis.',
                'priority': 'high'
            })
        
        if recommendations:
            for rec in recommendations:
                priority_class = "priority-high" if rec['priority'] == 'high' else ""
                badge_class = "priority-high-badge" if rec['priority'] == 'high' else "priority-medium-badge"
                badge_text = "PRIORITÉ HAUTE" if rec['priority'] == 'high' else "PRIORITÉ MOYENNE"
                
                st.markdown(f"""
                <div class="recommendation {priority_class}">
                    <h4>{rec['title']}</h4>
                    <p>{rec['description']}</p>
                    <span class="priority-badge {badge_class}">{badge_text}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert alert-success">
                <strong>Profil stable</strong><br>
                Aucune action urgente requise. Maintien du suivi standard.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Profil détaillé
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Fiche Client Détaillée</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>Informations Démographiques</h4>
                <ul>
                    <li><strong>Genre</strong> : """ + gender + """</li>
                    <li><strong>Âge</strong> : """ + str(age) + """ ans</li>
                    <li><strong>Localisation</strong> : """ + geography + """</li>
                    <li><strong>Ancienneté</strong> : """ + str(tenure) + """ année(s)</li>
                    <li><strong>Statut</strong> : """ + ('Actif' if is_active_encoded else 'Inactif') + """</li>
                    <li><strong>Profil à risque</strong> : """ + ('Oui' if high_risk else 'Non') + """</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-box">
                <h4>Données Financières</h4>
                <ul>
                    <li><strong>Credit Score</strong> : {credit_score}/850</li>
                    <li><strong>Solde</strong> : {balance:,.0f} €</li>
                    <li><strong>Salaire</strong> : {estimated_salary:,.0f} €</li>
                    <li><strong>Ratio Solde/Salaire</strong> : {balance_salary_ratio:.2f}</li>
                    <li><strong>Produits détenus</strong> : {num_products}</li>
                    <li><strong>Carte bancaire</strong> : {'Oui' if has_card_encoded else 'Non'}</li>
                    <li><strong>Score engagement</strong> : {engagement_score}/5</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Comparaison
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Analyse Comparative</h3>", unsafe_allow_html=True)
        
        comparison_data = pd.DataFrame({
            'Indicateur': ['Âge', 'Solde (k€)', 'Credit Score', 'Ancienneté', 'Nb Produits'],
            'Client': [age, balance/1000, credit_score, tenure, num_products],
            'Médiane Portfolio': [39, 76, 650, 5, 2]
        })
        
        fig_comp = go.Figure()
        
        fig_comp.add_trace(go.Bar(
            name='Client',
            x=comparison_data['Indicateur'],
            y=comparison_data['Client'],
            marker_color='#533483',
            text=comparison_data['Client'].round(1),
            textposition='outside'
        ))
        
        fig_comp.add_trace(go.Bar(
            name='Médiane Portfolio',
            x=comparison_data['Indicateur'],
            y=comparison_data['Médiane Portfolio'],
            marker_color='#1a1a2e',
            text=comparison_data['Médiane Portfolio'].round(1),
            textposition='outside'
        ))
        
        fig_comp.update_layout(
            barmode='group',
            height=400,
            title="Position Client vs Médiane du Portfolio",
            xaxis_title="",
            yaxis_title="Valeur",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12),
            title_font=dict(size=16, color='#1a1a2e')
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Export de l'Analyse</h3>", unsafe_allow_html=True)
        
        export_data = {
            'Date_Analyse': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Genre': gender,
            'Age': age,
            'Pays': geography,
            'Credit_Score': credit_score,
            'Anciennete': tenure,
            'Solde': balance,
            'Salaire': estimated_salary,
            'Nb_Produits': num_products,
            'Carte_Credit': 'Oui' if has_card_encoded else 'Non',
            'Membre_Actif': 'Oui' if is_active_encoded else 'Non',
            'Probabilite_Churn': f"{probability:.4f}",
            'Classification': 'CHURN' if prediction == 1 else 'RETENTION',
            'Niveau_Risque': risk_level,
            'Seuil_Utilise': f"{optimal_threshold:.4f}"
        }
        
        export_df = pd.DataFrame([export_data])
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger (CSV)",
                data=csv,
                file_name=f'analyse_churn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        with col2:
            json_str = export_df.to_json(orient='records', indent=2)
            st.download_button(
                label="Télécharger (JSON)",
                data=json_str,
                file_name=f'analyse_churn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mime='application/json',
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # État initial
        st.markdown("""
        <div class="card">
            <div style="text-align: center; padding: 4rem 2rem;">
                <h3 style="color: #1a1a2e; margin-bottom: 1.5rem; font-size: 2rem;">
                    Analyse Prédictive du Risque de Churn
                </h3>
                <p style="font-size: 1.1rem; color: #636e72; line-height: 1.8; max-width: 700px; margin: 0 auto;">
                    Configurez les paramètres client dans le panneau de gauche,<br>
                    puis lancez l'analyse pour obtenir une évaluation complète du risque.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Métriques du modèle
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Performance du Modèle</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{perf['accuracy']*100:.1f}%</div>
                <div class="label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{perf['roc_auc']:.3f}</div>
                <div class="label">ROC-AUC</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{perf_opt['recall']*100:.1f}%</div>
                <div class="label">Recall</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{perf_opt['f1_score']:.3f}</div>
                <div class="label">F1-Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown(f"""
<div class="footer">
    <strong>Projet Académique</strong> • Master ISADS • Université Paris-Saclay<br>
    Ibrahim DABRE • Aaron THEVA • Momar FALL<br>
</div>
""", unsafe_allow_html=True)