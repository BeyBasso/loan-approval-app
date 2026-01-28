import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# CHARGEMENT DES RESSOURCES (CHEMINS CORRIGÉS)
# =========================================================================

@st.cache_data
def load_data():
    # Utilisation du nom de fichier direct (doit être dans le même dossier)
    return pd.read_csv("loan_data_clean.csv")

@st.cache_resource
def load_model(model_name):
    # Utilisation de chemins relatifs ./models/ pour la portabilité
    if model_name == "Logistic Regression":
        return joblib.load("./models/logistic_regression.pkl")
    else:
        return joblib.load("./models/random_forest.pkl")
    
@st.cache_resource
def load_scaler():
    try:
        return joblib.load("./models/scaler.pkl")
    except:
        return None

# =========================================================================
# INTERFACE PRINCIPALE
# =========================================================================

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Choisir le modèle",
    ["Logistic Regression", "Random Forest"]
)

# Charger les données et le modèle
df = load_data()
model = load_model(model_choice)
scaler = load_scaler()

# Titre principal
st.title("🏦 Prédiction d'Approbation de Prêt")
st.markdown("Application de Machine Learning pour évaluer les demandes de prêt")
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🤖 Prédiction", "📈 Performance"])

# ---------------------------------------------------------------------
# TAB 1 : EXPLORATION
# ---------------------------------------------------------------------
with tab1:
    st.header("📊 Exploration des données")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total demandes", f"{len(df):,}")
    with col2:
        approval_rate = (df['Loan_Status'] == 1).mean() * 100
        st.metric("✅ Taux d'approbation", f"{approval_rate:.1f}%")
    with col3:
        avg_loan = df['LoanAmount'].mean()
        st.metric("💰 Montant moyen", f"{avg_loan:,.0f} €")
    with col4:
        avg_income = df['ApplicantIncome'].mean()
        st.metric("💵 Revenu moyen", f"{avg_income:,.0f} €")
    
    st.markdown("---")
    
    st.subheader("📈 Distributions")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.histogram(df, x='ApplicantIncome', title='Distribution des revenus', color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.box(df, y='LoanAmount', title='Distribution du montant des prêts')
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2 : PRÉDICTION
# ---------------------------------------------------------------------
with tab2:
    st.header("🤖 Faire une prédiction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Finances")
            gender = st.selectbox("Genre", options=[1, 0], format_func=lambda x: "👨 Homme" if x == 1 else "👩 Femme")
            applicant_income = st.number_input("Revenu mensuel (€)", min_value=0, value=5000)
            coapplicant_income = st.number_input("Revenu co-demandeur (€)", min_value=0, value=0)
            loan_amount = st.number_input("Montant demandé (€)", min_value=1000, value=150000)
            loan_term = st.number_input("Durée (mois)", min_value=12, max_value=480, value=360)
        
        with col2:
            st.subheader("👤 Profil")
            credit_history = st.selectbox("Historique de crédit", options=[1, 0], format_func=lambda x: "✅ Bon" if x == 1 else "❌ Mauvais")
            education = st.selectbox("Éducation", options=[1, 0], format_func=lambda x: "🎓 Graduate" if x == 1 else "📚 Not Graduate")
            married = st.selectbox("Statut marital", options=[1, 0], format_func=lambda x: "💑 Marié(e)" if x == 1 else "🧍 Célibataire")
            dependents = st.number_input("Personnes à charge", min_value=0, max_value=10, value=0)
            self_employed = st.selectbox("Indépendant", options=[0, 1], format_func=lambda x: "✅ Oui" if x == 1 else "❌ Non")
            property_area = st.selectbox("Zone du bien", options=["Urban", "Semiurban", "Rural"])

        submitted = st.form_submit_button("🔮 Prédire l'approbation", use_container_width=True, type="primary")

    if submitted:
        with st.spinner('Analyse en cours...'):
            # Préparation des données (Feature Engineering identique à l'entraînement)
            input_data = {
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_term,
                'Credit_History': credit_history,
                'Education': education,
                'Gender_Male': gender,
                'Married_Yes': married,
                'Dependents': dependents,
                'SelfEmployed_Yes': self_employed,
                'Area_Semiurban': 1 if property_area == "Semiurban" else 0,
                'Area_Urban': 1 if property_area == "Urban" else 0
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Ajout des features calculées
            input_df['TotalIncome'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
            input_df['LoanAmountToIncome'] = input_df['LoanAmount'] / (input_df['TotalIncome'] + 1)
            input_df['EMI'] = input_df['LoanAmount'] / input_df['Loan_Amount_Term']
            input_df['EMIToIncome'] = input_df['EMI'] / (input_df['TotalIncome'] + 1)
            input_df['Log_LoanAmount'] = np.log(input_df['LoanAmount'] + 1)
            input_df['Log_TotalIncome'] = np.log(input_df['TotalIncome'] + 1)
            input_df['Has_Coapplicant'] = (input_df['CoapplicantIncome'] > 0).astype(int)

            # Réorganisation des colonnes selon le modèle
            if hasattr(model, 'feature_names_in_'):
                input_df = input_df[model.feature_names_in_]

            # Normalisation et Prédiction
            if model_choice == "Logistic Regression" and scaler is not None:
                input_final = scaler.transform(input_df)
            else:
                input_final = input_df

            prediction = model.predict(input_final)[0]
            proba = model.predict_proba(input_final)[0]

            # Affichage du résultat
            st.markdown("---")
            if prediction == 1:
                st.success("### ✅ PRÊT APPROUVÉ")
                st.balloons()
            else:
                st.error("### ❌ PRÊT REJETÉ")

            col_p1, col_p2 = st.columns(2)
            col_p1.metric("Probabilité Approbation", f"{proba[1]*100:.1f}%")
            col_p2.metric("Probabilité Rejet", f"{proba[0]*100:.1f}%")

# ---------------------------------------------------------------------
# TAB 3 : PERFORMANCE
# ---------------------------------------------------------------------
with tab3:
    st.header("📈 Performance du modèle")
    st.info("Cette section affiche les métriques validées lors de l'entraînement.")