import streamlit as st
import pandas as pd
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

# Fonction de chargement des données (cachée)
@st.cache_data
def load_data():
    return pd.read_csv("loan_data_clean.csv")

# Fonction de chargement du modèle (cachée)
@st.cache_resource
def load_model(model_name):
    if model_name == "Logistic Regression":
        return joblib.load("logistic_regression.pkl")
    else:
        return joblib.load("random_forest.pkl")

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

# Titre principal
st.title("🏦 Prédiction d'Approbation de Prêt")
st.markdown("Application de Machine Learning pour évaluer les demandes de prêt")
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🤖 Prédiction", "📈 Performance"])

with tab1:
    st.header("📊 Exploration des données")
    
    # Métriques
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
        avg_income = df['CoapplicantIncome'].mean()
        st.metric("💵 Revenu moyen", f"{avg_income:,.0f} €")
    
    st.markdown("---")
    
    # Section Distributions
    st.subheader("📈 Distributions")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, x='ApplicantIncome',
            title='Distribution des revenus des demandeurs',
            labels={'ApplicantIncome': 'Revenu (€)'},
            color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df, y='LoanAmount',
            title='Distribution du montant des prêts',
            labels={'LoanAmount': 'Montant (€)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Section Analyses
    st.subheader("🔍 Analyses")
    col1, col2 = st.columns(2)
    
    with col1:
        # Convertir 0/1 en texte pour le groupby
        df_temp = df.copy()
        df_temp['Loan_Status_Text'] = df_temp['Loan_Status'].map({1: 'Approved', 0: 'Rejected'})
        
        approval_by_edu = df_temp.groupby('Education')['Loan_Status'].mean() * 100
        fig = px.bar(
            x=approval_by_edu.index.map({1: 'Graduate', 0: 'Not Graduate'}),
            y=approval_by_edu.values,
            title='Taux d\'approbation par niveau d\'éducation',
            labels={'x': 'Éducation', 'y': 'Taux (%)'},
            color=approval_by_edu.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        loan_counts = df['Loan_Status'].value_counts()
        fig = px.pie(
            values=loan_counts.values,
            names=['Approved', 'Rejected'],
            title='Répartition des décisions',
            color_discrete_sequence=['#00CC96', '#EF553B']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Section Corrélations
    st.subheader("🔗 Corrélations")
    corr = df.select_dtypes(include=['number']).corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    fig.update_layout(title='Matrice de corrélation des variables numériques', height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Dataset brut
    with st.expander("📄 Voir le dataset complet"):
        st.dataframe(df, use_container_width=True)

with tab2:
    st.header("Faire une prédiction")
    st.write("Section à compléter")

with tab3:
    st.header("Performance du modèle")
    st.write("Section à compléter")