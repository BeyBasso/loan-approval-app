import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

# Titre
st.title("🏦 Prédiction d'Approbation de Prêt")

# Contenu
st.write("Bienvenue dans notre application !")

# Texte et titres
st.title("Titre principal")
st.header("Titre de section")
st.subheader("Sous-titre")
st.text("Texte simple")
st.markdown("**Markdown** avec *formatage*")
st.caption("Légende en petit")
st.code("print('Hello')", language='python')

