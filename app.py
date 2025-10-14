import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client, Client

# --------------------- Configuração da página ---------------------
st.set_page_config(page_title="🌐 Platform Surface", layout="wide")
st.title("🌐 Platform Surface - Surface Characterization Dashboard")

# --------------------- Conexão Supabase ---------------------
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)

# --------------------- Importar módulos de otimização ---------------------
from modules.raman_processing import process_raman
from modules.resistivity_processing import process_resistivity
from modules.contact_angle_processing import process_contact_angle

# --------------------- Carregamento de amostras ---------------------
@st.cache_data(ttl=300)
def load_samples():
    data = supabase.table("samples").select("*").execute().data
    return pd.DataFrame(data) if data else pd.DataFrame()

df_samples = load_samples()

# --------------------- Abas do App ---------------------
abas = st.tabs(["1 Amostras", "2 Ensaios", "3 Propriedades de Superfície"])

# --------------------- Aba 1: Amostras ---------------------
with abas[0]:
    st.header("Gerenciamento de Amostras")
    st.dataframe(df_samples)
    uploaded_file = st.file_uploader("Importar CSV de amostras", type="csv")
    if uploaded_file:
        df_new = pd.read_csv(uploaded_file)
        st.write("Pré-visualização:")
        st.dataframe(df_new.head())
        if st.button("Cadastrar amostras"):
            for _, row in df_new.iterrows():
                supabase.table("samples").insert(row.to_dict()).execute()
            st.success("Amostras cadastradas!")
            df_samples = load_samples()

# --------------------- Aba 2: Ensaios ---------------------
with abas[1]:
    st.header("Ensaios por Amostra")
    if df_samples.empty:
        st.warning("Cadastre amostras primeiro.")
    else:
        sample_id = st.selectbox("Escolha a amostra", df_samples["id"])
        tipo = st.radio("Tipo de ensaio", ["Raman", "4 Pontas", "Ângulo de Contato"])
        uploaded_file = st.file_uploader(f"Importar arquivo de {tipo}", type=["csv", "txt"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(df.head())
            if tipo == "Raman":
                result = process_raman(df)
                st.pyplot(result["figure"])
                st.write("Picos detectados:", result["peaks"])
            elif tipo == "4 Pontas":
                result = process_resistivity(df)
                st.pyplot(result["figure"])
                st.write(f"R = {result['R']:.4e} Ω")
                st.write(f"ρ = {result['rho']:.4e} Ω·m")
                st.write(f"σ = {result['sigma']:.2e} S/m → {result['classe']}")
            elif tipo == "Ângulo de Contato":
                result = process_contact_angle(df)
                st.pyplot(result["figure"])
                st.write(f"Relação energia superficial (γ/γ₀): {result['gamma_ratio']:.3f}")

# --------------------- Aba 3: Propriedades de Superfície ---------------------
with abas[2]:
    st.header("Propriedades de Superfície")
    st.info("Nesta seção você pode gerar gráficos de comparação de propriedades de superfície entre amostras.")
    if not df_samples.empty:
        sample_choice = st.multiselect("Selecione amostras para comparar", df_samples["id"])
        if sample_choice:
            st.write("Funcionalidade de comparação ainda a implementar.")
