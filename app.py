import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client, Client

# Importa módulos de processamento
from raman_processing import process_raman
from resistivity_processing import process_resistivity
from contact_angle_processing import process_contact_angle

# ---------------------------
# CONFIGURAÇÃO
# ---------------------------
st.set_page_config(page_title="Plataforma de Caracterização", layout="wide")
st.title("🔬 Plataforma de Caracterização de Superfícies")

# ---------------------------
# CONEXÃO SUPABASE
# ---------------------------
@st.cache_resource
def init_connection() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

# Teste de conexão
try:
    res = supabase.table("samples").select("*").limit(3).execute()
    st.sidebar.success(f"✅ Conectado ao Supabase ({len(res.data)} amostras)")
except Exception as e:
    st.sidebar.error(f"Erro ao conectar Supabase: {e}")
    st.stop()

# ---------------------------
# FUNÇÕES AUXILIARES
# ---------------------------
@st.cache_data(ttl=300)
def load_samples():
    data = supabase.table("samples").select("id, sample_name, description, category_id, created_at").execute()
    if data.data is None:
        return pd.DataFrame(columns=["id", "sample_name", "description", "category_id", "created_at"])
    return pd.DataFrame(data.data)

def get_measurement_id(sample_id, exp_type):
    meas = supabase.table("measurements").select("id").eq("sample_id", sample_id).eq("type", exp_type).limit(1).execute()
    if meas.data:
        return meas.data[0]["id"]
    new_meas = supabase.table("measurements").insert({"sample_id": sample_id, "type": exp_type}).execute()
    return new_meas.data[0]["id"]

def insert_rows(table, rows):
    if not rows:
        return 0
    try:
        supabase.table(table).insert(rows).execute()
        return len(rows)
    except Exception as e:
        st.error(f"Erro ao inserir em {table}: {e}")
        return 0

# ---------------------------
# INTERFACE
# ---------------------------
tab1, tab2 = st.tabs(["1️⃣ Amostras", "2️⃣ Ensaios"])

# --- Aba 1 ---
with tab1:
    st.header("📋 Gestão de Amostras")
    df_samples = load_samples()

    st.subheader("Upload de arquivo CSV")
    uploaded = st.file_uploader("Selecione CSV com colunas 'sample_name' e 'description'", type="csv")
    if uploaded:
        df_new = pd.read_csv(uploaded)
        st.dataframe(df_new.head())
        if st.button("Cadastrar amostras"):
            inserted = 0
            for _, row in df_new.iterrows():
                payload = {"sample_name": row["sample_name"]}
                if "description" in row and not pd.isna(row["description"]):
                    payload["description"] = str(row["description"])
                try:
                    supabase.table("samples").insert(payload).execute()
                    inserted += 1
                except Exception as e:
                    st.error(f"Erro ao inserir amostra: {e}")
            st.success(f"{inserted} amostras cadastradas.")
            df_samples = load_samples()

    if df_samples.empty:
        st.info("Nenhuma amostra encontrada.")
    else:
        st.dataframe(df_samples)

# --- Aba 2 ---
with tab2:
    st.header("🧪 Importar e Processar Ensaios")

    sample_name = st.text_input("Nome da amostra:")
    if not sample_name:
        st.info("Digite o nome da amostra para continuar.")
        st.stop()

    tipo = st.radio("Tipo de experimento", ["Raman", "4 Pontas", "Ângulo de Contato"])
    uploaded_file = st.file_uploader("Carregar arquivo", type=["csv", "xls", "xlsx", "txt", "log"])

    if uploaded_file:
        existing = supabase.table("samples").select("*").eq("sample_name", sample_name).execute().data
        sample_id = existing[0]["id"] if existing else supabase.table("samples").insert({"sample_name": sample_name}).execute().data[0]["id"]

        try:
            if tipo == "Raman":
                result = process_raman(uploaded_file)
                df = result["df"].dropna()
                mid = get_measurement_id(sample_id, "raman")
                rows = [{"measurement_id": mid, "wavenumber_cm1": float(r["wavenumber_cm1"]), "intensity_a": float(r["intensity_a"])} for _, r in df.iterrows()]
                insert_rows("raman_spectra", rows)
                st.success(f"{len(rows)} pontos Raman importados.")
                st.pyplot(result["figure"])

            elif tipo == "4 Pontas":
                result = process_resistivity(uploaded_file)
                df = result["df"].dropna()
                mid = get_measurement_id(sample_id, "4_pontas")
                rows = [{"measurement_id": mid, "current_a": float(r["current_a"]), "voltage_v": float(r["voltage_v"])} for _, r in df.iterrows()]
                insert_rows("four_point_probe_points", rows)
                st.success(f"{len(rows)} pontos 4 Pontas importados.")
                st.pyplot(result["figure"])

            else:
                result = process_contact_angle(uploaded_file)
                df = result["df"].dropna()
                mid = get_measurement_id(sample_id, "tensiometria")
                rows = [{"measurement_id": mid, "t_seconds": float(r["t_seconds"]), "angle_mean_deg": float(r["angle_mean_deg"])} for _, r in df.iterrows()]
                insert_rows("contact_angle_points", rows)
                st.success(f"{len(rows)} pontos de Ângulo importados.")
                st.pyplot(result["figure"])

        except Exception as e:
            st.error(f"❌ Erro: {e}")
