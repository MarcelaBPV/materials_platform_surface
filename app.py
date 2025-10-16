import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client, Client

# Módulos de processamento
from raman_processing import process_raman
from resistivity_processing import process_resistivity
from contact_angle_processing import process_contact_angle

# -------------------------------------------------------
# Configuração geral
# -------------------------------------------------------
st.set_page_config(page_title="Plataforma de Caracterização", layout="wide")
st.title("*Plataforma de Caracterização de Superfícies*")

# -------------------------------------------------------
# Conexão com Supabase
# -------------------------------------------------------
if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
    st.error("Configura SUPABASE_URL e SUPABASE_KEY em st.secrets (Streamlit Cloud).")
    st.stop()

supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)

# -------------------------------------------------------
# Funções auxiliares
# -------------------------------------------------------
@st.cache_data(ttl=120)
def load_samples():
    try:
        data = supabase.table("samples").select("id, sample_name, description, category_id, created_at").execute().data
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar amostras do Supabase: {e}")
        return pd.DataFrame()

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

# -------------------------------------------------------
# Layout em abas
# -------------------------------------------------------
tab1, tab2 = st.tabs(["1️⃣ Cadastro de Amostras", "2️⃣ Técnicas de Análise"])

# -------------------------------------------------------
# Aba 1 - Cadastro de Amostras
# -------------------------------------------------------
with tab1:
    st.header("Cadastro de Amostras")
    df_samples = load_samples()

    st.subheader("Importar CSV de amostras")
    uploaded = st.file_uploader("CSV (sample_name, description, category_id)", type="csv")
    if uploaded:
        df_new = pd.read_csv(uploaded)
        st.dataframe(df_new.head())
        if st.button("Cadastrar amostras"):
            inserted = 0
            for _, row in df_new.iterrows():
                payload = {"sample_name": row["sample_name"]}
                if "description" in row and not pd.isna(row["description"]):
                    payload["description"] = str(row["description"])
                if "category_id" in row and not pd.isna(row["category_id"]):
                    payload["category_id"] = int(row["category_id"])
                try:
                    supabase.table("samples").insert(payload).execute()
                    inserted += 1
                except Exception as e:
                    st.error(f"Erro ao inserir amostra: {e}")
            st.success(f"{inserted} amostra(s) cadastrada(s).")
            df_samples = load_samples()

    st.subheader("Amostras existentes")
    if df_samples.empty:
        st.info("Nenhuma amostra encontrada.")
    else:
        st.dataframe(df_samples)

# -------------------------------------------------------
# Aba 2 - Técnicas de Análise (com castração automática)
# -------------------------------------------------------
with tab2:
    st.header("Importar e processar dados experimentais")

    # Campo para nome da amostra
    sample_name = st.text_input("Nome da amostra (cada arquivo corresponde a uma amostra)")

    if not sample_name:
        st.info("Digite o nome da amostra para prosseguir.")
    else:
        tipo = st.radio("Tipo de experimento", ["Raman", "4 Pontas", "Ângulo de Contato"])

        # Upload de arquivo conforme tipo
        if tipo == "Raman":
            uploaded_file = st.file_uploader("Carregar arquivo Raman (.xls ou .csv)", type=["xls", "xlsx", "csv"])
        elif tipo == "4 Pontas":
            uploaded_file = st.file_uploader("Carregar arquivo 4 Pontas (.csv)", type=["csv"])
        else:
            uploaded_file = st.file_uploader("Carregar arquivo Ângulo (.txt ou .log)", type=["txt", "log"])

        if uploaded_file:
            # 1️⃣ Cadastrar amostra automaticamente se não existir
            existing = supabase.table("samples").select("*").eq("sample_name", sample_name).execute().data
            if existing:
                sample_id = existing[0]["id"]
            else:
                resp = supabase.table("samples").insert({"sample_name": sample_name}).execute()
                sample_id = resp.data[0]["id"]

            try:
                # 2️⃣ Processar e castrar dados conforme tipo
                if tipo == "Raman":
                    result = process_raman(uploaded_file)
                    df = result["df"].dropna()  # Remove linhas com NaN
                    df = df[df["intensity_a"] >= 0]  # Remove intensidades negativas

                    mid = get_measurement_id(sample_id, "raman")
                    rows = [
                        {"measurement_id": mid,
                         "wavenumber_cm1": float(r["wavenumber_cm1"]),
                         "intensity_a": float(r["intensity_a"])}
                        for _, r in df.iterrows()
                    ]
                    insert_rows("raman_spectra", rows)
                    st.success(f"{len(rows)} pontos Raman válidos importados.")
                    st.pyplot(result["figure"])
                    st.write("Picos identificados:", result["peaks"])

                elif tipo == "4 Pontas":
                    result = process_resistivity(uploaded_file)
                    df = result["df"].dropna()
                    df = df[(df["current_a"] > 0) & (df["voltage_v"] >= 0)]  # Castração

                    mid = get_measurement_id(sample_id, "4_pontas")
                    rows = [
                        {"measurement_id": mid,
                         "current_a": float(r["current_a"]),
                         "voltage_v": float(r["voltage_v"])}
                        for _, r in df.iterrows()
                    ]
                    insert_rows("four_point_probe_points", rows)
                    st.success(f"{len(rows)} pontos 4 Pontas válidos importados.")
                    st.pyplot(result["figure"])
                    st.write(result)

                else:  # Ângulo de Contato
                    result = process_contact_angle(uploaded_file)
                    df = result["df"].dropna()
                    df = df[(df["angle_mean_deg"] >= 0) & (df["angle_mean_deg"] <= 180)]  # Castração

                    mid = get_measurement_id(sample_id, "tensiometria")
                    rows = [
                        {"measurement_id": mid,
                         "t_seconds": float(r["t_seconds"]),
                         "angle_mean_deg": float(r["angle_mean_deg"])}
                        for _, r in df.iterrows()
                    ]
                    insert_rows("contact_angle_points", rows)
                    st.success(f"{len(rows)} pontos de Ângulo válidos importados.")
                    st.pyplot(result["figure"])
                    st.write(result)

            except Exception as e:
                st.error(f"❌ Erro ao importar ou processar arquivo: {e}")

