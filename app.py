import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks

# Importa módulos de processamento
from raman_processing import process_raman
from resistivity_processing import process_resistivity
from contact_angle_processing import process_contact_angle

# -----------------------------------
# Configuração geral
# -----------------------------------
st.set_page_config(page_title="Plataforma de Caracterização", layout="wide")
st.title("*Plataforma de Caracterização de Superfícies*")

# -----------------------------------
# Conexão com Supabase
# -----------------------------------
if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
    st.error("Por favor configure SUPABASE_URL e SUPABASE_KEY em st.secrets (Streamlit Cloud Secrets).")
    st.stop()

supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)

# -----------------------------------
# Funções auxiliares para DB
# -----------------------------------
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

# -----------------------------------
# Layout por abas
# -----------------------------------
tab1, tab2, tab3 = st.tabs(["1️⃣ Cadastro de Amostras", "2️⃣ Técnicas de Análise", "3️⃣ Otimização / Raman IA"])

# -----------------------------------
# Aba 1 - Cadastro de Amostras
# -----------------------------------
with tab1:
    st.header("Cadastro de Amostras")
    df_samples = load_samples()
    st.subheader("Importar CSV de amostras")
    uploaded = st.file_uploader("CSV de amostras (sample_name, description, category_id opcional)", type="csv")
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

# -----------------------------------
# Aba 2 - Técnicas
# -----------------------------------
with tab2:
    st.header("Importar e visualizar dados experimentais")
    df_samples = load_samples()
    if df_samples.empty:
        st.warning("Cadastre uma amostra antes.")
    else:
        sample_choice = st.selectbox("Escolha a amostra", df_samples["id"])
        tipo = st.radio("Tipo de experimento", ["Raman", "4 Pontas", "Ângulo de Contato"])
        uploaded_file = st.file_uploader(f"Carregar arquivo para {tipo}", type=["csv", "txt", "log"])

        if uploaded_file:
            try:
                if tipo == "Raman":
                    df = pd.read_csv(uploaded_file)
                    if not {"wavenumber_cm1", "intensity_a"}.issubset(df.columns):
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep="\t", names=["wavenumber_cm1", "intensity_a"])
                    df = df.dropna()
                    # Salva no banco
                    mid = get_measurement_id(sample_choice, "raman")
                    rows = [{"measurement_id": mid, "wavenumber_cm1": float(r["wavenumber_cm1"]), "intensity_a": float(r["intensity_a"])} for _, r in df.iterrows()]
                    insert_rows("raman_spectra", rows)
                    st.success(f"{len(rows)} pontos Raman importados.")
                    result = process_raman(df)
                    st.pyplot(result["figure"])

                elif tipo == "4 Pontas":
                    df = pd.read_csv(uploaded_file).dropna()
                    mid = get_measurement_id(sample_choice, "4_pontas")
                    rows = [{"measurement_id": mid, "current_a": float(r["current_a"]), "voltage_v": float(r["voltage_v"])} for _, r in df.iterrows()]
                    insert_rows("four_point_probe_points", rows)
                    st.success(f"{len(rows)} pontos importados.")
                    result = process_resistivity(df)
                    st.pyplot(result["figure"])
                    st.write(result)

                else:  # Ângulo de Contato
                    df = pd.read_csv(uploaded_file, delim_whitespace=True).dropna()
                    df = df.rename(columns={"Time": "t_seconds", "Mean": "angle_mean_deg"})
                    mid = get_measurement_id(sample_choice, "tensiometria")
                    rows = [{"measurement_id": mid, "t_seconds": float(r["t_seconds"]), "angle_mean_deg": float(r["angle_mean_deg"])} for _, r in df.iterrows()]
                    insert_rows("contact_angle_points", rows)
                    st.success(f"{len(rows)} pontos importados.")
                    result = process_contact_angle(df)
                    st.pyplot(result["figure"])
                    st.write(result)

            except Exception as e:
                st.error(f"Erro ao importar ou processar arquivo: {e}")

# -----------------------------------
# Aba 3 - Otimização / Raman IA
# -----------------------------------
with tab3:
    st.header("Otimização e análise Raman com IA")
    df_samples = load_samples()
    if df_samples.empty:
        st.warning("Cadastre amostras primeiro.")
    else:
        sample_choice = st.selectbox("Selecione amostra", df_samples["id"], key="opt_sample")
        try:
            measurements = supabase.table("measurements").select("*").eq("sample_id", sample_choice).eq("type", "raman").execute().data or []
        except Exception as e:
            st.error(f"Erro: {e}")
            measurements = []

        if not measurements:
            st.warning("Nenhuma medição Raman disponível.")
        else:
            all_data = []
            for m in measurements:
                resp = supabase.table("raman_spectra").select("*").eq("measurement_id", m["id"]).execute()
                if resp and resp.data:
                    df_m = pd.DataFrame(resp.data).sort_values("wavenumber_cm1")
                    all_data.append((m["id"], df_m))

            meas_id_list = [m[0] for m in all_data]
            meas_choice = st.selectbox("Measurement para análise", meas_id_list)
            df = next((d for mid, d in all_data if mid == meas_choice), None)

            if df is not None:
                # Pré-processar com seu módulo
                r = process_raman(df)
                fig = r["figure"]
                st.pyplot(fig)

                # Encontrar picos
                y = r["processed_spectrum"].y if "processed_spectrum" in r else df["intensity_a"].values
                x = r["processed_spectrum"].x if "processed_spectrum" in r else df["wavenumber_cm1"].values
                thresh = np.mean(y)
                peaks, _ = find_peaks(y, height=thresh)
                st.write("Picos detectados:", x[peaks].tolist())
                if len(peaks) > 0:
                    fig2, ax2 = plt.subplots()
                    ax2.plot(x, y)
                    ax2.scatter(x[peaks], y[peaks], color="red")
                    ax2.invert_xaxis()
                    st.pyplot(fig2)

                # Similaridade espectral
                st.subheader("Similaridade espectral (cosine)")
                ref_spectra = []
                for mid, d in all_data:
                    if mid != meas_choice:
                        yr = d["intensity_a"].values
                        yr_i = np.interp(x, d["wavenumber_cm1"].values, yr)
                        ref_spectra.append((mid, yr_i))
                if ref_spectra:
                    scaler = StandardScaler()
                    stack = np.vstack([y] + [v for _, v in ref_spectra])
                    stack_s = scaler.fit_transform(stack.T).T
                    sims = [(mid, float(cosine_similarity(stack_s[0:1], v.reshape(1, -1))[0, 0])) for mid, v in ref_spectra]
                    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
                    st.write("Top similaridades:", sims_sorted[:5])

