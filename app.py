# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import ramanchada2 as rc2

# -------------------------------------------------------
# CONFIGURAÇÃO GERAL
# -------------------------------------------------------
st.set_page_config(page_title="Plataforma de Caracterização", layout="wide")
st.title("🔬 Plataforma de Caracterização de Superfícies")

# -------------------------------------------------------
# CONEXÃO SUPABASE
# -------------------------------------------------------
@st.cache_resource
def init_connection() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

try:
    res = supabase.table("samples").select("*").limit(3).execute()
    st.sidebar.success(f"✅ Conectado ao Supabase ({len(res.data)} amostras encontradas)")
except Exception as e:
    st.sidebar.error(f"Erro ao conectar Supabase: {e}")
    st.stop()

# -------------------------------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------------------------------
@st.cache_data(ttl=300)
def load_samples():
    data = supabase.table("samples").select("id, sample_name, description, created_at").execute()
    if not data.data:
        return pd.DataFrame(columns=["id", "sample_name", "description", "created_at"])
    return pd.DataFrame(data.data)

def insert_sample(name, desc=None):
    payload = {"sample_name": name}
    if desc:
        payload["description"] = desc
    supabase.table("samples").insert(payload).execute()

def get_measurement_id(sample_id, exp_type):
    meas = supabase.table("measurements").select("id").eq("sample_id", sample_id).eq("type", exp_type).limit(1).execute()
    if meas.data:
        return meas.data[0]["id"]
    new_meas = supabase.table("measurements").insert({"sample_id": sample_id, "type": exp_type}).execute()
    return new_meas.data[0]["id"]

def insert_rows(table, rows):
    if not rows:
        return 0
    supabase.table(table).insert(rows).execute()
    return len(rows)

# -------------------------------------------------------
# ABA 1 - AMOSTRAS
# -------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["1️⃣ Amostras", "2️⃣ Ensaios Raman", "3️⃣ Otimização (IA)"])

with tab1:
    st.header("📋 Cadastro e Visualização de Amostras")

    df_samples = load_samples()
    st.subheader("Cadastrar nova amostra")
    new_name = st.text_input("Nome da amostra")
    new_desc = st.text_area("Descrição (opcional)")
    if st.button("Cadastrar amostra"):
        if new_name:
            insert_sample(new_name, new_desc)
            st.success(f"Amostra '{new_name}' cadastrada com sucesso!")
            df_samples = load_samples()
        else:
            st.warning("Informe um nome de amostra.")

    st.subheader("Amostras existentes")
    if df_samples.empty:
        st.info("Nenhuma amostra encontrada.")
    else:
        st.dataframe(df_samples)

# -------------------------------------------------------
# ABA 2 - ENSAIOS RAMAN
# -------------------------------------------------------
with tab2:
    st.header("🧪 Processamento Raman com Ramanchada2")

    df_samples = load_samples()
    sample_options = df_samples["sample_name"].tolist()
    selected_sample = st.selectbox("Escolha a amostra", ["-- Selecione --"] + sample_options)

    if selected_sample != "-- Selecione --":
        sample_id = df_samples.loc[df_samples["sample_name"] == selected_sample, "id"].values[0]
        uploaded_file = st.file_uploader("Carregar arquivo Raman (.txt, .csv, .xls, .xlsx)", type=["txt", "csv", "xls", "xlsx"])

        if uploaded_file:
            try:
                # Leitura universal
                filename = uploaded_file.name
                if filename.endswith(".xls") or filename.endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)
                elif filename.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                elif filename.endswith(".txt"):
                    df = pd.read_csv(uploaded_file, sep="\t", comment="#", names=["wavenumber_cm1", "intensity_a"], header=0)
                else:
                    raise ValueError("Formato não suportado. Use .txt, .csv, .xls ou .xlsx")

                # Padronização de colunas
                df.columns = [c.lower().strip() for c in df.columns]
                if "wavenumber_cm1" not in df.columns and "wavenumber" in df.columns:
                    df = df.rename(columns={"wavenumber": "wavenumber_cm1"})
                if "intensity_a" not in df.columns and "intensity" in df.columns:
                    df = df.rename(columns={"intensity": "intensity_a"})

                df = df.dropna(subset=["wavenumber_cm1", "intensity_a"])
                df = df[df["intensity_a"] >= 0]

                # Processamento Ramanchada2
                s = rc2.spectrum.from_array(df["wavenumber_cm1"].values, df["intensity_a"].values)
                s_corr = s.remove_baseline().smooth().normalize()
                peaks = s_corr.find_peaks(threshold_rel=0.05)

                # Mostrar tabelas
                st.subheader("📄 Dados originais")
                st.dataframe(df.head())

                norm_df = pd.DataFrame({"wavenumber_cm1": s_corr.x, "normalized_intensity": s_corr.y})
                st.subheader("📈 Dados normalizados")
                st.dataframe(norm_df.head())

                # Gráfico
                fig, ax = plt.subplots()
                s_corr.plot(ax=ax, label="Normalizado")
                peaks.plot(ax=ax, marker="o", color="r", label="Picos")
                ax.set_xlabel("Número de onda (cm⁻¹)")
                ax.set_ylabel("Intensidade (a.u.)")
                ax.legend()
                ax.invert_xaxis()  # Convenção Raman
                fig.tight_layout()
                st.pyplot(fig)

                # Salvar no Supabase (normalizado)
                mid = get_measurement_id(sample_id, "raman")
                rows = [
                    {"measurement_id": mid, "wavenumber_cm1": float(x), "intensity_a": float(y)}
                    for x, y in zip(s_corr.x, s_corr.y)
                ]
                insert_rows("raman_spectra", rows)
                st.success(f"{len(rows)} pontos Raman normalizados importados com sucesso!")

            except Exception as e:
                st.error(f"Erro ao processar arquivo: {e}")

# -------------------------------------------------------
# ABA 3 - OTIMIZAÇÃO (IA)
# -------------------------------------------------------
with tab3:
    st.header("🤖 Otimização via Machine Learning (Random Forest)")
    st.markdown("Treina um modelo IA para aprender padrões de intensidades e identificar picos característicos.")

    try:
        data = supabase.table("raman_spectra").select("wavenumber_cm1, intensity_a").execute()
        if not data.data:
            st.warning("Nenhum dado Raman disponível.")
        else:
            df = pd.DataFrame(data.data).dropna()
            X = df[["wavenumber_cm1"]].values
            y = df["intensity_a"].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.success("✅ Modelo Random Forest treinado com sucesso!")
            st.write(f"**R²:** {r2:.3f}")
            st.write(f"**MAE:** {mae:.3f}")
            st.write(f"**RMSE:** {rmse:.3f}")

            # Gráfico real vs previsto
            fig, ax = plt.subplots()
            ax.scatter(X_test, y_test, label="Real", s=20)
            ax.scatter(X_test, y_pred, label="Previsto", alpha=0.6)
            ax.set_xlabel("Número de onda (cm⁻¹)")
            ax.set_ylabel("Intensidade (a.u.)")
            ax.legend()
            st.pyplot(fig)

            # Picos e grupos moleculares
            st.subheader("🔬 Identificação de picos e grupos moleculares")
            known_groups = {
                1000: "Fenilalanina (anel aromático)",
                1250: "Amidas (proteínas)",
                1600: "C=C (lipídios)",
                1650: "Amida I (proteína)"
            }

            detected_peaks = df.loc[df["intensity_a"] > np.percentile(df["intensity_a"], 98), "wavenumber_cm1"].round().unique()
            detected_peaks.sort()
            result_table = []
            for p in detected_peaks:
                group = known_groups.get(int(p), "Desconhecido")
                result_table.append({"Pico (cm⁻¹)": p, "Grupo Molecular": group})

            st.dataframe(pd.DataFrame(result_table))

    except Exception as e:
        st.error(f"Erro na otimização: {e}")
