# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# -------------------------------------------------------
# Import robusto do Spectrum (compatível com versões novas e antigas)
# -------------------------------------------------------
try:
    from ramanchada2.spectrum import Spectrum
except ImportError:
    import ramanchada2 as rc2
    Spectrum = getattr(rc2, "Spectrum", None)

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
    supabase.table("samples").select("id").limit(1).execute()
    st.sidebar.success("✅ Conectado ao Supabase!")
except Exception as e:
    st.sidebar.error(f"Erro ao conectar Supabase: {e}")
    st.stop()

# -------------------------------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------------------------------
def get_sample_by_name(sample_name: str):
    data = supabase.table("samples").select("id").eq("sample_name", sample_name).execute()
    return data.data[0]["id"] if data.data else None

def insert_sample(sample_name: str, description: str = None):
    """Cria amostra se não existir."""
    existing = get_sample_by_name(sample_name)
    if existing:
        return existing
    payload = {"sample_name": sample_name.strip(), "description": description}
    res = supabase.table("samples").insert(payload).execute()
    return res.data[0]["id"]

def create_measurement(sample_id: int, exp_type: str):
    """Cria um measurement para a amostra."""
    res = supabase.table("measurements").insert({"sample_id": sample_id, "type": exp_type}).execute()
    return res.data[0]["id"]

def insert_raman_points(measurement_id: int, df: pd.DataFrame):
    """Insere pontos Raman tratados no banco."""
    rows = [{"measurement_id": measurement_id, "wavenumber_cm1": float(x), "intensity_a": float(y)}
            for x, y in zip(df["wavenumber_cm1"], df["intensity_a"])]
    supabase.table("raman_spectra").insert(rows).execute()

def load_samples():
    data = supabase.table("samples").select("id, sample_name, created_at").order("id").execute()
    if not data.data:
        return pd.DataFrame(columns=["id", "sample_name", "created_at"])
    return pd.DataFrame(data.data)

def load_raman_data(sample_id: int):
    meas = supabase.table("measurements").select("id").eq("sample_id", sample_id).eq("type", "raman").limit(1).execute()
    if not meas.data:
        return None
    mid = meas.data[0]["id"]
    spectra = supabase.table("raman_spectra").select("wavenumber_cm1, intensity_a").eq("measurement_id", mid).execute()
    if not spectra.data:
        return None
    return pd.DataFrame(spectra.data)

# -------------------------------------------------------
# INTERFACE COM ABAS
# -------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["1️⃣ Cadastro de Amostras", "2️⃣ Visualização de Ensaios", "3️⃣ Otimização (IA)"])

# =======================================================
# ABA 1 - CADASTRO E UPLOAD DO ENSAIO RAMAN
# =======================================================
with tab1:
    st.header("🧪 Cadastrar nova amostra com ensaio Raman")

    col1, col2 = st.columns([2, 1])
    with col1:
        sample_name = st.text_input("Nome da amostra (ex: Amostra_1)")
    with col2:
        desc = st.text_input("Descrição (opcional)")

    uploaded_file = st.file_uploader("📂 Carregar arquivo Raman (.csv, .txt, .xlsx)", type=["csv", "txt", "xlsx"])

    if uploaded_file and sample_name:
        try:
            # Ler arquivo
            name = uploaded_file.name.lower()
            if name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            elif name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, sep=None, engine="python")

            df.columns = [c.lower().strip() for c in df.columns]
            if "wavenumber_cm1" not in df.columns and "wavenumber" in df.columns:
                df.rename(columns={"wavenumber": "wavenumber_cm1"}, inplace=True)
            if "intensity_a" not in df.columns and "intensity" in df.columns:
                df.rename(columns={"intensity": "intensity_a"}, inplace=True)

            df = df.dropna(subset=["wavenumber_cm1", "intensity_a"])

            # Criar amostra e measurement
            sample_id = insert_sample(sample_name, desc)
            meas_id = create_measurement(sample_id, "raman")

            # Processar espectro com ramanchada2
            if Spectrum is None:
                st.error("❌ Biblioteca ramanchada2 não disponível.")
                st.stop()

            s = Spectrum(x=df["wavenumber_cm1"].values, y=df["intensity_a"].values)

            # Aplica os tratamentos (sem .copy() e sem inplace=True)
            try:
                s = s.remove_baseline()
            except Exception:
                pass
            try:
                s = s.smooth()
            except Exception:
                pass
            try:
                s = s.normalize()
            except Exception:
                pass

            # Detectar picos
            try:
                peaks = s.find_peaks(threshold_rel=0.05)
            except Exception:
                peaks = None

            # Plot
            st.subheader("📈 Espectro Raman tratado")
            fig, ax = plt.subplots()
            s.plot(ax=ax)
            if peaks is not None:
                try:
                    peaks.plot(ax=ax, marker="o", color="r")
                except Exception:
                    ax.scatter(peaks.x, peaks.y, color="r")
            ax.set_xlabel("Número de onda (cm⁻¹)")
            ax.set_ylabel("Intensidade (a.u.)")
            ax.invert_xaxis()
            st.pyplot(fig)

            # Inserir no banco
            df_proc = pd.DataFrame({"wavenumber_cm1": s.x, "intensity_a": s.y})
            insert_raman_points(meas_id, df_proc)

            st.success(f"Amostra '{sample_name}' cadastrada com sucesso (ID={sample_id}, measurement={meas_id})!")

        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")

# =======================================================
# ABA 2 - VISUALIZAÇÃO DOS ENSAIOS
# =======================================================
with tab2:
    st.header("📊 Visualização de Ensaios Raman")

    df_samples = load_samples()
    if df_samples.empty:
        st.info("Nenhuma amostra cadastrada ainda.")
        st.stop()

    selected_sample = st.selectbox("Selecione uma amostra:", df_samples["sample_name"].tolist())
    sample_id = int(df_samples[df_samples["sample_name"] == selected_sample]["id"].values[0])

    df_raman = load_raman_data(sample_id)
    if df_raman is None or df_raman.empty:
        st.warning("Nenhum dado Raman encontrado para essa amostra.")
        st.stop()

    if Spectrum is None:
        st.error("Biblioteca ramanchada2 não disponível.")
        st.stop()

    s = Spectrum(x=df_raman["wavenumber_cm1"].values, y=df_raman["intensity_a"].values)

    # Tratamento novamente para visualização
    try:
        s = s.remove_baseline()
    except Exception:
        pass
    try:
        s = s.smooth()
    except Exception:
        pass
    try:
        s = s.normalize()
    except Exception:
        pass
    try:
        peaks = s.find_peaks(threshold_rel=0.05)
    except Exception:
        peaks = None

    st.subheader(f"📈 Espectro Raman da amostra '{selected_sample}'")
    fig, ax = plt.subplots()
    s.plot(ax=ax)
    if peaks is not None:
        try:
            peaks.plot(ax=ax, marker="o", color="r")
        except Exception:
            ax.scatter(peaks.x, peaks.y, color="r")
    ax.set_xlabel("Número de onda (cm⁻¹)")
    ax.set_ylabel("Intensidade (a.u.)")
    ax.invert_xaxis()
    st.pyplot(fig)

# =======================================================
# ABA 3 - OTIMIZAÇÃO (IA)
# =======================================================
with tab3:
    st.header("🤖 Otimização via Machine Learning (Random Forest)")
    st.markdown("Treina modelo com todos os pontos Raman do banco de dados.")

    try:
        data = supabase.table("raman_spectra").select("wavenumber_cm1, intensity_a").execute()
        if not data.data:
            st.warning("Nenhum dado Raman disponível.")
            st.stop()

        df = pd.DataFrame(data.data)
        X, y = df[["wavenumber_cm1"]].values, df["intensity_a"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success("✅ Modelo treinado com sucesso!")
        st.write(f"R² = {r2_score(y_test, y_pred):.3f}")
        st.write(f"MAE = {mean_absolute_error(y_test, y_pred):.3f}")
        st.write(f"RMSE = {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

        fig, ax = plt.subplots()
        ax.scatter(X_test, y_test, label="Real")
        ax.scatter(X_test, y_pred, label="Previsto", alpha=0.7)
        ax.set_xlabel("Número de onda (cm⁻¹)")
        ax.set_ylabel("Intensidade (a.u.)")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro na otimização: {e}")
