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
# Tentativa robusta de import do Spectrum (mudou de lugar em versões da ramanchada2)
# -------------------------------------------------------
Spectrum = None
try:
    from ramanchada2.spectrum import Spectrum  # versões mais novas
except Exception:
    try:
        import ramanchada2 as rc2              # fallback
        Spectrum = getattr(rc2, "Spectrum", None)
    except Exception:
        Spectrum = None

# -------------------------------------------------------
# CONFIGURAÇÕES STREAMLIT
# -------------------------------------------------------
st.set_page_config(page_title="Plataforma de Caracterização", layout="wide")
st.title("🔬 Plataforma de Caracterização de Superfícies")

if Spectrum is None:
    st.warning(
        "Não foi possível carregar `Spectrum` da biblioteca `ramanchada2`. "
        "Verifique a dependência no requirements.txt (ex.: ramanchada2>=0.3.0)."
    )

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
# FUNÇÕES AUXILIARES (DB + IO)
# -------------------------------------------------------
@st.cache_data(ttl=300)
def load_samples_df() -> pd.DataFrame:
    data = supabase.table("samples").select(
        "id, sample_name, description, category_id, created_at"
    ).order("id").execute()
    if not data.data:
        return pd.DataFrame(columns=["id", "sample_name", "description", "category_id", "created_at"])
    return pd.DataFrame(data.data)

def insert_sample(name: str, desc: str | None = None, category_id=None):
    payload = {"sample_name": str(name).strip()}
    if desc and not pd.isna(desc):
        payload["description"] = str(desc)
    if category_id is not None and not pd.isna(category_id):
        try:
            payload["category_id"] = int(category_id)
        except Exception:
            pass
    return supabase.table("samples").insert(payload).execute()

def get_or_create_measurement(sample_id: int, exp_type: str) -> int:
    m = supabase.table("measurements").select("id").eq("sample_id", sample_id).eq("type", exp_type).limit(1).execute()
    if m.data:
        return m.data[0]["id"]
    m2 = supabase.table("measurements").insert({"sample_id": sample_id, "type": exp_type}).execute()
    return m2.data[0]["id"]

def create_new_measurement(sample_id: int, exp_type: str) -> int:
    m = supabase.table("measurements").insert({"sample_id": sample_id, "type": exp_type}).execute()
    return m.data[0]["id"]

def get_latest_measurement_id(sample_id: int, exp_type: str) -> int | None:
    m = supabase.table("measurements").select("id").eq("sample_id", sample_id).eq("type", exp_type)\
        .order("id", desc=True).limit(1).execute()
    if m.data:
        return m.data[0]["id"]
    return None

def insert_rows(table: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    supabase.table(table).insert(rows).execute()
    return len(rows)

def read_any_table(table: str, measurement_id: int) -> pd.DataFrame:
    res = supabase.table(table).select("*").eq("measurement_id", measurement_id).order("id").execute()
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()

# --------- parsing de arquivos ----------
def read_samples_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    elif name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            df = pd.read_csv(uploaded, sep=";")
    elif name.endswith(".txt"):
        try:
            df = pd.read_csv(uploaded, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(uploaded, sep="\t", header=0)
    else:
        raise ValueError("Formato não suportado. Use .csv, .xlsx ou .txt")

    df.columns = [c.lower().strip() for c in df.columns]
    if "sample_name" not in df.columns:
        raise ValueError("O arquivo precisa ter uma coluna 'sample_name'.")
    if "description" not in df.columns:
        df["description"] = None
    if "category_id" not in df.columns:
        df["category_id"] = None
    return df[["sample_name", "description", "category_id"]]

def parse_raman_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    elif name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv(uploaded, sep=None, engine="python")
    df.columns = [c.lower().strip() for c in df.columns]
    if "wavenumber_cm1" not in df.columns and "wavenumber" in df.columns:
        df.rename(columns={"wavenumber": "wavenumber_cm1"}, inplace=True)
    if "intensity_a" not in df.columns and "intensity" in df.columns:
        df.rename(columns={"intensity": "intensity_a"}, inplace=True)
    df = df.dropna(subset=["wavenumber_cm1", "intensity_a"])
    df = df[df["intensity_a"] >= 0]
    return df[["wavenumber_cm1", "intensity_a"]]

def parse_four_point_file(uploaded) -> pd.DataFrame:
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    if "current_a" not in df.columns:
        df.rename(columns={"corrente": "current_a", "i": "current_a"}, inplace=True)
    if "voltage_v" not in df.columns:
        df.rename(columns={"tensao": "voltage_v", "v": "voltage_v"}, inplace=True)
    df = df.dropna(subset=["current_a", "voltage_v"])
    df["resistance_ohm"] = df["voltage_v"] / df["current_a"]
    return df[["current_a", "voltage_v", "resistance_ohm"]]

def parse_contact_angle_file(uploaded) -> pd.DataFrame:
    df = pd.read_csv(uploaded, sep=None, engine="python")
    df.columns = [c.lower().strip() for c in df.columns]
    if "mean" in df.columns and "angle_mean_deg" not in df.columns:
        df.rename(columns={"mean": "angle_mean_deg"}, inplace=True)
    if "time" in df.columns and "t_seconds" not in df.columns:
        df.rename(columns={"time": "t_seconds"}, inplace=True)
    df = df.dropna(subset=["t_seconds", "angle_mean_deg"])
    return df[["t_seconds", "angle_mean_deg"]]

# --------- salvar uploads na aba 1/2 ----------
def save_uploaded_experiment(sample_id: int, tipo: str, uploaded) -> int:
    """
    Cria um novo measurement para a amostra e insere os pontos
    Retorna o measurement_id criado.
    """
    tipo_key = {"Raman": "raman", "4 Pontas": "4_pontas", "Ângulo de Contato": "tensiometria"}[tipo]
    mid = create_new_measurement(sample_id, tipo_key)

    if tipo == "Raman":
        df = parse_raman_file(uploaded)
        rows = [{"measurement_id": mid,
                 "wavenumber_cm1": float(x),
                 "intensity_a": float(y)} for x, y in zip(df["wavenumber_cm1"], df["intensity_a"])]
        insert_rows("raman_spectra", rows)
    elif tipo == "4 Pontas":
        df = parse_four_point_file(uploaded)
        rows = df.copy()
        rows["measurement_id"] = mid
        insert_rows("four_point_probe_points", rows.to_dict(orient="records"))
    else:  # Ângulo
        df = parse_contact_angle_file(uploaded)
        rows = df.copy()
        rows["measurement_id"] = mid
        insert_rows("contact_angle_points", rows.to_dict(orient="records"))

    return mid

# -------------------------------------------------------
# ABAS
# -------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["1️⃣ Amostras", "2️⃣ Ensaios", "3️⃣ Otimização (IA)"])

# =======================================================
# ABA 1 – AMOSTRAS (CADASTRO + ANEXAR DADOS)
# =======================================================
with tab1:
    st.header("📋 Cadastro e Visualização de Amostras")
    df_samples = load_samples_df()

    st.subheader("Cadastrar amostra")
    c1, c2 = st.columns([2, 1])
    with c1:
        new_name = st.text_input("Nome da amostra")
    with c2:
        new_cat = st.text_input("Categoria (opcional)")
    new_desc = st.text_area("Descrição (opcional)")
    if st.button("Cadastrar amostra"):
        if new_name:
            insert_sample(new_name, new_desc, new_cat if new_cat else None)
            st.success(f"Amostra '{new_name}' cadastrada!")
            st.cache_data.clear()
        else:
            st.warning("Informe um nome.")

    st.divider()
    st.subheader("Anexar dados de ensaio à amostra (opcional)")
    if not df_samples.empty:
        sample_options = df_samples["sample_name"].tolist()
        sample_sel = st.selectbox("Amostra para anexar dados:", sample_options)
        amostra_id = int(df_samples[df_samples["sample_name"] == sample_sel]["id"].values[0])
        tipo_up = st.radio("Tipo de ensaio para anexar:", ["Raman", "4 Pontas", "Ângulo de Contato"], horizontal=True)
        up = st.file_uploader("📂 Carregar arquivo do ensaio", type=["txt", "csv", "xls", "xlsx"], key="up1")
        if up and st.button("Salvar dados do ensaio"):
            try:
                mid = save_uploaded_experiment(amostra_id, tipo_up, up)
                st.success(f"Dados salvos! measurement_id={mid}")
            except Exception as e:
                st.error(f"Erro ao salvar dados: {e}")
    else:
        st.info("Cadastre uma amostra para anexar dados.")

    st.divider()
    st.subheader("📋 Amostras existentes")
    if df_samples.empty:
        st.info("Nenhuma amostra encontrada.")
    else:
        st.dataframe(df_samples)

# =======================================================
# ABA 2 – ENSAIOS (SELECIONA AMOSTRA, CARREGA DO BANCO E PLOTA)
# =======================================================
with tab2:
    st.header("🧪 Processamento de Ensaios")
    df_samples = load_samples_df()
    if df_samples.empty:
        st.warning("Cadastre uma amostra na aba 1.")
        st.stop()

    sample_name = st.selectbox("🔖 Escolha a amostra:", df_samples["sample_name"].tolist(), key="sel_plot")
    sample_id = int(df_samples[df_samples["sample_name"] == sample_name]["id"].values[0])
    tipo_plot = st.radio("Tipo de ensaio a visualizar:", ["Raman", "4 Pontas", "Ângulo de Contato"], horizontal=True)

    st.markdown("Opcionalmente, você pode **adicionar/atualizar** dados agora mesmo:")
    up2 = st.file_uploader("📂 Carregar novo arquivo do ensaio (opcional)", type=["txt", "csv", "xls", "xlsx"], key="up2")
    if up2 and st.button("Anexar este arquivo à amostra"):
        try:
            mid_new = save_uploaded_experiment(sample_id, tipo_plot, up2)
            st.success(f"Novo ensaio anexado! measurement_id={mid_new}")
        except Exception as e:
            st.error(f"Erro ao anexar: {e}")

    # ---- Buscar último measurement desse tipo e plotar ----
    tipo_key = {"Raman": "raman", "4 Pontas": "4_pontas", "Ângulo de Contato": "tensiometria"}[tipo_plot]
    mid_last = get_latest_measurement_id(sample_id, tipo_key)
    if mid_last is None:
        st.info("Nenhum dado encontrado para este tipo. Anexe um arquivo acima.")
        st.stop()

    if tipo_plot == "Raman":
        df = read_any_table("raman_spectra", mid_last)
        if df.empty:
            st.info("Sem pontos para este measurement.")
            st.stop()
        df = df.sort_values("wavenumber_cm1")
        st.caption(f"measurement_id={mid_last} | pontos={len(df)}")

        if Spectrum is None:
            st.error("ramanchada2/Spectrum indisponível. Instale/atualize a dependência.")
            st.stop()

        # Construção do espectro e tratamentos
        s = Spectrum(x=df["wavenumber_cm1"].to_numpy(), y=df["intensity_a"].to_numpy())
        s_corr = s.copy()
        try:
            s_corr.remove_baseline(inplace=True)
        except Exception:
            pass
        try:
            s_corr.smooth(inplace=True)
        except Exception:
            pass
        try:
            s_corr.normalize(inplace=True)
        except Exception:
            pass
        try:
            peaks = s_corr.find_peaks(threshold_rel=0.05)
        except Exception:
            peaks = None

        st.subheader("📈 Espectro Raman (tratado)")
        fig, ax = plt.subplots()
        s_corr.plot(ax=ax)
        if peaks is not None:
            # algumas versões têm .plot no objeto retornado; se não, apenas marca manualmente
            try:
                peaks.plot(ax=ax, marker="o")
            except Exception:
                ax.scatter(getattr(peaks, "x", []), getattr(peaks, "y", []), marker="o")
        ax.set_xlabel("Número de onda (cm⁻¹)")
        ax.set_ylabel("Intensidade (a.u.)")
        ax.invert_xaxis()
        st.pyplot(fig)

    elif tipo_plot == "4 Pontas":
        df = read_any_table("four_point_probe_points", mid_last)
        if df.empty:
            st.info("Sem pontos para este measurement.")
            st.stop()
        st.caption(f"measurement_id={mid_last} | pontos={len(df)}")

        fig, ax = plt.subplots()
        ax.scatter(df["current_a"], df["voltage_v"], label="Dados")
        ax.plot(df["current_a"], df["current_a"] * df["resistance_ohm"].mean(), label="Ajuste médio")
        ax.set_xlabel("Corrente (A)")
        ax.set_ylabel("Tensão (V)")
        ax.legend()
        st.pyplot(fig)

    else:  # Ângulo de Contato
        df = read_any_table("contact_angle_points", mid_last)
        if df.empty:
            st.info("Sem pontos para este measurement.")
            st.stop()
        st.caption(f"measurement_id={mid_last} | pontos={len(df)}")

        fig, ax = plt.subplots()
        ax.plot(df["t_seconds"], df["angle_mean_deg"], "o-")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Ângulo (°)")
        st.pyplot(fig)

# =======================================================
# ABA 3 – OTIMIZAÇÃO (IA)
# =======================================================
with tab3:
    st.header("🤖 Otimização via Machine Learning (Random Forest)")
    st.markdown("Treina o modelo usando os pontos de **Raman** do banco (todas as amostras).")

    try:
        data = supabase.table("raman_spectra").select("wavenumber_cm1, intensity_a").execute()
        if not data.data:
            st.info("Nenhum dado Raman disponível.")
            st.stop()
        df = pd.DataFrame(data.data).dropna()
        X = df[["wavenumber_cm1"]].values
        y = df["intensity_a"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success("✅ Modelo treinado!")
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
