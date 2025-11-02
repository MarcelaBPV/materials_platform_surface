# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# ============= Import robusto do Spectrum (ramanchada2 muda entre versões) =============
Spectrum = None
try:
    from ramanchada2.spectrum import Spectrum  # versões novas
except Exception:
    try:
        import ramanchada2 as rc2
        Spectrum = getattr(rc2, "Spectrum", None)
    except Exception:
        Spectrum = None

# ========================== CONFIG STREAMLIT ==========================
st.set_page_config(page_title="Plataforma de Caracterização", layout="wide")
st.title("🔬 Plataforma de Caracterização de Superfícies")

# ========================== CONEXÃO SUPABASE ==========================
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

# ========================== ATRIBUIÇÕES RAMAN ==========================
ASSIGNMENTS = pd.DataFrame([
    (711,  "ν(C–C) celulose", "Celulose"),
    (898,  "δ(C–O–H)", "Celulose"),
    (1086, "ν(C–O)", "Celulose"),
    (1095, "ν(C–C) fibras", "Celulose"),
    (1120, "ν(C–O–H)", "Celulose"),
    (1150, "ν(C–O–C) éter", "Celulose"),
    (1379, "Deformação CH₂", "Celulose"),
    (1472, "δ(CH₂–CH₃)", "Celulose"),
    (1601, "ν(C=C) / lignina", "Celulose/Lignina"),
    (743,  "ν(C–O) celulose / ν(C–O,C–C) carboidratos", "Celulose+Sangue"),
    (965,  "δ(C–O–H) carboidratos/glutationa", "Carboidratos"),
    (1001, "νs(C–C) fenilalanina", "Aminoácido"),
    (1252, "Amida III", "Proteínas"),
    (1342, "Deformação CH₂ lipoproteínas", "Lipoproteínas"),
    (1454, "Colágeno/Fosfolipídios", "Colágeno/Fosfolipídios"),
    (1575, "δ(C=C) fenilalanina", "Aminoácido"),
    (1598, "ν(C=C) hemoglobina", "Hemoglobina"),
    (1620, "Fenilalanina/Tirosina", "Aminoácidos"),
    (1655, "Amida I", "Proteínas"),
], columns=["frequency_cm1", "assignment", "component"])

def match_peaks_to_assignments(peak_positions_cm1: np.ndarray, tol_cm1: float = 8.0) -> pd.DataFrame:
    rows = []
    for p in peak_positions_cm1:
        diffs = np.abs(ASSIGNMENTS["frequency_cm1"].values - p)
        idx = np.argmin(diffs)
        if diffs[idx] <= tol_cm1:
            rows.append({
                "peak_cm1": round(float(p), 2),
                "Δ(cm⁻¹)": round(float(diffs[idx]), 2),
                "ref_cm1": int(ASSIGNMENTS.iloc[idx]["frequency_cm1"]),
                "atribuição": ASSIGNMENTS.iloc[idx]["assignment"],
                "componente": ASSIGNMENTS.iloc[idx]["component"],
            })
    return pd.DataFrame(rows)

# ========================== HELPERS DB/IO ==========================
@st.cache_data(ttl=300)
def load_samples_df() -> pd.DataFrame:
    res = supabase.table("samples").select("id, sample_name, created_at").order("id").execute()
    return pd.DataFrame(res.data) if res.data else pd.DataFrame(columns=["id","sample_name","created_at"])

def get_sample_by_name(sample_name: str):
    res = supabase.table("samples").select("id").eq("sample_name", sample_name).execute()
    return res.data[0]["id"] if res.data else None

def insert_sample(sample_name: str, description: str | None = None):
    sid = get_sample_by_name(sample_name)
    if sid:
        return sid
    res = supabase.table("samples").insert({"sample_name": sample_name, "description": description}).execute()
    return res.data[0]["id"]

def create_measurement(sample_id: int, exp_type: str) -> int:
    res = supabase.table("measurements").insert({"sample_id": sample_id, "type": exp_type}).execute()
    return res.data[0]["id"]

def get_latest_measurement(sample_id: int, exp_type: str) -> int | None:
    res = supabase.table("measurements").select("id").eq("sample_id", sample_id).eq("type", exp_type)\
        .order("id", desc=True).limit(1).execute()
    return res.data[0]["id"] if res.data else None

def insert_rows(table: str, rows: list[dict]) -> int:
    if not rows: return 0
    supabase.table(table).insert(rows).execute()
    return len(rows)

# ========================== PARSERS ==========================
def parse_raman(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    elif name.endswith(".csv") or name.endswith(".txt"):
        df = pd.read_csv(uploaded, sep=None, engine="python")
    else:
        raise ValueError("Formato de arquivo não suportado. Use .csv, .txt ou .xlsx")

    df.columns = [c.lower().replace("#", "").strip() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        if any(k in c for k in ["wave", "shift", "raman", "x"]):
            rename_map[c] = "wavenumber_cm1"
        elif any(k in c for k in ["inten", "signal", "y"]):
            rename_map[c] = "intensity_a"
    df.rename(columns=rename_map, inplace=True)

    if not {"wavenumber_cm1", "intensity_a"}.issubset(df.columns):
        raise ValueError(f"Colunas não reconhecidas. Detectadas: {list(df.columns)}")

    df = df.dropna(subset=["wavenumber_cm1", "intensity_a"])
    return df[["wavenumber_cm1", "intensity_a"]]

# ========================== ABAS ==========================
tab1, tab2, tab3 = st.tabs(["1️⃣ Cadastro de Amostras", "2️⃣ Visualização de Ensaios", "3️⃣ Otimização (IA)"])

# ------------------ ABA 1: Cadastro + Envio ------------------
with tab1:
    st.header("🧪 Cadastrar amostra e enviar ensaio ao Supabase")

    col1, col2 = st.columns([2, 1])
    with col1:
        sample_name = st.text_input("Nome da amostra (ex: Amostra_1)")
    with col2:
        description = st.text_input("Descrição (opcional)")

    tipo_upload = st.radio("Tipo de ensaio a enviar:", ["Raman", "Ângulo de Contato", "4 Pontas"], horizontal=True)
    uploaded = st.file_uploader("📂 Carregar arquivo (.csv, .txt, .xlsx)", type=["csv","txt","xlsx"])

    if st.button("📤 Cadastrar amostra e enviar ao Supabase"):
        if not sample_name or not uploaded:
            st.warning("Informe o nome da amostra e selecione um arquivo.")
        else:
            try:
                sid = insert_sample(sample_name, description)
                mid = create_measurement(sid, tipo_upload.lower().replace(" ", "_"))
                if tipo_upload == "Raman":
                    df = parse_raman(uploaded)
                    rows = [{"measurement_id": mid, "wavenumber_cm1": float(x), "intensity_a": float(y)}
                            for x, y in zip(df["wavenumber_cm1"], df["intensity_a"])]
                    insert_rows("raman_spectra", rows)
                st.success(f"✅ Ensaio {tipo_upload} salvo! amostra={sample_name} (id={sid}), measurement={mid}")

            except Exception as e:
                st.error(f"Erro ao salvar ensaio: {e}")

# ------------------ ABA 2: Visualização ------------------
with tab2:
    st.header("📊 Visualização de ensaios salvos")
    df_samples = load_samples_df()
    if df_samples.empty:
        st.info("Nenhuma amostra cadastrada ainda.")
        st.stop()

    sample_sel = st.selectbox("Amostra:", df_samples["sample_name"].tolist())
    sid = int(df_samples[df_samples["sample_name"] == sample_sel]["id"].values[0])
    tipo_view = st.radio("Tipo de ensaio:", ["Raman", "Ângulo de Contato", "4 Pontas"], horizontal=True, index=0)

    if tipo_view == "Raman":
        mid = get_latest_measurement(sid, "raman")
        if mid is None:
            st.info("Sem ensaio Raman para esta amostra.")
            st.stop()
        res = supabase.table("raman_spectra").select("wavenumber_cm1,intensity_a").eq("measurement_id", mid).order("wavenumber_cm1").execute()
        if not res.data:
            st.info("Sem pontos Raman.")
            st.stop()
        df = pd.DataFrame(res.data)

        if Spectrum is None:
            st.error("Biblioteca ramanchada2 indisponível.")
            st.stop()

        s = Spectrum(x=df["wavenumber_cm1"].values, y=df["intensity_a"].values)
        for step in ("remove_baseline", "smooth", "normalize"):
            try: s = getattr(s, step)()
            except Exception: pass

        try:
            peaks = s.find_peaks(threshold_rel=0.05)
            peak_pos = np.array(getattr(peaks, "x", []), dtype=float)
            peak_int = np.array(getattr(peaks, "y", []), dtype=float)
        except Exception:
            from scipy.signal import find_peaks
            peak_idx, _ = find_peaks(s.y, prominence=0.03)
            peak_pos = s.x[peak_idx]
            peak_int = s.y[peak_idx]

        st.subheader(f"📈 Raman – {sample_sel} (measurement #{mid})")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(s.x, s.y, color="steelblue", lw=1.2)
        ax.scatter(peak_pos, peak_int, color="red", s=25, label="Picos detectados")
        ax.invert_xaxis()
        ax.set_xlabel("Deslocamento Raman (cm⁻¹)")
        ax.set_ylabel("Intensidade (a.u.)")
        ax.legend()
        st.pyplot(fig)

        df_peaks = pd.DataFrame({
            "Posição (cm⁻¹)": np.round(peak_pos, 1),
            "Intensidade (a.u.)": np.round(peak_int, 3),
        })
        st.write("### 🔎 Picos detectados")
        st.dataframe(df_peaks)

        if len(peak_pos):
            id_table = match_peaks_to_assignments(peak_pos, tol_cm1=8.0)
            st.write("### 🧬 Atribuições moleculares correspondentes")
            st.dataframe(id_table if not id_table.empty else pd.DataFrame(
                columns=["peak_cm1","Δ(cm⁻¹)","ref_cm1","atribuição","componente"]
            ))

# ------------------ ABA 3: Otimização ------------------
with tab3:
    st.header("🤖 Otimização via Machine Learning (Random Forest)")
    st.markdown("Treina um modelo com todos os pontos Raman do banco de dados.")

    try:
        data = supabase.table("raman_spectra").select("wavenumber_cm1,intensity_a").execute()
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

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X_test, y_test, label="Real")
        ax.scatter(X_test, y_pred, label="Previsto", alpha=0.7)
        ax.set_xlabel("Número de onda (cm⁻¹)")
        ax.set_ylabel("Intensidade (a.u.)")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro na otimização: {e}")
