# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# ---- fitting
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

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
    (743,  "ν(C–O) celulose / ν(C–O,C–C) carboidratos", "Celulose+Sangue"),
    (750,  "Triptofano", "Aminoácido"),
    (898,  "δ(C–O–H)", "Celulose"),
    (965,  "δ(C–O–H) carboidratos/glutationa", "Carboidratos"),
    (1001, "νs(C–C) fenilalanina", "Aminoácido"),
    (1086, "ν(C–O)", "Celulose"),
    (1095, "ν(C–C) fibras", "Celulose"),
    (1120, "ν(C–O–H)", "Celulose"),
    (1150, "ν(C–O–C) éter", "Celulose"),
    (1252, "Amida III", "Proteínas"),
    (1342, "Deformação CH₂ lipoproteínas", "Lipoproteínas"),
    (1379, "Deformação CH₂", "Celulose"),
    (1454, "Colágeno/Fosfolipídios", "Colágeno/Fosfolipídios"),
    (1575, "δ(C=C) fenilalanina", "Aminoácido"),
    (1598, "ν(C=C) hemoglobina", "Hemoglobina"),
    (1601, "ν(C=C) / lignina", "Celulose/Lignina"),
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

def parse_four_point(uploaded) -> pd.DataFrame:
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    if "current_a" not in df.columns:
        df.rename(columns={"corrente": "current_a", "i": "current_a"}, inplace=True)
    if "voltage_v" not in df.columns:
        df.rename(columns={"tensao": "voltage_v", "v": "voltage_v"}, inplace=True)
    df = df.dropna(subset=["current_a", "voltage_v"])
    df["resistance_ohm"] = df["voltage_v"] / df["current_a"]
    return df[["current_a","voltage_v","resistance_ohm"]]

def parse_contact_angle(uploaded) -> pd.DataFrame:
    df = pd.read_csv(uploaded, sep=None, engine="python")
    df.columns = [c.lower().strip() for c in df.columns]
    if "mean" in df.columns and "angle_mean_deg" not in df.columns:
        df.rename(columns={"mean":"angle_mean_deg"}, inplace=True)
    if "time" in df.columns and "t_seconds" not in df.columns:
        df.rename(columns={"time":"t_seconds"}, inplace=True)
    df = df.dropna(subset=["t_seconds","angle_mean_deg"])
    return df[["t_seconds","angle_mean_deg"]]

# ========================== PERSISTERS ==========================
def persist_raman(sample_id: int, uploaded) -> int:
    df = parse_raman(uploaded)
    mid = create_measurement(sample_id, "raman")
    rows = [{"measurement_id": mid, "wavenumber_cm1": float(x), "intensity_a": float(y)}
            for x, y in zip(df["wavenumber_cm1"], df["intensity_a"])]
    insert_rows("raman_spectra", rows)
    return mid

def persist_four_point(sample_id: int, uploaded) -> int:
    df = parse_four_point(uploaded)
    mid = create_measurement(sample_id, "4_pontas")
    rows = df.to_dict(orient="records")
    for r in rows: r["measurement_id"] = mid
    insert_rows("four_point_probe_points", rows)
    return mid

def persist_contact_angle(sample_id: int, uploaded) -> int:
    df = parse_contact_angle(uploaded)
    mid = create_measurement(sample_id, "tensiometria")
    rows = df.to_dict(orient="records")
    for r in rows: r["measurement_id"] = mid
    insert_rows("contact_angle_points", rows)
    return mid

# ========================== MODELOS DE PICO ==========================
def gauss(x, A, x0, sigma):
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

def lorentz(x, A, x0, gamma):
    return A / (1.0 + ((x - x0) / gamma) ** 2)

def voigt_like(x, A, x0, sigma, gamma):
    # Aproximação simples (não é Voigt exato, evita dependência extra)
    return 0.5 * (gauss(x, A, x0, sigma) + lorentz(x, A, x0, gamma))

def build_model(kind, n):
    if kind == "Gaussiana":
        def model(x, *p):
            y = np.zeros_like(x, dtype=float)
            for i in range(n):
                A, x0, sigma = p[3*i:3*i+3]
                y += gauss(x, A, x0, sigma)
            return y
        npar = 3
    elif kind == "Voigt":
        def model(x, *p):
            y = np.zeros_like(x, dtype=float)
            for i in range(n):
                A, x0, sigma, gamma = p[4*i:4*i+4]
                y += voigt_like(x, A, x0, sigma, gamma)
            return y
        npar = 4
    else:  # Lorentziana
        def model(x, *p):
            y = np.zeros_like(x, dtype=float)
            for i in range(n):
                A, x0, gamma = p[3*i:3*i+3]
                y += lorentz(x, A, x0, gamma)
            return y
        npar = 3
    return model, npar

def initial_guess_from_peaks(x, y, n, kind):
    # detectar picos iniciais
    idx, props = find_peaks(y, prominence=0.03, distance=max(1, len(x)//50))
    idx = idx[np.argsort(y[idx])[::-1]]  # ordenar por altura
    idx = idx[:n] if len(idx) >= n else idx
    idx = np.sort(idx)
    x0s = x[idx] if len(idx) else np.linspace(x.min(), x.max(), n)
    As  = y[idx] if len(idx) else np.full(n, y.max()/n)

    # larguras iniciais aproximadas
    width = (x.max() - x.min()) / (12 if n <= 7 else 16)
    p0 = []
    if kind == "Gaussiana":
        for A, x0 in zip(As, x0s): p0 += [max(A, 1e-3), x0, max(width/2, 1.0)]
    elif kind == "Voigt":
        for A, x0 in zip(As, x0s): p0 += [max(A, 1e-3), x0, max(width/2, 1.0), max(width/2, 1.0)]
    else:  # Lorentziana
        for A, x0 in zip(As, x0s): p0 += [max(A, 1e-3), x0, max(width/2, 1.0)]
    return np.array(p0), x0s

def bounds_for_model(x, kind, n):
    lo, hi = [], []
    xmin, xmax = x.min(), x.max()
    if kind == "Gaussiana":
        for _ in range(n):
            lo += [0.0, xmin, 0.1]
            hi += [np.inf, xmax, (xmax - xmin)]
    elif kind == "Voigt":
        for _ in range(n):
            lo += [0.0, xmin, 0.1, 0.1]
            hi += [np.inf, xmax, (xmax - xmin), (xmax - xmin)]
    else:  # Lorentziana
        for _ in range(n):
            lo += [0.0, xmin, 0.1]
            hi += [np.inf, xmax, (xmax - xmin)]
    return (np.array(lo), np.array(hi))

def fit_peaks(x, y, kind="Lorentziana", n_peaks=7):
    model, npar = build_model(kind, n_peaks)
    p0, guess_centers = initial_guess_from_peaks(x, y, n_peaks, kind)
    bounds = bounds_for_model(x, kind, n_peaks)
    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
        return popt, model, npar, guess_centers
    except Exception as e:
        return None, model, npar, guess_centers

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
                if tipo_upload == "Raman":
                    mid = persist_raman(sid, uploaded)
                elif tipo_upload == "4 Pontas":
                    mid = persist_four_point(sid, uploaded)
                else:
                    mid = persist_contact_angle(sid, uploaded)
                st.success(f"✅ Ensaio {tipo_upload} salvo! amostra={sample_name} (id={sid}), measurement={mid}")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Erro ao salvar ensaio: {e}")

# ------------------ ABA 2: Visualização + Ajuste de Picos ------------------
with tab2:
    st.header("📊 Visualização de ensaios salvos")
    df_samples = load_samples_df()
    if df_samples.empty:
        st.info("Nenhuma amostra cadastrada ainda.")
        st.stop()

    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        sample_sel = st.selectbox("Amostra:", df_samples["sample_name"].tolist())
    with c2:
        tipo_view = st.radio("Tipo:", ["Raman", "Ângulo de Contato", "4 Pontas"], horizontal=False, index=0)
    with c3:
        pass

    sid = int(df_samples[df_samples["sample_name"] == sample_sel]["id"].values[0])

    if tipo_view == "Raman":
        # parâmetros de ajuste
        st.markdown("#### ⚙️ Parâmetros do ajuste")
        colp1, colp2, colp3 = st.columns(3)
        with colp1:
            n_peaks = st.number_input("Nº máx. de picos", min_value=3, max_value=12, value=7, step=1)
        with colp2:
            func_kind = st.selectbox("Função do pico", ["Lorentziana", "Gaussiana", "Voigt"], index=0)
        with colp3:
            thresh_rel = st.slider("Threshold relativo (detecção)", 0.01, 0.2, 0.05, 0.01)

        mid = get_latest_measurement(sid, "raman")
        if mid is None:
            st.info("Sem ensaio Raman para esta amostra.")
            st.stop()

        res = supabase.table("raman_spectra").select("wavenumber_cm1,intensity_a").eq("measurement_id", mid)\
                .order("wavenumber_cm1").execute()
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

        # picos rápidos para exibir marcadores
        try:
            rpeaks = s.find_peaks(threshold_rel=float(thresh_rel))
            quick_pos = np.array(getattr(rpeaks, "x", []), dtype=float)
            quick_int = np.array(getattr(rpeaks, "y", []), dtype=float)
        except Exception:
            qidx, _ = find_peaks(s.y, prominence=0.03)
            quick_pos, quick_int = s.x[qidx], s.y[qidx]

        # Ajuste não linear
        popt, model, npar, guess_centers = fit_peaks(s.x, s.y, kind=func_kind, n_peaks=int(n_peaks))

        st.subheader(f"📈 Raman – {sample_sel} (measurement #{mid})")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(s.x, s.y, lw=1.2, label="Espectro tratado")

        if popt is not None:
            # desenhar componentes
            colors = plt.cm.tab10(np.linspace(0, 1, n_peaks))
            X = s.x
            Ysum = np.zeros_like(X, dtype=float)
            comps = []

            if func_kind == "Gaussiana":
                for i in range(n_peaks):
                    A, x0, sigma = popt[3*i:3*i+3]
                    comp = gauss(X, A, x0, sigma)
                    ax.fill_between(X, 0, comp, alpha=0.4, step="mid", label=f"p{i+1} ({x0:.1f} cm⁻¹)", color=colors[i])
                    Ysum += comp
                    comps.append((x0, A, sigma, 2.355*sigma))  # FWHM ≈ 2.355σ
            elif func_kind == "Voigt":
                for i in range(n_peaks):
                    A, x0, sigma, gamma = popt[4*i:4*i+4]
                    comp = voigt_like(X, A, x0, sigma, gamma)
                    ax.fill_between(X, 0, comp, alpha=0.4, step="mid", label=f"p{i+1} ({x0:.1f} cm⁻¹)", color=colors[i])
                    Ysum += comp
                    # FWHM aproximada (média)
                    fwhm = 0.5*(2.355*sigma + 2*gamma)
                    comps.append((x0, A, (sigma, gamma), fwhm))
            else:  # Lorentziana
                for i in range(n_peaks):
                    A, x0, gamma = popt[3*i:3*i+3]
                    comp = lorentz(X, A, x0, gamma)
                    ax.fill_between(X, 0, comp, alpha=0.4, step="mid", label=f"p{i+1} ({x0:.1f} cm⁻¹)", color=colors[i])
                    Ysum += comp
                    comps.append((x0, A, gamma, 2*gamma))  # FWHM = 2γ

            ax.plot(X, Ysum, linestyle="--", lw=1.2, label="Soma ajustada")
            ax.legend(ncol=3, fontsize=8)
        else:
            ax.scatter(quick_pos, quick_int, color="red", s=22, label="Picos detectados")
            ax.legend(fontsize=8)
            st.info("Ajuste não convergiu — exibindo apenas picos detectados.")

        ax.invert_xaxis()
        ax.set_xlabel("Deslocamento Raman (cm⁻¹)")
        ax.set_ylabel("Intensidade (a.u.)")
        st.pyplot(fig)

        # Tabela de picos ajustados
        if popt is not None:
            if func_kind == "Gaussiana":
                data_rows = []
                for i in range(n_peaks):
                    A, x0, sigma = popt[3*i:3*i+3]
                    data_rows.append({"cm⁻¹": round(x0,2), "Amplitude": A, "Largura": sigma, "FWHM": 2.355*sigma})
                peaks_df = pd.DataFrame(data_rows)
            elif func_kind == "Voigt":
                data_rows = []
                for i in range(n_peaks):
                    A, x0, sigma, gamma = popt[4*i:4*i+4]
                    fwhm = 0.5*(2.355*sigma + 2*gamma)
                    data_rows.append({"cm⁻¹": round(x0,2), "Amplitude": A, "σ": sigma, "γ": gamma, "FWHM≈": fwhm})
                peaks_df = pd.DataFrame(data_rows)
            else:  # Lorentz
                data_rows = []
                for i in range(n_peaks):
                    A, x0, gamma = popt[3*i:3*i+3]
                    data_rows.append({"cm⁻¹": round(x0,2), "Amplitude": A, "γ (HWHM)": gamma, "FWHM": 2*gamma})
                peaks_df = pd.DataFrame(data_rows)

            st.write("### 🔎 Picos ajustados")
            st.dataframe(peaks_df)

            # Atribuições moleculares para posições ajustadas
            id_table = match_peaks_to_assignments(peaks_df["cm⁻¹"].to_numpy(), tol_cm1=8.0)
            st.write("### 🧬 Atribuições moleculares")
            st.dataframe(id_table if not id_table.empty else pd.DataFrame(
                columns=["peak_cm1","Δ(cm⁻¹)","ref_cm1","atribuição","componente"]
            ))
        else:
            # fallback: tabela com picos rápidos
            df_peaks = pd.DataFrame({
                "Posição (cm⁻¹)": np.round(quick_pos, 1),
                "Intensidade (a.u.)": np.round(quick_int, 3),
            })
            st.write("### 🔎 Picos detectados (sem ajuste)")
            st.dataframe(df_peaks)
            id_table = match_peaks_to_assignments(quick_pos, tol_cm1=8.0)
            st.write("### 🧬 Atribuições moleculares")
            st.dataframe(id_table if not id_table.empty else pd.DataFrame(
                columns=["peak_cm1","Δ(cm⁻¹)","ref_cm1","atribuição","componente"]
            ))

    elif tipo_view == "4 Pontas":
        mid = get_latest_measurement(sid, "4_pontas")
        if mid is None:
            st.info("Sem ensaio de 4 Pontas para esta amostra.")
            st.stop()
        res = supabase.table("four_point_probe_points").select("*").eq("measurement_id", mid).order("id").execute()
        df = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        if df.empty:
            st.info("Sem pontos.")
        else:
            fig, ax = plt.subplots(figsize=(7,4))
            ax.scatter(df["current_a"], df["voltage_v"], label="Dados")
            ax.plot(df["current_a"], df["current_a"] * df["resistance_ohm"].mean(), label="Ajuste médio")
            ax.set_xlabel("Corrente (A)")
            ax.set_ylabel("Tensão (V)")
            ax.legend()
            st.pyplot(fig)
            st.dataframe(df)

    else:  # Ângulo de Contato
        mid = get_latest_measurement(sid, "tensiometria")
        if mid is None:
            st.info("Sem ensaio de Ângulo de Contato para esta amostra.")
            st.stop()
        res = supabase.table("contact_angle_points").select("*").eq("measurement_id", mid).order("id").execute()
        df = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        if df.empty:
            st.info("Sem pontos.")
        else:
            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(df["t_seconds"], df["angle_mean_deg"], "o-")
            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Ângulo (°)")
            st.pyplot(fig)
            st.dataframe(df)

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
