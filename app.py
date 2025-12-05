# -*- coding: utf-8 -*-
"""
SurfaceXLab — Versão Dark Silver
- Tema dark (Datadog/Splunk like)
- Plotly interativo em dark
- Export CSV / Excel para tabelas e picos
- Painel lateral (slide-over) para visualizar tabela ao lado do gráfico
Mantém toda a lógica (Supabase, parsers, persistência, ajuste de picos, ML).
"""
import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Plotly
import plotly.graph_objects as go
import plotly.express as px

# robust Spectrum import
Spectrum = None
try:
    from ramanchada2.spectrum import Spectrum
except Exception:
    try:
        import ramanchada2 as rc2
        Spectrum = getattr(rc2, "Spectrum", None)
    except Exception:
        Spectrum = None

# -------------------- Streamlit config --------------------
st.set_page_config(page_title="SurfaceXLab — Dark", layout="wide")
st.markdown("<h2 style='margin:6px 0 0 0;color:#F5F7FA'>SurfaceXLab — Plataforma de Caracterização (Dark)</h2>", unsafe_allow_html=True)
st.markdown("<div style='color:#B8BCC2;margin-bottom:12px;'>Tema Dark Silver • cards escuros • tabelas exportáveis • painel lateral</div>", unsafe_allow_html=True)

# -------------------- Dark Silver CSS --------------------
DARK_CSS = """
<style>
:root{
  --bg: #181A1F;         /* fundo */
  --card: #202328;       /* card */
  --card-2: #23262b;
  --muted: #A6A9AE;      /* texto secundário */
  --text: #F5F5F5;       /* texto principal */
  --accent: #9CA3AF;     /* prata metálico */
  --accent-2: #6B7280;
  --border: rgba(255,255,255,0.04);
  --shadow: 0 8px 24px rgba(0,0,0,0.6);
}

/* page background */
.reportview-container, .main, body {
  background-color: var(--bg) !important;
  color: var(--text);
}

/* card base */
.sx-card {
  background: linear-gradient(180deg, var(--card), var(--card-2));
  border-radius: 10px;
  padding: 14px;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  margin-bottom: 12px;
}

/* card header */
.sx-card-head {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  margin-bottom:8px;
}
.sx-card-title { font-size:14px; font-weight:700; color:var(--text); margin:0; }
.sx-card-sub { font-size:12px; color:var(--muted); margin-top:3px; }

/* KPI */
.sx-kpi { font-size:26px; font-weight:700; color:var(--text); }
.sx-kpi-sub { font-size:12px; color:var(--muted); }

/* buttons small */
.sx-btn {
  background: transparent;
  border: 1px solid rgba(255,255,255,0.06);
  padding:6px 10px;
  border-radius:8px;
  color:var(--muted);
  font-size:13px;
}

/* layout helpers */
.row-flex { display:flex; gap:12px; align-items:stretch; }
.col-1 { flex:1; min-width:0; }
.col-2 { flex:2; min-width:0; }
.col-3 { flex:3; min-width:0; }

/* small table panel */
.side-panel {
  background: linear-gradient(180deg, #1B1D21, #16181B);
  border-radius:8px;
  padding:10px;
  border:1px solid rgba(255,255,255,0.03);
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# -------------------- Minimal SVG (professional) --------------------
SVG_BAR = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><rect x="3" y="3" width="4" height="18" rx="1.5" stroke="#F5F5F5" stroke-width="1.1"/><rect x="10" y="9" width="4" height="12" rx="1.5" stroke="#F5F5F5" stroke-width="1.1"/><rect x="17" y="13" width="4" height="8" rx="1.5" stroke="#F5F5F5" stroke-width="1.1"/></svg>'

# -------------------- Helpers UI --------------------
def open_card(title: str, subtitle: str | None = None):
    subtitle_html = f"<div class='sx-card-sub'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"<div class='sx-card'><div class='sx-card-head'><div><div class='sx-card-title'>{title}</div>{subtitle_html}</div></div>", unsafe_allow_html=True)

def close_card():
    st.markdown("</div>", unsafe_allow_html=True)

def kpi_card(title: str, value: str, subtitle: str | None = None):
    open_card(title, subtitle)
    st.markdown(f"<div style='display:flex;align-items:center;justify-content:space-between'><div class='sx-kpi'>{value}</div></div>", unsafe_allow_html=True)
    close_card()

def plotly_card_container(title: str, subtitle: str | None = None):
    open_card(title, subtitle)

# -------------------- ASSIGNMENTS --------------------
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

# -------------------- SUPABASE --------------------
@st.cache_resource
def init_connection() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()
try:
    supabase.table("samples").select("id").limit(1).execute()
    st.sidebar.success("Conectado Supabase")
except Exception as e:
    st.sidebar.error(f"Erro Supabase: {e}")
    st.stop()

# -------------------- DB helpers --------------------
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

# -------------------- Parsers --------------------
def parse_raman(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    elif name.endswith(".csv") or name.endswith(".txt"):
        df = pd.read_csv(uploaded, sep=None, engine="python")
    else:
        raise ValueError("Formato não suportado. Use .csv, .txt ou .xlsx")
    df.columns = [c.lower().replace("#","").strip() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        if any(k in c for k in ["wave","shift","raman","x"]):
            rename_map[c] = "wavenumber_cm1"
        elif any(k in c for k in ["inten","signal","y"]):
            rename_map[c] = "intensity_a"
    df.rename(columns=rename_map, inplace=True)
    if not {"wavenumber_cm1","intensity_a"}.issubset(df.columns):
        raise ValueError(f"Colunas não reconhecidas. Detectadas: {list(df.columns)}")
    df = df.dropna(subset=["wavenumber_cm1","intensity_a"])
    return df[["wavenumber_cm1","intensity_a"]]

def parse_four_point(uploaded) -> pd.DataFrame:
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    if "current_a" not in df.columns:
        df.rename(columns={"corrente":"current_a","i":"current_a"}, inplace=True)
    if "voltage_v" not in df.columns:
        df.rename(columns={"tensao":"voltage_v","v":"voltage_v"}, inplace=True)
    df = df.dropna(subset=["current_a","voltage_v"])
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

# -------------------- Persist --------------------
def persist_raman(sample_id: int, uploaded) -> int:
    df = parse_raman(uploaded)
    mid = create_measurement(sample_id, "raman")
    rows = [{"measurement_id": mid, "wavenumber_cm1": float(x), "intensity_a": float(y)}
            for x,y in zip(df["wavenumber_cm1"], df["intensity_a"])]
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

# -------------------- Peak models --------------------
def gauss(x, A, x0, sigma):
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

def lorentz(x, A, x0, gamma):
    return A / (1.0 + ((x - x0) / gamma) ** 2)

def voigt_like(x, A, x0, sigma, gamma):
    return 0.5 * (gauss(x, A, x0, sigma) + lorentz(x, A, x0, gamma))

def build_model(kind, n):
    if kind == "Gaussiana":
        def model(x, *p):
            y = np.zeros_like(x, dtype=float)
            for i in range(n):
                A,x0,sigma = p[3*i:3*i+3]
                y += gauss(x,A,x0,sigma)
            return y
        npar = 3
    elif kind == "Voigt":
        def model(x, *p):
            y = np.zeros_like(x, dtype=float)
            for i in range(n):
                A,x0,sigma,gamma = p[4*i:4*i+4]
                y += voigt_like(x,A,x0,sigma,gamma)
            return y
        npar = 4
    else:
        def model(x, *p):
            y = np.zeros_like(x, dtype=float)
            for i in range(n):
                A,x0,gamma = p[3*i:3*i+3]
                y += lorentz(x,A,x0,gamma)
            return y
        npar = 3
    return model, npar

def initial_guess_from_peaks(x, y, n, kind):
    idx, props = find_peaks(y, prominence=0.03, distance=max(1, len(x)//50))
    idx = idx[np.argsort(y[idx])[::-1]]
    idx = idx[:n] if len(idx) >= n else idx
    idx = np.sort(idx)
    x0s = x[idx] if len(idx) else np.linspace(x.min(), x.max(), n)
    As  = y[idx] if len(idx) else np.full(n, y.max()/n)
    width = (x.max() - x.min()) / (12 if n <= 7 else 16)
    p0 = []
    if kind == "Gaussiana":
        for A,x0 in zip(As,x0s): p0 += [max(A,1e-3), x0, max(width/2,1.0)]
    elif kind == "Voigt":
        for A,x0 in zip(As,x0s): p0 += [max(A,1e-3), x0, max(width/2,1.0), max(width/2,1.0)]
    else:
        for A,x0 in zip(As,x0s): p0 += [max(A,1e-3), x0, max(width/2,1.0)]
    return np.array(p0), x0s

def bounds_for_model(x, kind, n):
    lo, hi = [], []
    xmin, xmax = x.min(), x.max()
    if kind == "Gaussiana":
        for _ in range(n):
            lo += [0.0, xmin, 0.1]
            hi += [np.inf, xmax, (xmax-xmin)]
    elif kind == "Voigt":
        for _ in range(n):
            lo += [0.0, xmin, 0.1, 0.1]
            hi += [np.inf, xmax, (xmax-xmin), (xmax-xmin)]
    else:
        for _ in range(n):
            lo += [0.0, xmin, 0.1]
            hi += [np.inf, xmax, (xmax-xmin)]
    return (np.array(lo), np.array(hi))

def fit_peaks(x, y, kind="Lorentziana", n_peaks=7):
    model, npar = build_model(kind, n_peaks)
    p0, guess_centers = initial_guess_from_peaks(x, y, n_peaks, kind)
    bounds = bounds_for_model(x, kind, n_peaks)
    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
        return popt, model, npar, guess_centers
    except Exception:
        return None, model, npar, guess_centers

# -------------------- Session state for side panel --------------------
if "show_side_table" not in st.session_state:
    st.session_state["show_side_table"] = False
if "side_table_df" not in st.session_state:
    st.session_state["side_table_df"] = None
if "side_table_title" not in st.session_state:
    st.session_state["side_table_title"] = ""

def toggle_side_table(df: pd.DataFrame | None = None, title: str = ""):
    # if df provided, open and set df; otherwise toggle
    if df is not None:
        st.session_state["side_table_df"] = df
        st.session_state["side_table_title"] = title
        st.session_state["show_side_table"] = True
    else:
        st.session_state["show_side_table"] = not st.session_state["show_side_table"]

# -------------------- Main UI: Tabs --------------------
tab1, tab2, tab3 = st.tabs(["1) Cadastro & Upload", "2) Visualização & Ajuste", "3) Otimização (IA)"])

# ---- TAB 1 ----
with tab1:
    left, right = st.columns([3,1])
    with left:
        open_card("Cadastrar amostra e enviar ensaio", "Nome, tipo e arquivo (.csv/.txt/.xlsx)")
        sample_name = st.text_input("Nome da amostra (ex: Amostra_01)")
        description = st.text_input("Descrição (opcional)")
        tipo_upload = st.radio("Tipo de ensaio:", ["Raman", "Ângulo de Contato", "4 Pontas"], horizontal=True)
        uploaded = st.file_uploader("Carregar arquivo", type=["csv","txt","xlsx"])
        if st.button("Enviar ao Supabase"):
            if not sample_name or not uploaded:
                st.warning("Informe nome da amostra e selecione arquivo.")
            else:
                try:
                    sid = insert_sample(sample_name, description)
                    if tipo_upload == "Raman":
                        mid = persist_raman(sid, uploaded)
                    elif tipo_upload == "4 Pontas":
                        mid = persist_four_point(sid, uploaded)
                    else:
                        mid = persist_contact_angle(sid, uploaded)
                    st.success(f"Ensaio salvo: amostra={sample_name} (id={sid}), measurement={mid}")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Erro ao salvar: {e}")
        close_card()
    with right:
        samples_df = load_samples_df()
        total_samples = len(samples_df) if not samples_df.empty else 0
        kpi_card("Amostras cadastradas", str(total_samples), "Total")
        try:
            rres = supabase.table("measurements").select("id").eq("type","raman").execute()
            total_raman = len(rres.data) if rres.data else 0
        except Exception:
            total_raman = 0
        kpi_card("Ensaios Raman", str(total_raman), "No DB")
        kpi_card("Conexão", "OK", "Supabase")

# ---- TAB 2 ----
with tab2:
    st.markdown("## Visualização de Ensaios")
    df_samples = load_samples_df()
    if df_samples.empty:
        st.info("Nenhuma amostra cadastrada.")
        st.stop()

    c1, c2 = st.columns([2,1])
    with c1:
        sample_sel = st.selectbox("Amostra", df_samples["sample_name"].tolist())
    with c2:
        tipo_view = st.radio("Tipo", ["Raman", "Ângulo de Contato", "4 Pontas"], index=0, horizontal=False)
    sid = int(df_samples[df_samples["sample_name"] == sample_sel]["id"].values[0])

    # Prepare UX layout depending on side panel state
    show_side = st.session_state["show_side_table"]

    if tipo_view == "Raman":
        st.markdown("### Parâmetros de ajuste")
        p1, p2, p3 = st.columns(3)
        with p1:
            n_peaks = st.number_input("Nº máx picos", min_value=3, max_value=12, value=7, step=1)
        with p2:
            func_kind = st.selectbox("Função", ["Lorentziana", "Gaussiana", "Voigt"], index=0)
        with p3:
            thresh_rel = st.slider("Threshold relativo", min_value=0.01, max_value=0.2, value=0.05, step=0.01)

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
        for step in ("remove_baseline","smooth","normalize"):
            try: s = getattr(s, step)()
            except Exception: pass

        try:
            rpeaks = s.find_peaks(threshold_rel=float(thresh_rel))
            quick_pos = np.array(getattr(rpeaks,"x",[]), dtype=float)
            quick_int = np.array(getattr(rpeaks,"y",[]), dtype=float)
        except Exception:
            qidx, _ = find_peaks(s.y, prominence=0.03)
            quick_pos, quick_int = s.x[qidx], s.y[qidx]

        popt, model, npar, guess_centers = fit_peaks(s.x, s.y, kind=func_kind, n_peaks=int(n_peaks))

        # Build Plotly dark figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.x, y=s.y, mode="lines", name="Espectro", line=dict(width=1.6, color="#F5F5F5")))
        colors = px.colors.qualitative.Dark24
        X = s.x
        Ysum = np.zeros_like(X, dtype=float)

        if popt is not None:
            if func_kind == "Gaussiana":
                for i in range(n_peaks):
                    A,x0,sigma = popt[3*i:3*i+3]
                    comp = gauss(X,A,x0,sigma)
                    Ysum += comp
                    fig.add_trace(go.Scatter(x=X, y=comp, mode="lines", name=f"p{i+1} ({x0:.1f})",
                                             line=dict(width=1), fill='tozeroy', opacity=0.35,
                                             marker=dict(color=colors[i % len(colors)])))
            elif func_kind == "Voigt":
                for i in range(n_peaks):
                    A,x0,sigma,gamma = popt[4*i:4*i+4]
                    comp = voigt_like(X,A,x0,sigma,gamma)
                    Ysum += comp
                    fig.add_trace(go.Scatter(x=X, y=comp, mode="lines", name=f"p{i+1} ({x0:.1f})",
                                             line=dict(width=1), fill='tozeroy', opacity=0.35,
                                             marker=dict(color=colors[i % len(colors)])))
            else:
                for i in range(n_peaks):
                    A,x0,gamma = popt[3*i:3*i+3]
                    comp = lorentz(X,A,x0,gamma)
                    Ysum += comp
                    fig.add_trace(go.Scatter(x=X, y=comp, mode="lines", name=f"p{i+1} ({x0:.1f})",
                                             line=dict(width=1), fill='tozeroy', opacity=0.35,
                                             marker=dict(color=colors[i % len(colors)])))
            fig.add_trace(go.Scatter(x=X, y=Ysum, mode="lines", name="Soma ajustada",
                                     line=dict(dash='dash', width=1.6, color="#9CA3AF")))
        else:
            fig.add_trace(go.Scatter(x=quick_pos, y=quick_int, mode="markers", name="Picos detectados",
                                     marker=dict(color='tomato', size=6)))

        # dark layout
        fig.update_layout(template="plotly_dark", height=420,
                          plot_bgcolor="#151617", paper_bgcolor="#151617",
                          margin=dict(l=40,r=10,t=30,b=30),
                          xaxis=dict(title="Deslocamento Raman (cm⁻¹)", color="#F5F5F5", autorange='reversed'),
                          yaxis=dict(title="Intensidade (a.u.)", color="#F5F5F5"))
        # Now render layout depending on side panel
        if st.session_state["show_side_table"]:
            # two columns: main (3/4) and side (1/4)
            main_col, side_col = st.columns([3,1])
            with main_col:
                plotly_card_container(f"Raman — {sample_sel} (#{mid})", f"Fit: {func_kind} • picos máx: {n_peaks}")
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config={"displaylogo":False})
                close_card()
            with side_col:
                st.markdown("<div class='side-panel'>", unsafe_allow_html=True)
                st.markdown(f"### {st.session_state['side_table_title']}")
                df_side = st.session_state["side_table_df"]
                if df_side is None or df_side.empty:
                    st.markdown("_Nenhuma tabela disponível._")
                else:
                    st.dataframe(df_side, use_container_width=True)
                    # export buttons
                    csv = df_side.to_csv(index=False).encode('utf-8')
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        df_side.to_excel(writer, index=False, sheet_name="tabela")
                    buf.seek(0)
                    st.download_button("Exportar CSV", csv, file_name=f"{st.session_state['side_table_title']}.csv", mime="text/csv")
                    st.download_button("Exportar Excel", buf, file_name=f"{st.session_state['side_table_title']}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                st.markdown("</div>", unsafe_allow_html=True)
                if st.button("Fechar tabela"):
                    toggle_side_table(None, "")
        else:
            # full-width chart with small action buttons below
            plotly_card_container(f"Raman — {sample_sel} (#{mid})", f"Fit: {func_kind} • picos máx: {n_peaks}")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit", config={"displaylogo":False})
            close_card()
            # Prepare peaks table and actions
            if popt is not None:
                # construct peaks_df as before
                if func_kind == "Gaussiana":
                    rows = []
                    for i in range(n_peaks):
                        A,x0,sigma = popt[3*i:3*i+3]
                        rows.append({"cm⁻¹": round(x0,2), "Amplitude": float(A), "σ": float(sigma), "FWHM": float(2.355*sigma)})
                    peaks_df = pd.DataFrame(rows)
                elif func_kind == "Voigt":
                    rows = []
                    for i in range(n_peaks):
                        A,x0,sigma,gamma = popt[4*i:4*i+4]
                        fwhm = 0.5*(2.355*sigma + 2*gamma)
                        rows.append({"cm⁻¹": round(x0,2), "Amplitude": float(A), "σ": float(sigma), "γ": float(gamma), "FWHM≈": float(fwhm)})
                    peaks_df = pd.DataFrame(rows)
                else:
                    rows = []
                    for i in range(n_peaks):
                        A,x0,gamma = popt[3*i:3*i+3]
                        rows.append({"cm⁻¹": round(x0,2), "Amplitude": float(A), "γ (HWHM)": float(gamma), "FWHM": float(2*gamma)})
                    peaks_df = pd.DataFrame(rows)
                # action row
                col_a, col_b, col_c = st.columns([1,1,6])
                with col_a:
                    if st.button("Ver tabela de picos →"):
                        toggle_side_table(peaks_df, f"Picos_{sample_sel}_{mid}")
                with col_b:
                    # exports inline
                    csv = peaks_df.to_csv(index=False).encode('utf-8')
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        peaks_df.to_excel(writer, index=False, sheet_name="peaks")
                    buf.seek(0)
                    st.download_button("CSV picos", csv, file_name=f"peaks_{sample_sel}_{mid}.csv", mime="text/csv", key=f"dl_peaks_csv_{mid}")
                    st.download_button("XLSX picos", buf, file_name=f"peaks_{sample_sel}_{mid}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_peaks_xlsx_{mid}")
                with col_c:
                    st.markdown("<div style='color:#A6A9AE'>Atribuições e tabela podem ser abertas no painel lateral.</div>", unsafe_allow_html=True)

                st.markdown("### 🔎 Picos ajustados")
                st.dataframe(peaks_df, use_container_width=True)
                id_table = match_peaks_to_assignments(peaks_df["cm⁻¹"].to_numpy(), tol_cm1=8.0)
                st.markdown("### 🧬 Atribuições moleculares")
                st.dataframe(id_table if not id_table.empty else pd.DataFrame(columns=["peak_cm1","Δ(cm⁻¹)","ref_cm1","atribuição","componente"]), use_container_width=True)
            else:
                df_peaks = pd.DataFrame({
                    "Posição (cm⁻¹)": np.round(quick_pos,1),
                    "Intensidade (a.u.)": np.round(quick_int,3)
                })
                col_a, col_b = st.columns([1,6])
                with col_a:
                    if st.button("Ver tabela de picos →"):
                        toggle_side_table(df_peaks, f"Picos_detectados_{sample_sel}_{mid}")
                with col_b:
                    st.markdown("<div style='color:#A6A9AE'>Ajuste não convergiu — mostra apenas picos detectados.</div>", unsafe_allow_html=True)
                st.markdown("### Picos detectados (sem ajuste)")
                st.dataframe(df_peaks, use_container_width=True)
                id_table = match_peaks_to_assignments(df_peaks["Posição (cm⁻¹)"].to_numpy(), tol_cm1=8.0)
                st.markdown("### Atribuições moleculares")
                st.dataframe(id_table if not id_table.empty else pd.DataFrame(columns=["peak_cm1","Δ(cm⁻¹)","ref_cm1","atribuição","componente"]), use_container_width=True)

    elif tipo_view == "4 Pontas":
        mid = get_latest_measurement(sid, "4_pontas")
        if mid is None:
            st.info("Sem ensaio 4 Pontas")
            st.stop()
        res = supabase.table("four_point_probe_points").select("*").eq("measurement_id", mid).order("id").execute()
        df = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        if df.empty:
            st.info("Sem pontos.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["current_a"], y=df["voltage_v"], mode="markers", name="Dados", marker=dict(size=6, color="#F5F5F5")))
            avgR = df["resistance_ohm"].mean()
            xline = np.linspace(df["current_a"].min(), df["current_a"].max(), 50)
            fig.add_trace(go.Scatter(x=xline, y=xline*avgR, mode="lines", name="Ajuste médio", line=dict(color="#9CA3AF")))
            fig.update_layout(template="plotly_dark", height=420, plot_bgcolor="#151617", paper_bgcolor="#151617",
                              margin=dict(l=40,r=10,t=30,b=30),
                              xaxis=dict(title="Corrente (A)", color="#F5F5F5"),
                              yaxis=dict(title="Tensão (V)", color="#F5F5F5"))
            if st.session_state["show_side_table"]:
                main_col, side_col = st.columns([3,1])
                with main_col:
                    plotly_card_container(f"4 Pontas — {sample_sel} (#{mid})")
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit", config={"displaylogo":False})
                    close_card()
                with side_col:
                    st.markdown("<div class='side-panel'>", unsafe_allow_html=True)
                    st.markdown(f"### Dados 4P • {sample_sel}")
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode('utf-8')
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False, sheet_name="4pontas")
                    buf.seek(0)
                    st.download_button("Exportar CSV", csv, file_name=f"4pontos_{sample_sel}_{mid}.csv", mime="text/csv")
                    st.download_button("Exportar Excel", buf, file_name=f"4pontos_{sample_sel}_{mid}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    st.markdown("</div>", unsafe_allow_html=True)
                    if st.button("Fechar tabela"):
                        toggle_side_table(None, "")
            else:
                plotly_card_container(f"4 Pontas — {sample_sel} (#{mid})")
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config={"displaylogo":False})
                close_card()
                col_a, col_b = st.columns([1,6])
                with col_a:
                    if st.button("Ver dados →"):
                        toggle_side_table(df, f"4pontos_{sample_sel}_{mid}")
                with col_b:
                    st.markdown("<div style='color:#A6A9AE'>Exporte ou abra os dados no painel lateral.</div>", unsafe_allow_html=True)

    else:  # contact angle
        mid = get_latest_measurement(sid, "tensiometria")
        if mid is None:
            st.info("Sem ensaio de ângulo")
            st.stop()
        res = supabase.table("contact_angle_points").select("*").eq("measurement_id", mid).order("id").execute()
        df = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        if df.empty:
            st.info("Sem pontos.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["t_seconds"], y=df["angle_mean_deg"], mode="lines+markers", name="θ(t)", line=dict(color="#F5F5F5")))
            fig.update_layout(template="plotly_dark", height=420, plot_bgcolor="#151617", paper_bgcolor="#151617",
                              margin=dict(l=40,r=10,t=30,b=30),
                              xaxis=dict(title="Tempo (s)", color="#F5F5F5"),
                              yaxis=dict(title="Ângulo (°)", color="#F5F5F5"))
            if st.session_state["show_side_table"]:
                main_col, side_col = st.columns([3,1])
                with main_col:
                    plotly_card_container(f"Ângulo de Contato — {sample_sel} (#{mid})")
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit", config={"displaylogo":False})
                    close_card()
                with side_col:
                    st.markdown("<div class='side-panel'>", unsafe_allow_html=True)
                    st.markdown(f"### Dados θ(t) • {sample_sel}")
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode('utf-8')
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False, sheet_name="tensiometria")
                    buf.seek(0)
                    st.download_button("Exportar CSV", csv, file_name=f"tensiometria_{sample_sel}_{mid}.csv", mime="text/csv")
                    st.download_button("Exportar Excel", buf, file_name=f"tensiometria_{sample_sel}_{mid}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    st.markdown("</div>", unsafe_allow_html=True)
                    if st.button("Fechar tabela"):
                        toggle_side_table(None, "")
            else:
                plotly_card_container(f"Ângulo de Contato — {sample_sel} (#{mid})")
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config={"displaylogo":False})
                close_card()
                col_a, col_b = st.columns([1,6])
                with col_a:
                    if st.button("Ver dados →"):
                        toggle_side_table(df, f"tensiometria_{sample_sel}_{mid}")
                with col_b:
                    st.markdown("<div style='color:#A6A9AE'>Abra os dados no painel lateral para export.</div>", unsafe_allow_html=True)

# ---- TAB 3: Otimização ----
with tab3:
    st.markdown("## Otimização — Random Forest (exemplo)")
    try:
        data = supabase.table("raman_spectra").select("wavenumber_cm1,intensity_a").execute()
        if not data.data:
            st.info("Nenhum dado Raman disponível para treinar.")
            st.stop()
        df = pd.DataFrame(data.data).dropna()
        X = df[["wavenumber_cm1"]].values
        y = df["intensity_a"].values
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        open_card("Desempenho do modelo", f"R² {r2:.3f} — MAE {mae:.3f} — RMSE {rmse:.3f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_test.squeeze(), y=y_test, mode="markers", name="Real", marker=dict(size=6, color="#F5F5F5")))
        fig.add_trace(go.Scatter(x=X_test.squeeze(), y=y_pred, mode="markers", name="Previsto", marker=dict(size=6, color="#9CA3AF")))
        fig.update_layout(template="plotly_dark", height=420, plot_bgcolor="#151617", paper_bgcolor="#151617",
                          margin=dict(l=40,r=10,t=30,b=30),
                          xaxis=dict(title="Número de onda (cm⁻¹)", color="#F5F5F5"),
                          yaxis=dict(title="Intensidade (a.u.)", color="#F5F5F5"))
        st.plotly_chart(fig, use_container_width=True, theme="streamlit", config={"displaylogo":False})
        close_card()
    except Exception as e:
        st.error(f"Erro otimização: {e}")
