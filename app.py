# app.py
# -*- coding: utf-8 -*-
"""
SurfaceXLab — CRM-style single-file Streamlit app (Dark Silver)
- Navbar, KPI row, 3 main analysis tabs (Raman, 4-pontas, Tensiometria)
- AG-Grid for tables (st_aggrid)
- Plotly interactive charts
- Side panel "slide-over" implemented with columns + session_state
- Tries to import processing functions from local modules if present
"""

from typing import Optional
import io
import importlib
import base64
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client, Client
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# AG-Grid
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

st.set_page_config(page_title="SurfaceXLab CRM", layout="wide")
# ---------- Theme CSS (Dark Silver / CRM-like) ----------
CSS = """
<style>
:root{
  --bg: #181A1F; --card: #202328; --card-2:#23262b;
  --muted:#A6A9AE; --text:#F5F5F5; --accent:#9CA3AF;
  --border: rgba(255,255,255,0.04);
  --shadow: 0 8px 24px rgba(0,0,0,0.6);
}
/* page */
.reportview-container, .main, body { background-color: var(--bg) !important; color:var(--text); }
/* navbar */
.sx-navbar { background: linear-gradient(90deg,#151619,#1b1d21); padding:12px 18px; border-bottom:1px solid rgba(255,255,255,0.03); display:flex; align-items:center; gap:20px; }
.sx-logo { font-weight:700; color:var(--text); font-size:16px; }
.sx-navlink { color:var(--muted); padding:8px 10px; border-radius:8px; text-decoration:none; }
.sx-navlink.active { color:var(--text); background: rgba(255,255,255,0.02); }
/* card */
.sx-card{ background:linear-gradient(180deg,var(--card),var(--card-2)); border-radius:10px; padding:12px; border:1px solid var(--border); box-shadow:var(--shadow); margin-bottom:12px;}
.sx-card-title{font-weight:700;color:var(--text); font-size:14px;}
.sx-card-sub{color:var(--muted);font-size:12px;margin-top:6px;}
.sx-kpi{font-size:26px;font-weight:700;color:var(--text);}
/* side panel */
.side-panel{ background: linear-gradient(180deg,#1B1D21,#16181B); border-radius:8px; padding:12px; border:1px solid rgba(255,255,255,0.03); }
.sx-smallbtn{ background:transparent; color:var(--muted); padding:6px 10px; border-radius:8px; border:1px solid rgba(255,255,255,0.03); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- Attempt to import user's processing modules (fallback if absent) ----------
# Expected functions (try these names): parse_raman, parse_four_point, parse_contact_angle,
# fit_peaks (or functions for peak modelling), plus any helpers.
user_modules = {}
for modname in ("raman_processing", "resistivity_processing", "contact_angle_processing"):
    try:
        user_modules[modname] = importlib.import_module(modname)
    except Exception:
        user_modules[modname] = None

# ---------- Internal fallback parsers and models (used if user's modules not found) ----------
def fallback_parse_raman(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded, sep=None, engine="python")
    df.columns = [c.lower().replace("#", "").strip() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        if any(k in c for k in ["wave", "shift", "raman", "x"]):
            rename_map[c] = "wavenumber_cm1"
        elif any(k in c for k in ["inten", "signal", "y"]):
            rename_map[c] = "intensity_a"
    df.rename(columns=rename_map, inplace=True)
    if not {"wavenumber_cm1", "intensity_a"}.issubset(df.columns):
        raise ValueError("Arquivo Raman sem colunas reconhecíveis.")
    return df[["wavenumber_cm1", "intensity_a"]].dropna()

def fallback_parse_four_point(uploaded) -> pd.DataFrame:
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    if "current_a" not in df.columns:
        df.rename(columns={"corrente":"current_a","i":"current_a"}, inplace=True)
    if "voltage_v" not in df.columns:
        df.rename(columns={"tensao":"voltage_v","v":"voltage_v"}, inplace=True)
    df = df.dropna(subset=["current_a","voltage_v"])
    df["resistance_ohm"] = df["voltage_v"] / df["current_a"]
    return df[["current_a","voltage_v","resistance_ohm"]]

def fallback_parse_contact_angle(uploaded) -> pd.DataFrame:
    df = pd.read_csv(uploaded, sep=None, engine="python")
    df.columns = [c.lower().strip() for c in df.columns]
    if "mean" in df.columns and "angle_mean_deg" not in df.columns:
        df.rename(columns={"mean":"angle_mean_deg"}, inplace=True)
    if "time" in df.columns and "t_seconds" not in df.columns:
        df.rename(columns={"time":"t_seconds"}, inplace=True)
    df = df.dropna(subset=["t_seconds","angle_mean_deg"])
    return df[["t_seconds","angle_mean_deg"]]

# Peak model helpers (same as before)
def gauss(x, A, x0, sigma): return A * np.exp(-0.5 * ((x-x0)/sigma)**2)
def lorentz(x, A, x0, gamma): return A / (1.0 + ((x-x0)/gamma)**2)
def voigt_like(x, A, x0, sigma, gamma): return 0.5*(gauss(x,A,x0,sigma) + lorentz(x,A,x0,gamma))

def fallback_fit_peaks(x, y, kind="Lorentziana", n_peaks=7):
    # simple implementation: use find_peaks for guesses and don't try heavy curve_fit if not converging
    try:
        indices, props = find_peaks(y, prominence=np.max(y)*0.03, distance=max(1, len(x)//50))
    except Exception:
        indices = np.array([], dtype=int)
    if len(indices) == 0:
        # return quick markers only
        return None, indices
    # sort by intensity and pick top n_peaks
    order = np.argsort(y[indices])[::-1][:n_peaks]
    chosen = indices[order]
    centers = np.sort(x[chosen])
    return None, centers

# ---------- Supabase (expects st.secrets to have SUPABASE_URL and SUPABASE_KEY) ----------
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Defina SUPABASE_URL e SUPABASE_KEY em st.secrets")
    return create_client(url, key)

try:
    supabase = init_supabase()
    st.sidebar.success("Supabase conectado")
except Exception as e:
    supabase = None
    st.sidebar.error(f"Supabase não disponível: {e}")

# ---------- Utility: AG-Grid helper ----------
def show_aggrid(df: pd.DataFrame, height: int = 300):
    if AGGRID_AVAILABLE:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
        gb.configure_side_bar()
        gb.configure_selection(selection_mode="single", use_checkbox=False)
        gridOptions = gb.build()
        grid_response = AgGrid(df, gridOptions=gridOptions, height=height,
                               data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                               update_mode=GridUpdateMode.SELECTION_CHANGED)
        return grid_response
    else:
        st.dataframe(df, height=height)
        return None

# ---------- Side panel control ----------
if "side_open" not in st.session_state:
    st.session_state.side_open = False
if "side_df" not in st.session_state:
    st.session_state.side_df = None
if "side_title" not in st.session_state:
    st.session_state.side_title = ""

def open_side(df: Optional[pd.DataFrame], title: str):
    st.session_state.side_open = True
    st.session_state.side_df = df
    st.session_state.side_title = title

def close_side():
    st.session_state.side_open = False
    st.session_state.side_df = None
    st.session_state.side_title = ""

# ---------- Small UI helpers ----------
def nav_bar(active: str):
    st.markdown(f"""
    <div class="sx-navbar">
      <div class="sx-logo">SurfaceXLab</div>
      <div style="display:flex;gap:8px;">
        <a class="sx-navlink {'active' if active=='dashboard' else ''}">Dashboard</a>
        <a class="sx-navlink {'active' if active=='raman' else ''}">Raman</a>
        <a class="sx-navlink {'active' if active=='four' else ''}">4 Pontas</a>
        <a class="sx-navlink {'active' if active=='tensi' else ''}">Tensiometria</a>
        <a class="sx-navlink {'active' if active=='ml' else ''}">Otimização</a>
      </div>
      <div style="margin-left:auto;color:var(--muted)">Usuário • SurfaceXLab</div>
    </div>
    """, unsafe_allow_html=True)

def open_card(title: str, subtitle: str = ""):
    st.markdown(f"<div class='sx-card'><div class='sx-card-title'>{title}</div>{('<div class=\\'sx-card-sub\\'>%s</div>'%subtitle) if subtitle else ''}", unsafe_allow_html=True)

def close_card():
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Top navbar and KPI row ----------
nav_bar("dashboard")
# KPI row
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    open_card("Total Amostras")
    try:
        res = supabase.table("samples").select("id").execute() if supabase else None
        total_samples = len(res.data) if res and res.data else 0
    except Exception:
        total_samples = 0
    st.markdown(f"<div class='sx-kpi'>{total_samples}</div>", unsafe_allow_html=True)
    close_card()
with k2:
    open_card("Ensaios Raman")
    try:
        res = supabase.table("measurements").select("id").eq("type","raman").execute() if supabase else None
        total_raman = len(res.data) if res and res.data else 0
    except Exception:
        total_raman = 0
    st.markdown(f"<div class='sx-kpi'>{total_raman}</div>", unsafe_allow_html=True)
    close_card()
with k3:
    open_card("Ensaios 4 Pontas")
    try:
        res = supabase.table("measurements").select("id").eq("type","4_pontas").execute() if supabase else None
        total_4p = len(res.data) if res and res.data else 0
    except Exception:
        total_4p = 0
    st.markdown(f"<div class='sx-kpi'>{total_4p}</div>", unsafe_allow_html=True)
    close_card()
with k4:
    open_card("Ensaios Tensiometria")
    try:
        res = supabase.table("measurements").select("id").eq("type","tensiometria").execute() if supabase else None
        total_tensi = len(res.data) if res and res.data else 0
    except Exception:
        total_tensi = 0
    st.markdown(f"<div class='sx-kpi'>{total_tensi}</div>", unsafe_allow_html=True)
    close_card()
with k5:
    open_card("Modelos IA")
    st.markdown(f"<div class='sx-kpi'>—</div>", unsafe_allow_html=True)
    close_card()

st.markdown("<br>")

# ---------- Main Tabs (Raman / 4P / Tensi / Otimização) ----------
tabs = st.tabs(["🔬 Raman", "⚡ 4 Pontas", "🧪 Tensiometria", "🤖 Otimização"])
# ---------- RAMAN TAB ----------
with tabs[0]:
    st.header("Análises Moleculares — Raman")
    c1, c2 = st.columns([2,1])
    with c1:
        open_card("Upload e Cadastro")
        sample_name = st.text_input("Nome da amostra (Raman)")
        description = st.text_input("Descrição (opcional)")
        file_up = st.file_uploader("Upload (.csv/.txt/.xlsx)", type=["csv","txt","xlsx"])
        up_type = st.selectbox("Salvar como (supabase measurement type)", ["raman"], index=0)
        if st.button("Cadastrar e Salvar (Raman)"):
            if not sample_name or not file_up:
                st.warning("Informe nome da amostra e arquivo")
            else:
                # insert sample
                try:
                    if supabase:
                        # insert sample if not exists
                        r = supabase.table("samples").select("id").eq("sample_name", sample_name).execute()
                        if r.data:
                            sid = r.data[0]["id"]
                        else:
                            sid = supabase.table("samples").insert({"sample_name": sample_name, "description": description}).execute().data[0]["id"]
                        # parse with user module if available
                        if user_modules.get("raman_processing") and hasattr(user_modules["raman_processing"], "parse_raman"):
                            df_r = user_modules["raman_processing"].parse_raman(file_up)
                        else:
                            df_r = fallback_parse_raman(file_up)
                        # create measurement
                        mid = supabase.table("measurements").insert({"sample_id": sid, "type":"raman"}).execute().data[0]["id"]
                        rows = [{"measurement_id": mid, "wavenumber_cm1": float(x), "intensity_a": float(y)} for x,y in zip(df_r["wavenumber_cm1"], df_r["intensity_a"])]
                        # batch insert
                        for i in range(0, len(rows), 500):
                            chunk = rows[i:i+500]
                            supabase.table("raman_spectra").insert(chunk).execute()
                        st.success(f"Salvo: sample_id={sid} measurement_id={mid}")
                    else:
                        st.error("Supabase não conectado.")
                except Exception as e:
                    st.error(f"Erro ao salvar: {e}")
        close_card()

        open_card("Ensaios Recentes")
        # list recent samples
        try:
            sres = supabase.table("samples").select("id,sample_name,created_at").order("id", desc=True).limit(20).execute() if supabase else None
            recent_df = pd.DataFrame(sres.data) if sres and sres.data else pd.DataFrame(columns=["id","sample_name","created_at"])
        except Exception:
            recent_df = pd.DataFrame(columns=["id","sample_name","created_at"])
        if not recent_df.empty:
            if AGGRID_AVAILABLE:
                show_aggrid(recent_df)
            else:
                st.dataframe(recent_df)
        else:
            st.info("Nenhuma amostra recente.")
        close_card()

    with c2:
        open_card("Ações Rápidas", "Abrir painel lateral com dados/tabela")
        if st.button("Abrir painel lateral (última tabela de picos)"):
            # read last measurement and open placeholder
            try:
                last = supabase.table("measurements").select("id,sample_id").eq("type","raman").order("id", desc=True).limit(1).execute()
                if last and last.data:
                    mid = last.data[0]["id"]
                    res = supabase.table("raman_spectra").select("*").eq("measurement_id", mid).order("wavenumber_cm1").execute()
                    df_side = pd.DataFrame(res.data) if res and res.data else pd.DataFrame()
                    open_side(df_side, f"Raman_raw_m{mid}")
                else:
                    st.info("Sem ensaios Raman.")
            except Exception as e:
                st.error(f"Erro: {e}")
        close_card()

    # Visualization block (full width below)
    st.markdown("<br>")
    open_card("Visualização — Raman (escolha amostra abaixo)")
    # selector
    try:
        samples = supabase.table("samples").select("id,sample_name").order("id", desc=True).execute().data if supabase else []
        samples_df = pd.DataFrame(samples) if samples else pd.DataFrame(columns=["id","sample_name"])
    except Exception:
        samples_df = pd.DataFrame(columns=["id","sample_name"])
    if samples_df.empty:
        st.info("Cadastre amostras primeiro.")
    else:
        sel = st.selectbox("Selecione amostra", samples_df["sample_name"].tolist())
        sid = int(samples_df[samples_df["sample_name"]==sel]["id"].values[0])
        mid = None
        try:
            mres = supabase.table("measurements").select("id").eq("sample_id", sid).eq("type","raman").order("id", desc=True).limit(1).execute()
            if mres and mres.data:
                mid = mres.data[0]["id"]
        except Exception:
            mid = None
        if mid is None:
            st.info("Sem measurement Raman nesta amostra.")
        else:
            res = supabase.table("raman_spectra").select("wavenumber_cm1,intensity_a").eq("measurement_id", mid).order("wavenumber_cm1").execute()
            df_r = pd.DataFrame(res.data) if res and res.data else pd.DataFrame()
            if df_r.empty:
                st.info("Sem pontos Raman.")
            else:
                # preprocessing using user's module if offered
                if user_modules.get("raman_processing") and hasattr(user_modules["raman_processing"], "preprocess_spectrum"):
                    try:
                        sproc = user_modules["raman_processing"].preprocess_spectrum(df_r)
                        x = sproc["x"]; y = sproc["y"]
                    except Exception:
                        x = df_r["wavenumber_cm1"].values; y = df_r["intensity_a"].values
                else:
                    x = df_r["wavenumber_cm1"].values; y = df_r["intensity_a"].values

                # parameters
                col_p1, col_p2, col_p3 = st.columns([1,1,2])
                with col_p1:
                    n_peaks = st.number_input("Nº máximo picos", min_value=3, max_value=12, value=7)
                with col_p2:
                    func_kind = st.selectbox("Função pico", ["Lorentziana", "Gaussiana", "Voigt"])
                with col_p3:
                    thresh = st.slider("Threshold relativo", 0.01, 0.2, 0.05, 0.01)

                # first try user's fit implementation if exists
                popt = None
                centers = None
                if user_modules.get("raman_processing") and hasattr(user_modules["raman_processing"], "fit_peaks"):
                    try:
                        popt, model, meta = user_modules["raman_processing"].fit_peaks(x, y, kind=func_kind, n_peaks=n_peaks)
                    except Exception:
                        popt = None
                if popt is None:
                    # fallback to simple detection
                    _, centers = fallback_fit_peaks(x, y, kind=func_kind, n_peaks=n_peaks)

                # Build Plotly figure (dark)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Espectro", line=dict(color="#F5F5F5")))
                if popt is not None:
                    # try to plot components based on kind (best-effort)
                    if func_kind == "Gaussiana":
                        Ysum = np.zeros_like(x, dtype=float)
                        for i in range(n_peaks):
                            A,x0,sigma = popt[3*i:3*i+3]
                            comp = gauss(x,A,x0,sigma); Ysum += comp
                            fig.add_trace(go.Scatter(x=x, y=comp, mode="lines", name=f"p{i+1}", line=dict(width=1), fill='tozeroy', opacity=0.25))
                        fig.add_trace(go.Scatter(x=x, y=Ysum, mode="lines", name="Soma ajustada", line=dict(dash='dash', color="#9CA3AF")))
                else:
                    if centers is not None and len(centers)>0:
                        fig.add_trace(go.Scatter(x=centers, y=np.interp(centers, x, y), mode="markers", marker=dict(color="tomato", size=6), name="Picos detectados"))

                fig.update_layout(template="plotly_dark", plot_bgcolor="#151617", paper_bgcolor="#151617",
                                  xaxis=dict(autorange='reversed', title="cm⁻¹"), yaxis=dict(title="Intensidade"))
                st.plotly_chart(fig, use_container_width=True)

                # Peaks table and side operations
                if popt is not None:
                    # try to build peaks df from popt (best-effort)
                    peaks_rows = []
                    if func_kind == "Gaussiana":
                        for i in range(n_peaks):
                            A,x0,sigma = popt[3*i:3*i+3]
                            peaks_rows.append({"cm^-1": float(x0), "Amplitude": float(A), "sigma": float(sigma), "FWHM": float(2.355*sigma)})
                    elif func_kind == "Voigt":
                        for i in range(n_peaks):
                            A,x0,sigma,gamma = popt[4*i:4*i+4]
                            peaks_rows.append({"cm^-1": float(x0), "Amplitude": float(A), "sigma": float(sigma), "gamma": float(gamma)})
                    else:
                        for i in range(n_peaks):
                            A,x0,gamma = popt[3*i:3*i+3]
                            peaks_rows.append({"cm^-1": float(x0), "Amplitude": float(A), "gamma": float(gamma)})
                    peaks_df = pd.DataFrame(peaks_rows)
                else:
                    peaks_df = pd.DataFrame({"cm^-1": np.round(centers,2), "intensity": np.round(np.interp(centers, x, y),4)}) if centers is not None else pd.DataFrame()

                cols_a, cols_b, cols_c = st.columns([1,2,6])
                with cols_a:
                    if st.button("Abrir tabela lateral (picos)"):
                        open_side(peaks_df, f"Picos_Raman_m{mid}")
                with cols_b:
                    # export buttons
                    if not peaks_df.empty:
                        csv = peaks_df.to_csv(index=False).encode('utf-8')
                        st.download_button("CSV picos", csv, file_name=f"peaks_raman_{mid}.csv", mime="text/csv")
                        # excel
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                            peaks_df.to_excel(writer, index=False, sheet_name="peaks")
                        buf.seek(0)
                        st.download_button("XLSX picos", buf, file_name=f"peaks_raman_{mid}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                with cols_c:
                    st.markdown("<div style='color:#A6A9AE'>Use o painel lateral para visualizar/exportar tabelas completas.</div>", unsafe_allow_html=True)

                st.markdown("### Tabela de picos (preview)")
                if not peaks_df.empty:
                    if AGGRID_AVAILABLE:
                        show_aggrid(peaks_df, height=220)
                    else:
                        st.dataframe(peaks_df)
                else:
                    st.info("Nenhum pico identificado.")
    close_card()

    # Render side panel if open
    if st.session_state.side_open:
        # two-column layout with side panel
        left_col, right_col = st.columns([3,1])
        with left_col:
            st.markdown("")  # placeholder to keep layout
        with right_col:
            st.markdown("<div class='side-panel'>", unsafe_allow_html=True)
            st.markdown(f"### {st.session_state.side_title}")
            if st.session_state.side_df is None or st.session_state.side_df.empty:
                st.write("_Tabela vazia_")
            else:
                if AGGRID_AVAILABLE:
                    show_aggrid(st.session_state.side_df, height=400)
                else:
                    st.dataframe(st.session_state.side_df, height=400)
                csv = st.session_state.side_df.to_csv(index=False).encode('utf-8')
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    st.session_state.side_df.to_excel(writer, index=False, sheet_name="table")
                buf.seek(0)
                st.download_button("Exportar CSV", csv, file_name=f"{st.session_state.side_title}.csv", mime="text/csv")
                st.download_button("Exportar Excel", buf, file_name=f"{st.session_state.side_title}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            if st.button("Fechar painel"):
                close_side()
            st.markdown("</div>", unsafe_allow_html=True)

# ---------- 4 PONTAS TAB ----------
with tabs[1]:
    st.header("Análises Elétricas — 4 Pontas")
    c1, c2 = st.columns([2,1])
    with c1:
        open_card("Upload / Cadastro 4 Pontas")
        sname = st.text_input("Nome amostra (4 Pontas)")
        fdesc = st.text_input("Descrição (opcional)")
        fup = st.file_uploader("Arquivo (.csv/.txt/.xlsx)", type=["csv","txt","xlsx"], key="4p_up")
        if st.button("Salvar 4 Pontas"):
            if not sname or not fup:
                st.warning("Informe nome e arquivo")
            else:
                try:
                    if supabase:
                        r = supabase.table("samples").select("id").eq("sample_name", sname).execute()
                        if r.data:
                            sid = r.data[0]["id"]
                        else:
                            sid = supabase.table("samples").insert({"sample_name": sname, "description": fdesc}).execute().data[0]["id"]
                        # use user's parser if available
                        if user_modules.get("resistivity_processing") and hasattr(user_modules["resistivity_processing"], "parse_four_point"):
                            df4 = user_modules["resistivity_processing"].parse_four_point(fup)
                        else:
                            df4 = fallback_parse_four_point(fup)
                        mid = supabase.table("measurements").insert({"sample_id": sid, "type":"4_pontas"}).execute().data[0]["id"]
                        rows = df4.to_dict(orient="records")
                        for r in rows: r["measurement_id"] = mid
                        # insert in chunks
                        for i in range(0, len(rows), 500):
                            supabase.table("four_point_probe_points").insert(rows[i:i+500]).execute()
                        st.success(f"Salvo 4P sample={sname} mid={mid}")
                    else:
                        st.error("Supabase não conectado.")
                except Exception as e:
                    st.error(f"Erro: {e}")
        close_card()

        open_card("Visualização 4P")
        # select sample
        try:
            samples = supabase.table("samples").select("id,sample_name").order("id", desc=True).execute().data if supabase else []
            s_df = pd.DataFrame(samples) if samples else pd.DataFrame(columns=["id","sample_name"])
        except Exception:
            s_df = pd.DataFrame(columns=["id","sample_name"])
        if s_df.empty:
            st.info("Cadastre amostras")
        else:
            sel = st.selectbox("Selecione amostra", s_df["sample_name"].tolist(), key="sel4p")
            sid = int(s_df[s_df["sample_name"]==sel]["id"].values[0])
            mres = supabase.table("measurements").select("id").eq("sample_id", sid).eq("type","4_pontas").order("id", desc=True).limit(1).execute()
            if mres and mres.data:
                mid = mres.data[0]["id"]
                df4 = pd.DataFrame(supabase.table("four_point_probe_points").select("*").eq("measurement_id", mid).execute().data)
                if df4.empty:
                    st.info("Sem dados")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df4["current_a"], y=df4["voltage_v"], mode="markers", marker=dict(color="#F5F5F5")))
                    avgR = df4["resistance_ohm"].mean()
                    xline = np.linspace(df4["current_a"].min(), df4["current_a"].max(), 50)
                    fig.add_trace(go.Scatter(x=xline, y=xline*avgR, mode="lines", line=dict(color="#9CA3AF"), name="Ajuste médio"))
                    fig.update_layout(template="plotly_dark", plot_bgcolor="#151617", paper_bgcolor="#151617",
                                      xaxis_title="Corrente (A)", yaxis_title="Tensão (V)")
                    st.plotly_chart(fig, use_container_width=True)
                    if st.button("Abrir tabela lateral (4P)"):
                        open_side(df4, f"4P_m{mid}")
                    st.markdown("### Dados 4 Pontas (preview)")
                    if AGGRID_AVAILABLE:
                        show_aggrid(df4)
                    else:
                        st.dataframe(df4)
            else:
                st.info("Sem measurement 4P nesta amostra")
        close_card()
    with c2:
        open_card("Ações rápidas 4P")
        st.markdown("Exportar / Abrir painel lateral")
        close_card()

# ---------- TENSIOMETRIA TAB ----------
with tabs[2]:
    st.header("Análises Físico-mecânicas — Tensiometria")
    left, right = st.columns([2,1])
    with left:
        open_card("Upload / Cadastro Tensiometria")
        sname = st.text_input("Nome amostra (tensiometria)", key="tname")
        fdesc = st.text_input("Descrição (opcional)", key="tdesc")
        fup = st.file_uploader("Arquivo (.csv/.txt/.xlsx)", type=["csv","txt","xlsx"], key="tup")
        if st.button("Salvar tensiometria"):
            if not sname or not fup:
                st.warning("Informe nome e arquivo")
            else:
                try:
                    if supabase:
                        r = supabase.table("samples").select("id").eq("sample_name", sname).execute()
                        if r.data:
                            sid = r.data[0]["id"]
                        else:
                            sid = supabase.table("samples").insert({"sample_name": sname, "description": fdesc}).execute().data[0]["id"]
                        if user_modules.get("contact_angle_processing") and hasattr(user_modules["contact_angle_processing"], "parse_contact_angle"):
                            dfc = user_modules["contact_angle_processing"].parse_contact_angle(fup)
                        else:
                            dfc = fallback_parse_contact_angle(fup)
                        mid = supabase.table("measurements").insert({"sample_id": sid, "type":"tensiometria"}).execute().data[0]["id"]
                        rows = dfc.to_dict(orient="records")
                        for r in rows: r["measurement_id"] = mid
                        for i in range(0, len(rows), 500):
                            supabase.table("contact_angle_points").insert(rows[i:i+500]).execute()
                        st.success(f"Salvo tensiometria sample={sname} mid={mid}")
                    else:
                        st.error("Supabase não conectado.")
                except Exception as e:
                    st.error(f"Erro: {e}")
        close_card()

        open_card("Visualização θ(t)")
        # select sample
        try:
            samples = supabase.table("samples").select("id,sample_name").order("id", desc=True).execute().data if supabase else []
            s_df = pd.DataFrame(samples) if samples else pd.DataFrame(columns=["id","sample_name"])
        except Exception:
            s_df = pd.DataFrame(columns=["id","sample_name"])
        if not s_df.empty:
            sel = st.selectbox("Selecione amostra", s_df["sample_name"].tolist(), key="sel_t")
            sid = int(s_df[s_df["sample_name"]==sel]["id"].values[0])
            mres = supabase.table("measurements").select("id").eq("sample_id", sid).eq("type","tensiometria").order("id", desc=True).limit(1).execute()
            if mres and mres.data:
                mid = mres.data[0]["id"]
                df_t = pd.DataFrame(supabase.table("contact_angle_points").select("*").eq("measurement_id", mid).order("t_seconds").execute().data)
                if df_t.empty:
                    st.info("Sem pontos.")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_t["t_seconds"], y=df_t["angle_mean_deg"], mode="lines+markers", line=dict(color="#F5F5F5")))
                    fig.update_layout(template="plotly_dark", plot_bgcolor="#151617", paper_bgcolor="#151617",
                                      xaxis_title="Tempo (s)", yaxis_title="Ângulo (°)")
                    st.plotly_chart(fig, use_container_width=True)
                    if st.button("Abrir tabela lateral (tensiometria)"):
                        open_side(df_t, f"tensiometria_m{mid}")
                    st.markdown("### Dados (preview)")
                    if AGGRID_AVAILABLE:
                        show_aggrid(df_t)
                    else:
                        st.dataframe(df_t)
            else:
                st.info("Sem measurement tensiometria")
        else:
            st.info("Nenhuma amostra cadastrada")
        close_card()

# ---------- OTIMIZAÇÃO TAB ----------
with tabs[3]:
    st.header("Otimização & Correlações (IA)")
    open_card("Treinar modelos por ensaio")
    st.markdown("Exemplos: treinar RF com todos os pontos Raman / 4P / Tensiometria")
    cols = st.columns(3)
    with cols[0]:
        if st.button("Treinar RF (Raman)"):
            try:
                data = supabase.table("raman_spectra").select("wavenumber_cm1,intensity_a").execute().data
                df = pd.DataFrame(data)
                X = df[["wavenumber_cm1"]].values; y = df["intensity_a"].values
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
                rf = RandomForestRegressor(n_estimators=200, random_state=42); rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                st.success("Modelo Raman treinado")
                st.write(f"R² {r2_score(y_test,y_pred):.3f} • MAE {mean_absolute_error(y_test,y_pred):.3f} • RMSE {np.sqrt(mean_squared_error(y_test,y_pred)):.3f}")
            except Exception as e:
                st.error(f"Erro: {e}")
    with cols[1]:
        if st.button("Treinar RF (4 Pontas)"):
            try:
                data = supabase.table("four_point_probe_points").select("current_a,voltage_v,resistance_ohm").execute().data
                df = pd.DataFrame(data)
                X = df[["current_a","voltage_v"]].fillna(0).values; y = df["resistance_ohm"].values
                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
                rf.fit(X_train,y_train); y_pred = rf.predict(X_test)
                st.success("Modelo 4P treinado"); st.write(f"R² {r2_score(y_test,y_pred):.3f}")
            except Exception as e:
                st.error(f"Erro: {e}")
    with cols[2]:
        if st.button("Treinar RF (Tensiometria)"):
            try:
                data = supabase.table("contact_angle_points").select("t_seconds,angle_mean_deg").execute().data
                df = pd.DataFrame(data)
                X = df[["t_seconds"]].values; y = df["angle_mean_deg"].values
                rf = RandomForestRegressor(n_estimators=200, random_state=42)
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
                rf.fit(X_train,y_train); y_pred=rf.predict(X_test)
                st.success("Modelo tensiometria treinado"); st.write(f"R² {r2_score(y_test,y_pred):.3f}")
            except Exception as e:
                st.error(f"Erro: {e}")
    close_card()

# ---------- Render side panel globally if open ----------
if st.session_state.side_open:
    left, right = st.columns([3,1])
    with left:
        st.markdown("")  # keep space
    with right:
        st.markdown("<div class='side-panel'>", unsafe_allow_html=True)
        st.markdown(f"### {st.session_state.side_title}")
        if st.session_state.side_df is None or st.session_state.side_df.empty:
            st.write("_Nenhuma tabela disponível_")
        else:
            if AGGRID_AVAILABLE:
                show_aggrid(st.session_state.side_df, height=420)
            else:
                st.dataframe(st.session_state.side_df, height=420)
            csv = st.session_state.side_df.to_csv(index=False).encode('utf-8')
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                st.session_state.side_df.to_excel(writer, index=False, sheet_name="table")
            buf.seek(0)
            st.download_button("Exportar CSV", csv, file_name=f"{st.session_state.side_title}.csv", mime="text/csv")
            st.download_button("Exportar Excel", buf, file_name=f"{st.session_state.side_title}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if st.button("Fechar painel"):
            close_side()
        st.markdown("</div>", unsafe_allow_html=True)
