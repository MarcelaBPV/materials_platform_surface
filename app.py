# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client, Client
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Plataforma de Caracterização", layout="wide")
st.title("*Plataforma de Caracterização de Superfícies*")

# -----------------------
# Supabase connection via st.secrets
# -----------------------
if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
    st.error("Por favor configure SUPABASE_URL e SUPABASE_KEY em st.secrets (Streamlit Cloud Secrets).")
    st.stop()

supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)

# -----------------------
# Try to import ramanchada2 (with safe fallbacks)
# -----------------------
USE_RAMANCHADA = False
try:
    # The library might expose modules like preprocessing/harmonization — adapt if API differs
    import ramanchada2 as r2
    # try some known helpers (best-effort)
    try:
        from ramanchada2.preprocessing import baseline_als, normalize_area, smooth_sg
        from ramanchada2.harmonization import align_spectrum
    except Exception:
        # try alternative submodules
        baseline_als = getattr(r2, "baseline_als", None)
        normalize_area = getattr(r2, "normalize_area", None)
        align_spectrum = getattr(r2, "align_spectrum", None)
    USE_RAMANCHADA = True
except Exception:
    # fallback implementations
    def baseline_als(y, lam=1e6, p=0.01, niter=10):
        # simple iterative baseline (Asymmetric Least Squares) fallback (lightweight)
        L = len(y)
        w = np.ones(L)
        for i in range(niter):
            W = np.diag(w)
            # approximate using difference matrix - use simple smoothing instead to avoid heavy solvers
            z = np.poly1d(np.polyfit(np.arange(L), y * w, 3))(np.arange(L))
            w = p * (y > z) + (1 - p) * (y < z)
        return y - z

    def normalize_area(y):
        a = np.trapz(np.abs(y))
        return y / a if a != 0 else y

    def smooth_sg(y, window=7, order=2):
        from scipy.signal import savgol_filter
        win = window if window % 2 == 1 else window + 1
        try:
            return savgol_filter(y, win, order)
        except Exception:
            return pd.Series(y).rolling(window=win, center=True, min_periods=1).mean().values

    def align_spectrum(x, y, x_ref, y_ref):
        # simple alignment by interpolation to reference x_ref
        y_interp = np.interp(x_ref, x, y)
        return y_interp

# -----------------------
# Helpers for DB operations
# -----------------------
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

# -----------------------
# Page layout: tabs
# -----------------------
tab1, tab2, tab3 = st.tabs(["1️⃣ Cadastro de Amostras", "2️⃣ Técnicas de Análise", "3️⃣ Otimização / Raman IA"])

# -----------------------
# TAB 1 - Samples
# -----------------------
with tab1:
    st.header("Cadastro de Amostras")
    df_samples = load_samples()
    st.subheader("Importar CSV de amostras")
    uploaded = st.file_uploader("CSV de amostras (colunas: sample_name, description, category_id opcional)", type="csv", key="up_samples")
    if uploaded:
        df_new = pd.read_csv(uploaded)
        st.dataframe(df_new.head())
        if st.button("Cadastrar amostras no Supabase"):
            inserted = 0
            for _, row in df_new.iterrows():
                payload = {}
                if "sample_name" in row and not pd.isna(row["sample_name"]):
                    payload["sample_name"] = str(row["sample_name"])
                else:
                    continue
                if "description" in row and not pd.isna(row["description"]):
                    payload["description"] = str(row["description"])
                if "category_id" in row and not pd.isna(row["category_id"]):
                    payload["category_id"] = int(row["category_id"])
                try:
                    supabase.table("samples").insert(payload).execute()
                    inserted += 1
                except Exception as e:
                    st.error(f"Erro ao inserir amostra {payload.get('sample_name')}: {e}")
            st.success(f"{inserted} amostra(s) cadastrada(s).")
            df_samples = load_samples()

    st.subheader("Amostras existentes")
    if df_samples.empty:
        st.info("Nenhuma amostra encontrada.")
    else:
        st.dataframe(df_samples)

# -----------------------
# TAB 2 - Técnicas
# -----------------------
with tab2:
    st.header("Importar e visualizar dados experimentais")
    df_samples = load_samples()
    if df_samples.empty:
        st.warning("Cadastre ao menos uma amostra na aba 1 antes.")
    else:
        sample_choice = st.selectbox("Escolha a amostra", df_samples["id"])
        tipo = st.radio("Tipo de experimento", ["Raman", "4 Pontas", "Ângulo de Contato"])

        uploaded_file = st.file_uploader(f"Carregar arquivo para {tipo}", type=["csv", "txt", "log"], key="up_ensaios")
        if uploaded_file is not None:
            # detectar e importar dependendo do tipo
            try:
                if tipo == "Raman":
                    # tentamos ler formatos comuns: csv com ',' ou '\t'
                    try:
                        df = pd.read_csv(uploaded_file)
                        # try to detect columns
                        if set(["wavenumber_cm1", "intensity_a"]).issubset(df.columns):
                            pass
                        else:
                            # try tab separated with two cols
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, sep="\t", names=["wavenumber_cm1", "intensity_a"], comment="#")
                    except Exception:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep="\t", names=["wavenumber_cm1", "intensity_a"], comment="#")
                    df = df.dropna().reset_index(drop=True)
                    rows = []
                    mid = get_measurement_id(sample_choice, "raman")
                    for _, r in df.iterrows():
                        rows.append({"measurement_id": mid,
                                     "wavenumber_cm1": float(r["wavenumber_cm1"]),
                                     "intensity_a": float(r["intensity_a"])})
                    cnt = insert_rows("raman_spectra", rows)
                    st.success(f"Importados {cnt} pontos Raman.")
                    st.dataframe(df.head())

                elif tipo == "4 Pontas":
                    df = pd.read_csv(uploaded_file)
                    df = df.dropna().reset_index(drop=True)
                    mid = get_measurement_id(sample_choice, "4_pontas")
                    rows = []
                    for _, r in df.iterrows():
                        rows.append({"measurement_id": mid,
                                     "current_a": float(r.get("current_a", r.get("I", r.get("Current", np.nan)))),
                                     "voltage_v": float(r.get("voltage_v", r.get("V", r.get("Voltage", np.nan))))})
                    cnt = insert_rows("four_point_probe_points", rows)
                    st.success(f"Importados {cnt} pontos 4-pontas.")
                    st.dataframe(df.head())

                else:  # Ângulo de Contato
                    # Many .LOG formats are column separated — we'll do a robust parse
                    try:
                        df = pd.read_csv(uploaded_file, delim_whitespace=True)
                    except Exception:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep="\t", comment="#", engine="python")
                    # try common column names
                    if "Mean" in df.columns and "Time" in df.columns:
                        df = df.rename(columns={"Time": "t_seconds", "Mean": "angle_mean_deg"})
                    # if file from your example, it has columns like: Time Theta(L) Theta(R) Mean Dev Height Width ...
                    expected = ["Time", "Theta(L)", "Theta(R)", "Mean", "Dev", "Height", "Width"]
                    if set(expected).issubset(df.columns):
                        df2 = pd.DataFrame({
                            "t_seconds": df["Time"].astype(float),
                            "theta_l": df["Theta(L)"].astype(float),
                            "theta_r": df["Theta(R)"].astype(float),
                            "angle_mean_deg": df["Mean"].astype(float),
                            "dev": df["Dev"].astype(float),
                            "height_mm": df["Height"].astype(float),
                            "width_mm": df["Width"].astype(float),
                        })
                    else:
                        # try to map by name heuristics
                        df2 = df.rename(columns={
                            "t_seconds": "t_seconds",
                            "theta_l": "theta_l", "theta_r": "theta_r",
                            "angle_mean_deg": "angle_mean_deg", "dev": "dev",
                            "Height": "height_mm", "Width": "width_mm"
                        })
                        df2 = df2[[c for c in ["t_seconds", "theta_l", "theta_r", "angle_mean_deg", "dev", "height_mm", "width_mm"] if c in df2.columns]]
                    mid = get_measurement_id(sample_choice, "tensiometria")
                    rows = []
                    for _, r in df2.iterrows():
                        payload = {"measurement_id": mid}
                        for c in ["t_seconds","theta_l","theta_r","angle_mean_deg","dev","height_mm","width_mm"]:
                            if c in r and not pd.isna(r[c]):
                                payload[c] = float(r[c])
                        rows.append(payload)
                    cnt = insert_rows("contact_angle_points", rows)
                    st.success(f"Importados {cnt} pontos de tensiometria.")
                    st.dataframe(df2.head())

            except Exception as e:
                st.error(f"Erro ao importar arquivo: {e}")

        # Visualizar dados já cadastrados para a amostra
        st.markdown("---")
        st.subheader("Visualizar dados já cadastrados")
        try:
            measurements = supabase.table("measurements").select("*").eq("sample_id", sample_choice).execute().data or []
        except Exception as e:
            st.error(f"Erro ao buscar measurements: {e}")
            measurements = []

        if measurements:
            # pick the measurement ids of the selected type
            sel_type_map = {"Raman": "raman", "4 Pontas": "4_pontas", "Ângulo de Contato": "tensiometria"}
            selected_type_key = sel_type_map[tipo]
            measurement_ids = [m["id"] for m in measurements if m["type"] == selected_type_key]
            data = []
            table_map = {"Raman": "raman_spectra", "4 Pontas": "four_point_probe_points", "Ângulo de Contato": "contact_angle_points"}
            table_name = table_map[tipo]
            for mid in measurement_ids:
                resp = supabase.table(table_name).select("*").eq("measurement_id", mid).execute()
                if resp and resp.data:
                    data.extend(resp.data)
            df_existing = pd.DataFrame(data) if data else pd.DataFrame()
            if df_existing.empty:
                st.info("Nenhum dado encontrado para o tipo selecionado nesta amostra.")
            else:
                st.dataframe(df_existing.head(200))
                fig, ax = plt.subplots()
                if tipo == "Raman":
                    ax.plot(df_existing["wavenumber_cm1"], df_existing["intensity_a"])
                    ax.set_xlabel("Wavenumber (cm⁻¹)")
                    ax.set_ylabel("Intensity (a.u.)")
                    ax.invert_xaxis()
                elif tipo == "4 Pontas":
                    ax.plot(df_existing["current_a"], df_existing["voltage_v"], "o-")
                    ax.set_xlabel("Current (A)")
                    ax.set_ylabel("Voltage (V)")
                else:
                    ax.plot(df_existing["t_seconds"], df_existing["angle_mean_deg"], "o-")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Angle (deg)")
                st.pyplot(fig)

# -----------------------
# TAB 3 - Otimização / Raman IA
# -----------------------
with tab3:
    st.header("Otimização e Análise Raman (pré-processamento + picos + similaridade)")
    df_samples = load_samples()
    if df_samples.empty:
        st.warning("Cadastre amostras primeiro.")
    else:
        sample_choice = st.selectbox("Selecione amostra", df_samples["id"], key="opt_sample")
        # load all raman measurements for that sample
        try:
            measurements = supabase.table("measurements").select("*").eq("sample_id", sample_choice).eq("type", "raman").execute().data or []
        except Exception as e:
            st.error(f"Erro: {e}")
            measurements = []

        if not measurements:
            st.warning("Nenhuma medição Raman disponível para essa amostra.")
        else:
            # aggregate all spectra points by measurement (we'll assume one measurement per sample in many cases)
            all_data = []
            for m in measurements:
                resp = supabase.table("raman_spectra").select("*").eq("measurement_id", m["id"]).execute()
                if resp and resp.data:
                    df_m = pd.DataFrame(resp.data)
                    df_m = df_m.sort_values("wavenumber_cm1").reset_index(drop=True)
                    all_data.append((m["id"], df_m))

            # choose which measurement to analyze
            meas_id_list = [m[0] for m in all_data]
            meas_choice = st.selectbox("Selecione measurement_id para análise", meas_id_list)
            df = None
            for mid, d in all_data:
                if mid == meas_choice:
                    df = d.copy()
                    break

            if df is None or df.empty:
                st.warning("Dados vazios para o measurement selecionado.")
            else:
                # Preprocessing
                x = df["wavenumber_cm1"].values
                y = df["intensity_a"].values

                st.subheader("Pré-processamento")
                st.write(f"Usando ramanchada2? {'Sim' if USE_RAMANCHADA else 'Não (fallback)'}")

                # smoothing
                window = st.slider("Tamanho da janela de suavização (S-G)", min_value=5, max_value=31, value=9, step=2)
                y_smooth = smooth_sg(y, window=window) if not USE_RAMANCHADA else (locals().get("smooth_sg")(y, window=window) if 'smooth_sg' in locals() else y)

                # baseline
                use_baseline = st.checkbox("Remover baseline (ALS)", value=True)
                if use_baseline:
                    y_baseline = baseline_als(y_smooth)
                else:
                    y_baseline = y_smooth

                # normalize
                y_norm = normalize_area(y_baseline) if (USE_RAMANCHADA or 'normalize_area' in globals()) else normalize_area(y_baseline)

                fig, ax = plt.subplots()
                ax.plot(x, y, label="Original", alpha=0.4)
                ax.plot(x, y_norm, label="Pré-processado")
                ax.invert_xaxis()
                ax.set_xlabel("Wavenumber (cm⁻¹)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.legend()
                st.pyplot(fig)

                # Peak detection
                st.subheader("Detecção de picos")
                height_factor = st.slider("Fator de altura mínima (multiplicado pela média)", 0.1, 5.0, 1.0)
                prominence = st.slider("Prominência mínima", 0.0, float(np.max(y_norm)), float(np.mean(y_norm)/2))
                thresh = np.mean(y_norm) * height_factor
                peaks, props = find_peaks(y_norm, height=thresh, prominence=prominence)
                st.write("Picos detectados (wavenumbers):", x[peaks].tolist())
                if peaks.size:
                    fig2, ax2 = plt.subplots()
                    ax2.plot(x, y_norm, label="Pré-processado")
                    ax2.scatter(x[peaks], y_norm[peaks], c="red")
                    ax2.invert_xaxis()
                    ax2.set_xlabel("Wavenumber (cm⁻¹)")
                    ax2.set_ylabel("Intensity (a.u.)")
                    st.pyplot(fig2)

                # Similaridade espectral: comparar com outras espectras da mesma amostra ou com referências
                st.subheader("Similaridade espectral")
                # build reference spectra set (other measurements of same sample or all in DB)
                mode = st.radio("Comparar com", ["Outras medições desta amostra", "Todas as medições no DB"], index=0)
                ref_spectra = []
                if mode.startswith("Outras"):
                    for mid, d in all_data:
                        if mid != meas_choice:
                            xr = d["wavenumber_cm1"].values
                            yr = d["intensity_a"].values
                            # interpolate reference to current x
                            yr_i = np.interp(x, xr, yr)
                            # preprocess same pipeline
                            yr_s = smooth_sg(yr_i, window=window)
                            if use_baseline:
                                yr_b = baseline_als(yr_s)
                            else:
                                yr_b = yr_s
                            yr_n = normalize_area(yr_b)
                            ref_spectra.append((mid, yr_n))
                else:
                    # fetch some sample of other spectra in DB (limit to avoid timeouts)
                    resp_all = supabase.table("raman_spectra").select("*").limit(5000).execute()
                    all_rows = resp_all.data if resp_all and resp_all.data else []
                    if all_rows:
                        df_all = pd.DataFrame(all_rows)
                        # group by measurement_id
                        for mid, group in df_all.groupby("measurement_id"):
                            grp = group.sort_values("wavenumber_cm1")
                            xr = grp["wavenumber_cm1"].values
                            yr = grp["intensity_a"].values
                            yr_i = np.interp(x, xr, yr)
                            yr_s = smooth_sg(yr_i, window=window)
                            if use_baseline:
                                yr_b = baseline_als(yr_s)
                            else:
                                yr_b = yr_s
                            yr_n = normalize_area(yr_b)
                            ref_spectra.append((mid, yr_n))
                # compute similarity (cosine)
                if not ref_spectra:
                    st.info("Nenhuma referência encontrada para comparar.")
                else:
                    # prepare target vector
                    vec_target = y_norm.reshape(1, -1)
                    # scaling optional
                    scaler = StandardScaler()
                    try:
                        # stack for scaling to make similarity robust
                        stack = np.vstack([vec_target] + [v.reshape(1, -1) for _, v in ref_spectra])
                        stack_s = scaler.fit_transform(stack.T).T
                        target_s = stack_s[0].reshape(1, -1)
                        sims = []
                        for i, (mid, v) in enumerate(ref_spectra):
                            v_s = stack_s[i+1].reshape(1, -1)
                            sim = float(cosine_similarity(target_s, v_s)[0,0])
                            sims.append((mid, sim))
                    except Exception:
                        # fallback: cosine on raw vectors
                        sims = []
                        for mid, v in ref_spectra:
                            try:
                                sim = float(cosine_similarity(vec_target, v.reshape(1, -1))[0,0])
                            except Exception:
                                sim = 0.0
                            sims.append((mid, sim))

                    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
                    st.write("Top referências por similaridade (measurement_id, score):")
                    st.write(sims_sorted[:10])
                    # plot the best match
                    best_mid, best_score = sims_sorted[0]
                    st.write(f"Melhor correspondência: measurement_id={best_mid} (score={best_score:.4f})")
                    # find best spectrum data to plot
                    best_spec = None
                    for mid, v in ref_spectra:
                        if mid == best_mid:
                            best_spec = v
                            break
                    if best_spec is not None:
                        fig3, ax3 = plt.subplots()
                        ax3.plot(x, y_norm, label="Target")
                        ax3.plot(x, best_spec, label=f"Best match {best_mid}")
                        ax3.invert_xaxis()
                        ax3.legend()
                        st.pyplot(fig3)

                # Option: export peaks as CSV
                if peaks.size:
                    df_peaks = pd.DataFrame({
                        "wavenumber_cm1": x[peaks],
                        "intensity": y_norm[peaks],
                        "measurement_id": meas_choice
                    })
                    st.download_button("Baixar picos (CSV)", df_peaks.to_csv(index=False), file_name=f"peaks_meas_{meas_choice}.csv")
