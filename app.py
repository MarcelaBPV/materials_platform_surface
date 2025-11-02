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
import pickle, io

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

def insert_sample(name, desc=None, category_id=None):
    payload = {"sample_name": str(name).strip()}
    if desc and not pd.isna(desc):
        payload["description"] = str(desc)
    if category_id is not None and not pd.isna(category_id):
        try:
            payload["category_id"] = int(category_id)
        except Exception:
            pass
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

def read_samples_file(uploaded) -> pd.DataFrame:
    """
    Lê arquivo de amostras em .csv, .xlsx/.xls ou .txt (tab/;/, autodetect),
    padroniza colunas para: sample_name, description, category_id
    """
    name = uploaded.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded)
    elif name.endswith(".csv"):
        # autodetect de separador (',' ou ';')
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            df = pd.read_csv(uploaded, sep=";")
    elif name.endswith(".txt"):
        # tenta autodetectar (tab, vírgula, ponto e vírgula)
        try:
            df = pd.read_csv(uploaded, sep=None, engine="python")
        except Exception:
            # fallback comum de arquivos exportados por instrumentos (tab)
            df = pd.read_csv(uploaded, sep="\t", header=0)
    else:
        raise ValueError("Formato não suportado. Use .csv, .xlsx/.xls ou .txt")

    # padroniza nomes de colunas
    df.columns = [c.lower().strip() for c in df.columns]

    # mapeamentos comuns
    colmap = {}
    if "nome" in df.columns and "sample_name" not in df.columns:
        colmap["nome"] = "sample_name"
    if "descricao" in df.columns and "description" not in df.columns:
        colmap["descricao"] = "description"
    if "categoria" in df.columns and "category_id" not in df.columns:
        colmap["categoria"] = "category_id"
    df.rename(columns=colmap, inplace=True)

    # exige sample_name
    if "sample_name" not in df.columns:
        raise ValueError("O arquivo precisa ter a coluna 'sample_name'.")

    # garante colunas opcionais
    if "description" not in df.columns:
        df["description"] = None
    if "category_id" not in df.columns:
        df["category_id"] = None

    # limpa linhas vazias
    df = df[df["sample_name"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)
    return df[["sample_name", "description", "category_id"]]

# -------------------------------------------------------
# ABAS
# -------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["1️⃣ Amostras", "2️⃣ Ensaios", "3️⃣ Otimização (IA)"])

# -------------------------------------------------------
# ABA 1 - AMOSTRAS
# -------------------------------------------------------
with tab1:
    st.header("📋 Cadastro e Visualização de Amostras")

    df_samples = load_samples()

    # --- Cadastro manual ---
    st.subheader("Cadastrar manualmente")
    new_name = st.text_input("Nome da amostra")
    new_desc = st.text_area("Descrição (opcional)")
    if st.button("Cadastrar amostra individual"):
        if new_name:
            insert_sample(new_name, new_desc)
            st.success(f"Amostra '{new_name}' cadastrada com sucesso!")
            df_samples = load_samples()
        else:
            st.warning("Informe um nome de amostra.")

    st.divider()

    # --- Cadastro por arquivo (agora aceita csv/xls/xlsx/txt) ---
    st.subheader("📂 Importar lista de amostras (.csv, .xls/.xlsx ou .txt)")
    uploaded = st.file_uploader("Selecione o arquivo de amostras", type=["csv", "xls", "xlsx", "txt"])

    if uploaded:
        try:
            df_new = read_samples_file(uploaded)
            st.write("Pré-visualização:")
            st.dataframe(df_new.head())

            if st.button("Cadastrar amostras em lote"):
                inserted = 0
                for _, row in df_new.iterrows():
                    try:
                        insert_sample(row["sample_name"], row.get("description"), row.get("category_id"))
                        inserted += 1
                    except Exception as e:
                        st.warning(f"Erro ao inserir '{row['sample_name']}': {e}")

                st.success(f"✅ {inserted} amostras cadastradas com sucesso!")
                df_samples = load_samples()

        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")

    st.divider()

    # --- Exibir amostras cadastradas ---
    st.subheader("📋 Amostras existentes")
    if df_samples.empty:
        st.info("Nenhuma amostra encontrada.")
    else:
        st.dataframe(df_samples)

# -------------------------------------------------------
# ABA 2 - ENSAIOS (RAMAN, 4 PONTAS, ÂNGULO)
# -------------------------------------------------------
with tab2:
    st.header("🧪 Processamento de Ensaios")

    df_samples = load_samples()
    sample_options = df_samples["sample_name"].tolist()
    selected_sample = st.selectbox("Escolha a amostra", ["-- Selecione --"] + sample_options)

    if selected_sample != "-- Selecione --":
        sample_id = df_samples.loc[df_samples["sample_name"] == selected_sample, "id"].values[0]
        tipo = st.radio("Tipo de ensaio:", ["Raman", "4 Pontas", "Ângulo de Contato"])

        uploaded_file = st.file_uploader("Carregar arquivo do ensaio", type=["txt", "csv", "xls", "xlsx"])
        if uploaded_file:
            try:
                if tipo == "Raman":
                    # --- Processamento Raman ---
                    filename = uploaded_file.name.lower()
                    if filename.endswith(".xls") or filename.endswith(".xlsx"):
                        df = pd.read_excel(uploaded_file)
                    elif filename.endswith(".csv"):
                        try:
                            df = pd.read_csv(uploaded_file)
                        except Exception:
                            df = pd.read_csv(uploaded_file, sep=";")
                    elif filename.endswith(".txt"):
                        try:
                            df = pd.read_csv(uploaded_file, sep=None, engine="python")
                        except Exception:
                            df = pd.read_csv(uploaded_file, sep="\t", header=0)
                    else:
                        raise ValueError("Formato não suportado.")

                    df.columns = [c.lower().strip() for c in df.columns]
                    if "wavenumber_cm1" not in df.columns and "wavenumber" in df.columns:
                        df.rename(columns={"wavenumber": "wavenumber_cm1"}, inplace=True)
                    if "intensity_a" not in df.columns and "intensity" in df.columns:
                        df.rename(columns={"intensity": "intensity_a"}, inplace=True)

                    df = df.dropna(subset=["wavenumber_cm1", "intensity_a"])
                    df = df[df["intensity_a"] >= 0]

                    s = rc2.spectrum.from_array(df["wavenumber_cm1"].values, df["intensity_a"].values)
                    s_corr = s.remove_baseline().smooth().normalize()
                    peaks = s_corr.find_peaks(threshold_rel=0.05)

                    st.subheader("📄 Dados originais")
                    st.dataframe(df.head())
                    norm_df = pd.DataFrame({"wavenumber_cm1": s_corr.x, "normalized_intensity": s_corr.y})
                    st.subheader("📈 Dados normalizados")
                    st.dataframe(norm_df.head())

                    fig, ax = plt.subplots()
                    s_corr.plot(ax=ax, label="Normalizado")
                    peaks.plot(ax=ax, marker="o", color="r", label="Picos")
                    ax.set_xlabel("Número de onda (cm⁻¹)")
                    ax.set_ylabel("Intensidade (a.u.)")
                    ax.legend()
                    ax.invert_xaxis()
                    fig.tight_layout()
                    st.pyplot(fig)

                    mid = get_measurement_id(sample_id, "raman")
                    rows = [
                        {"measurement_id": mid, "wavenumber_cm1": float(x), "intensity_a": float(y)}
                        for x, y in zip(s_corr.x, s_corr.y)
                    ]
                    insert_rows("raman_spectra", rows)
                    st.success(f"{len(rows)} pontos Raman importados e normalizados!")

                elif tipo == "4 Pontas":
                    # --- Processamento Resistividade ---
                    # Aceita csv padrão; adapte aqui se seus arquivos 4p vierem em xlsx/txt
                    df = pd.read_csv(uploaded_file)
                    df.columns = [c.lower().strip() for c in df.columns]
                    if "current_a" not in df.columns:
                        df.rename(columns={"corrente": "current_a", "i": "current_a"}, inplace=True)
                    if "voltage_v" not in df.columns:
                        df.rename(columns={"tensao": "voltage_v", "v": "voltage_v"}, inplace=True)

                    df = df.dropna(subset=["current_a", "voltage_v"])
                    df = df[(df["current_a"] > 0) & (df["voltage_v"] >= 0)]

                    df["resistance_ohm"] = df["voltage_v"] / df["current_a"]
                    R_med = df["resistance_ohm"].mean()

                    fig, ax = plt.subplots()
                    ax.scatter(df["current_a"], df["voltage_v"], label="Dados experimentais")
                    ax.plot(df["current_a"], df["current_a"] * R_med, "r-", label=f"Ajuste linear (R={R_med:.2f} Ω)")
                    ax.set_xlabel("Corrente (A)")
                    ax.set_ylabel("Tensão (V)")
                    ax.legend()
                    st.pyplot(fig)

                    mid = get_measurement_id(sample_id, "4_pontas")
                    rows = df.to_dict(orient="records")
                    for r in rows:
                        r["measurement_id"] = mid
                    insert_rows("four_point_probe_points", rows)
                    st.success(f"{len(rows)} pontos 4 Pontas cadastrados!")

                elif tipo == "Ângulo de Contato":
                    # --- Processamento Tensiometria ---
                    # Autodetect de separador para .txt também
                    try:
                        df = pd.read_csv(uploaded_file, sep=None, engine="python")
                    except Exception:
                        df = pd.read_csv(uploaded_file, delim_whitespace=True, comment="#")

                    df.columns = [c.lower().strip() for c in df.columns]
                    if "mean" in df.columns and "angle_mean_deg" not in df.columns:
                        df.rename(columns={"mean": "angle_mean_deg"}, inplace=True)
                    if "time" in df.columns and "t_seconds" not in df.columns:
                        df.rename(columns={"time": "t_seconds"}, inplace=True)

                    df = df.dropna(subset=["t_seconds", "angle_mean_deg"])
                    df = df[(df["angle_mean_deg"] >= 0) & (df["angle_mean_deg"] <= 180)]

                    fig, ax = plt.subplots()
                    ax.plot(df["t_seconds"], df["angle_mean_deg"], "bo-", label="Ângulo de contato")
                    ax.set_xlabel("Tempo (s)")
                    ax.set_ylabel("Ângulo (°)")
                    ax.legend()
                    st.pyplot(fig)

                    mid = get_measurement_id(sample_id, "tensiometria")
                    rows = df.to_dict(orient="records")
                    for r in rows:
                        r["measurement_id"] = mid
                    insert_rows("contact_angle_points", rows)
                    st.success(f"{len(rows)} pontos de ângulo de contato cadastrados!")

            except Exception as e:
                st.error(f"Erro ao processar arquivo: {e}")

# -------------------------------------------------------
# ABA 3 - OTIMIZAÇÃO (IA) - APENAS RAMAN
# -------------------------------------------------------
with tab3:
    st.header("🤖 Otimização via Machine Learning (Random Forest)")
    st.markdown("Treina um modelo IA apenas para **Raman**, aprendendo padrões e criando uma assinatura molecular.")

    model_path = "models/raman_model.pkl"

    try:
        data = supabase.table("raman_spectra").select("wavenumber_cm1, intensity_a").execute()
        if not data.data:
            st.warning("Nenhum dado Raman disponível.")
            st.stop()

        df = pd.DataFrame(data.data).dropna()
        X = df[["wavenumber_cm1"]].values
        y = df["intensity_a"].values

        # Verifica modelo salvo
        model_exists = False
        try:
            res = supabase.storage.from_("models").download("raman_model.pkl")
            model_file = io.BytesIO(res)
            model = pickle.load(model_file)
            model_exists = True
            st.success("📦 Modelo carregado do Supabase Storage!")
        except Exception:
            st.info("Nenhum modelo salvo encontrado. Será treinado um novo.")

        retrain = st.button("🔁 Re-treinar modelo IA")

        if retrain or not model_exists:
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

            buf = io.BytesIO()
            pickle.dump(model, buf)
            buf.seek(0)
            supabase.storage.from_("models").upload("raman_model.pkl", buf, {"content-type": "application/octet-stream"})
            st.info("💾 Modelo salvo no Supabase Storage!")

        # Previsão manual
        st.subheader("🔍 Prever intensidade Raman")
        wnum = st.number_input("Digite número de onda (cm⁻¹):", min_value=0.0, step=10.0)
        if st.button("Prever intensidade"):
            pred = model.predict(np.array([[wnum]]))[0]
            st.info(f"Intensidade prevista: **{pred:.2f} a.u.**")

        # Identificação molecular
        st.subheader("🔬 Picos e grupos moleculares")
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
