# -*- coding: utf-8 -*-
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
    res = supabase.table("samples").select("id").limit(3).execute()
    st.sidebar.success("✅ Conectado ao Supabase!")
except Exception as e:
    st.sidebar.error(f"Erro ao conectar Supabase: {e}")
    st.stop()

# -------------------------------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------------------------------

@st.cache_data(ttl=300)
def load_samples():
    """Carrega todas as amostras existentes"""
    data = supabase.table("samples").select("id, sample_name, description, category_id, created_at").execute()
    if not data.data:
        return pd.DataFrame(columns=["id", "sample_name", "description", "category_id", "created_at"])
    return pd.DataFrame(data.data)

def insert_sample(name: str, desc: str = None, category_id=None):
    """Insere nova amostra na tabela samples"""
    payload = {"sample_name": str(name).strip()}
    if desc and not pd.isna(desc):
        payload["description"] = str(desc)
    if category_id is not None and not pd.isna(category_id):
        try:
            payload["category_id"] = int(category_id)
        except Exception:
            pass
    return supabase.table("samples").insert(payload).execute()

def get_or_create_measurement(sample_id: int, exp_type: str):
    """Retorna o measurement_id da amostra e tipo; cria se não existir"""
    meas = supabase.table("measurements").select("id").eq("sample_id", sample_id).eq("type", exp_type).limit(1).execute()
    if meas.data:
        return meas.data[0]["id"]
    new_meas = supabase.table("measurements").insert({"sample_id": sample_id, "type": exp_type}).execute()
    return new_meas.data[0]["id"]

def insert_rows(table: str, rows: list):
    """Insere várias linhas em uma tabela"""
    if not rows:
        return 0
    supabase.table(table).insert(rows).execute()
    return len(rows)

def read_samples_file(uploaded):
    """Lê arquivo de amostras (.csv, .xlsx, .txt)"""
    name = uploaded.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
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

    st.subheader("📂 Importar lista de amostras (.csv, .xlsx, .txt)")
    uploaded = st.file_uploader("Selecione o arquivo de amostras", type=["csv", "xls", "xlsx", "txt"])

    if uploaded:
        try:
            df_new = read_samples_file(uploaded)
            if any(col in df_new.columns for col in ["wavenumber_cm1", "intensity_a"]):
                st.error("⚠️ Este arquivo contém dados de ensaio. Use a **aba 2 – Ensaios**.")
                st.stop()
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
    st.subheader("📋 Amostras existentes")
    if df_samples.empty:
        st.info("Nenhuma amostra encontrada.")
    else:
        st.dataframe(df_samples)

# -------------------------------------------------------
# ABA 2 - ENSAIOS
# -------------------------------------------------------

# -------------------------------------------------------
# ABA 2 - ENSAIOS (atualizada com seletor de amostras)
# -------------------------------------------------------

with tab2:
    st.header("🧪 Processamento de Ensaios")
    st.markdown("Selecione uma amostra existente ou cadastre uma nova para associar o ensaio.")

    # Carregar amostras existentes
    df_samples = load_samples()

    if not df_samples.empty:
        sample_options = ["➕ Cadastrar nova amostra"] + df_samples["sample_name"].dropna().tolist()
        selected_sample = st.selectbox("🔖 Escolha a amostra:", sample_options)

        if selected_sample == "➕ Cadastrar nova amostra":
            new_name = st.text_input("Nome da nova amostra")
            new_desc = st.text_area("Descrição (opcional)")
            if st.button("Salvar nova amostra"):
                if new_name:
                    insert_sample(new_name, new_desc)
                    st.success(f"Amostra '{new_name}' criada com sucesso!")
                    st.rerun()
                else:
                    st.warning("Informe um nome para a amostra.")
            st.stop()
        else:
            sample_name = selected_sample
            sample_id = int(df_samples[df_samples["sample_name"] == sample_name]["id"].values[0])
            st.info(f"Amostra selecionada: **{sample_name}** (ID={sample_id})")
    else:
        st.warning("Nenhuma amostra cadastrada ainda. Vá até a aba **1️⃣ Amostras** para adicionar.")
        st.stop()

    # Tipo de ensaio
    tipo = st.radio("Tipo de ensaio:", ["Raman", "4 Pontas", "Ângulo de Contato"])
    uploaded_file = st.file_uploader("📂 Carregar arquivo do ensaio", type=["txt", "csv", "xls", "xlsx"])

    if uploaded_file:
        try:
            # --- ENSAIO RAMAN ---
            if tipo == "Raman":
                name = uploaded_file.name.lower()
                if name.endswith(".xlsx") or name.endswith(".xls"):
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
                df = df[df["intensity_a"] >= 0]

                s = rc2.spectrum.from_array(df["wavenumber_cm1"].values, df["intensity_a"].values)
                s_corr = s.remove_baseline().smooth().normalize()
                peaks = s_corr.find_peaks(threshold_rel=0.05)

                st.subheader("📈 Espectro Raman Normalizado")
                fig, ax = plt.subplots()
                s_corr.plot(ax=ax)
                peaks.plot(ax=ax, marker="o", color="r")
                ax.set_xlabel("Número de onda (cm⁻¹)")
                ax.set_ylabel("Intensidade (a.u.)")
                ax.invert_xaxis()
                st.pyplot(fig)

                mid = get_or_create_measurement(sample_id, "raman")
                rows = [{"measurement_id": mid, "wavenumber_cm1": float(x), "intensity_a": float(y)} for x, y in zip(s_corr.x, s_corr.y)]
                insert_rows("raman_spectra", rows)
                st.success(f"{len(rows)} pontos Raman vinculados à amostra '{sample_name}'.")

            # --- ENSAIO 4 PONTAS ---
            elif tipo == "4 Pontas":
                df = pd.read_csv(uploaded_file)
                df.columns = [c.lower().strip() for c in df.columns]
                if "current_a" not in df.columns:
                    df.rename(columns={"corrente": "current_a", "i": "current_a"}, inplace=True)
                if "voltage_v" not in df.columns:
                    df.rename(columns={"tensao": "voltage_v", "v": "voltage_v"}, inplace=True)

                df = df.dropna(subset=["current_a", "voltage_v"])
                df["resistance_ohm"] = df["voltage_v"] / df["current_a"]

                fig, ax = plt.subplots()
                ax.scatter(df["current_a"], df["voltage_v"], label="Dados")
                ax.plot(df["current_a"], df["current_a"] * df["resistance_ohm"].mean(), "r-", label="Ajuste médio")
                ax.set_xlabel("Corrente (A)")
                ax.set_ylabel("Tensão (V)")
                ax.legend()
                st.pyplot(fig)

                mid = get_or_create_measurement(sample_id, "4_pontas")
                rows = df.to_dict(orient="records")
                for r in rows:
                    r["measurement_id"] = mid
                insert_rows("four_point_probe_points", rows)
                st.success(f"{len(rows)} pontos 4 Pontas inseridos.")

            # --- ENSAIO ÂNGULO DE CONTATO ---
            elif tipo == "Ângulo de Contato":
                df = pd.read_csv(uploaded_file, sep=None, engine="python")
                df.columns = [c.lower().strip() for c in df.columns]
                if "mean" in df.columns:
                    df.rename(columns={"mean": "angle_mean_deg"}, inplace=True)
                if "time" in df.columns:
                    df.rename(columns={"time": "t_seconds"}, inplace=True)

                df = df.dropna(subset=["t_seconds", "angle_mean_deg"])

                fig, ax = plt.subplots()
                ax.plot(df["t_seconds"], df["angle_mean_deg"], "bo-", label="Ângulo")
                ax.set_xlabel("Tempo (s)")
                ax.set_ylabel("Ângulo (°)")
                ax.legend()
                st.pyplot(fig)

                mid = get_or_create_measurement(sample_id, "tensiometria")
                rows = df.to_dict(orient="records")
                for r in rows:
                    r["measurement_id"] = mid
                insert_rows("contact_angle_points", rows)
                st.success(f"{len(rows)} pontos de ângulo inseridos.")

        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")


# -------------------------------------------------------
# ABA 3 - OTIMIZAÇÃO (IA)
# -------------------------------------------------------

with tab3:
    st.header("🤖 Otimização via Machine Learning (Random Forest)")
    st.markdown("Treina modelo IA apenas para dados **Raman** armazenados no Supabase.")

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
