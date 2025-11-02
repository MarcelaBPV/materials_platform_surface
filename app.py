# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Importa módulos de processamento
from raman_processing import process_raman
from resistivity_processing import process_resistivity
from contact_angle_processing import process_contact_angle

# -------------------------------------------------------
# CONFIGURAÇÃO GERAL
# -------------------------------------------------------
st.set_page_config(page_title="Plataforma de Caracterização", layout="wide")
st.title("*Plataforma de Caracterização de Superfícies*")

# -------------------------------------------------------
# CONEXÃO SUPABASE
# -------------------------------------------------------
@st.cache_resource
def init_connection() -> Client:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        client = create_client(url, key)
        return client
    except Exception as e:
        st.sidebar.error(f"Erro ao inicializar conexão: {e}")
        st.stop()

supabase = init_connection()

# Teste de conexão
try:
    res = supabase.table("samples").select("*").limit(3).execute()
    st.sidebar.success(f"Conectado ao Supabase ({len(res.data)} amostras encontradas)")
except Exception as e:
    st.sidebar.error(f"Erro ao conectar Supabase: {e}")
    st.stop()

# -------------------------------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------------------------------
@st.cache_data(ttl=300)
def load_samples():
    try:
        data = supabase.table("samples").select("id, sample_name, description, category_id, created_at").execute()
        if not data.data:
            return pd.DataFrame(columns=["id", "sample_name", "description", "category_id", "created_at"])
        return pd.DataFrame(data.data)
    except Exception as e:
        st.error(f"Erro ao carregar amostras: {e}")
        return pd.DataFrame(columns=["id", "sample_name", "description", "category_id", "created_at"])

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
        st.error(f"Erro ao inserir dados em {table}: {e}")
        return 0

# -------------------------------------------------------
# INTERFACE PRINCIPAL
# -------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["1️⃣ Amostras", "2️⃣ Ensaios", "3️⃣ Otimização (IA)"])

# -------------------------------------------------------
# ABA 1 - CADASTRO DE AMOSTRAS
# -------------------------------------------------------
with tab1:
    st.header("📋 Gestão de Amostras")

    df_samples = load_samples()
    st.subheader("Upload de arquivo CSV")
    uploaded = st.file_uploader("Selecione CSV com colunas 'sample_name' e 'description'", type="csv")

    if uploaded:
        df_new = pd.read_csv(uploaded)
        st.dataframe(df_new.head())
        if st.button("Cadastrar amostras"):
            inserted = 0
            for _, row in df_new.iterrows():
                payload = {"sample_name": row["sample_name"]}
                if "description" in row and not pd.isna(row["description"]):
                    payload["description"] = str(row["description"])
                try:
                    supabase.table("samples").insert(payload).execute()
                    inserted += 1
                except Exception as e:
                    st.error(f"Erro ao inserir amostra: {e}")
            st.success(f"{inserted} amostras cadastradas com sucesso.")
            df_samples = load_samples()

    if df_samples.empty:
        st.info("Nenhuma amostra encontrada.")
    else:
        st.dataframe(df_samples)

# -------------------------------------------------------
# ABA 2 - ENSAIOS EXPERIMENTAIS
# -------------------------------------------------------
with tab2:
    st.header("🧪 Importar e Processar Ensaios")

    sample_name = st.text_input("Nome da amostra:")
    if not sample_name:
        st.info("Digite o nome da amostra para continuar.")
        st.stop()

    tipo = st.radio("Tipo de experimento", ["Raman", "4 Pontas", "Ângulo de Contato"])
    uploaded_file = st.file_uploader("Carregar arquivo de dados", type=["csv", "xls", "xlsx", "txt", "log"])

    if uploaded_file:
        existing = supabase.table("samples").select("*").eq("sample_name", sample_name).execute().data
        sample_id = existing[0]["id"] if existing else supabase.table("samples").insert({"sample_name": sample_name}).execute().data[0]["id"]

        try:
            if tipo == "Raman":
                result = process_raman(uploaded_file)
                df = result["df"].dropna()
                mid = get_measurement_id(sample_id, "raman")
                rows = [{"measurement_id": mid, "wavenumber_cm1": float(r["wavenumber_cm1"]), "intensity_a": float(r["intensity_a"])} for _, r in df.iterrows()]
                insert_rows("raman_spectra", rows)
                st.success(f"{len(rows)} pontos Raman importados.")
                st.pyplot(result["figure"])

            elif tipo == "4 Pontas":
                result = process_resistivity(uploaded_file)
                df = result["df"].dropna()
                mid = get_measurement_id(sample_id, "4_pontas")
                rows = [{"measurement_id": mid, "current_a": float(r["current_a"]), "voltage_v": float(r["voltage_v"])} for _, r in df.iterrows()]
                insert_rows("four_point_probe_points", rows)
                st.success(f"{len(rows)} pontos 4 Pontas importados.")
                st.pyplot(result["figure"])

            else:
                result = process_contact_angle(uploaded_file)
                df = result["df"].dropna()
                mid = get_measurement_id(sample_id, "tensiometria")
                rows = [{"measurement_id": mid, "t_seconds": float(r["t_seconds"]), "angle_mean_deg": float(r["angle_mean_deg"])} for _, r in df.iterrows()]
                insert_rows("contact_angle_points", rows)
                st.success(f"{len(rows)} pontos de Ângulo de Contato importados.")
                st.pyplot(result["figure"])

        except Exception as e:
            st.error(f"❌ Erro ao processar: {e}")

# -------------------------------------------------------
# ABA 3 - OTIMIZAÇÃO (IA)
# -------------------------------------------------------
with tab3:
    st.header("🤖 Otimização com Machine Learning (Random Forest)")
    st.markdown("Esta seção aplica um modelo de aprendizado de máquina para prever intensidades Raman e identificar regiões críticas.")

    try:
        # 1️⃣ Carregar dados Raman do banco
        raman_data = supabase.table("raman_spectra").select("wavenumber_cm1, intensity_a").execute()
        if not raman_data.data:
            st.warning("Nenhum dado Raman disponível no banco de dados.")
        else:
            df = pd.DataFrame(raman_data.data).dropna()

            # 2️⃣ Separar variáveis
            X = df[["wavenumber_cm1"]].values
            y = df["intensity_a"].values

            # 3️⃣ Divisão treino/teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 4️⃣ Modelo Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # 5️⃣ Métricas
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.success("✅ Modelo treinado com sucesso!")
            st.write(f"**R²:** {r2:.3f}")
            st.write(f"**MAE:** {mae:.3f}")
            st.write(f"**RMSE:** {rmse:.3f}")

            # 6️⃣ Visualização
            fig, ax = plt.subplots()
            ax.scatter(X_test, y_test, label="Real", s=20)
            ax.scatter(X_test, y_pred, label="Previsto", alpha=0.6)
            ax.set_xlabel("Número de onda (cm⁻¹)")
            ax.set_ylabel("Intensidade (a.u.)")
            ax.set_title("Random Forest - Intensidade Real vs Prevista")
            ax.legend()
            st.pyplot(fig)

            # 7️⃣ Previsão customizada
            st.subheader("🔍 Fazer previsão manual")
            wnum = st.number_input("Digite um número de onda (cm⁻¹):", min_value=0.0, step=10.0)
            if st.button("Prever intensidade"):
                pred = model.predict(np.array([[wnum]]))[0]
                st.info(f"Intensidade prevista: **{pred:.2f} a.u.**")

    except Exception as e:
        st.error(f"Erro ao executar otimização: {e}")
