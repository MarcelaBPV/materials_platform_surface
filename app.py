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
        from
