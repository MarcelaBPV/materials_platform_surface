import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ramanchada2 as rc2

def process_raman(file):
    """
    Lê arquivo .xls contendo duas colunas: wavenumber_cm1 e intensity_a
    Aplica baseline, suavização, normalização e detecção de picos via ramanchada2.
    """
    # Ler Excel
    df = pd.read_excel(file)
    # Tentativa de mapear colunas mesmo que o nome varie
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    if "wavenumber_cm1" not in df.columns and "wavenumber" in df.columns:
        df = df.rename(columns={"wavenumber": "wavenumber_cm1"})
    if "intensity_a" not in df.columns and "intensity" in df.columns:
        df = df.rename(columns={"intensity": "intensity_a"})

    s = rc2.spectrum.from_array(df["wavenumber_cm1"].values, df["intensity_a"].values)
    s_corr = s.remove_baseline()
    s_smooth = s_corr.smooth()
    s_norm = s_smooth.normalize()
    peaks = s_norm.find_peaks(threshold_rel=0.05)

    peak_positions = [round(p.pos, 2) for p in peaks]
    peak_intensities = [round(p.height, 3) for p in peaks]

    # Plot
    fig, ax = plt.subplots()
    s_norm.plot(ax=ax, label="Processado")
    peaks.plot(ax=ax, marker="o", color="r", label="Picos")
    ax.set_xlabel("Número de onda (cm⁻¹)")
    ax.set_ylabel("Intensidade (a.u.)")
    ax.legend()

    return {
        "df": df,
        "processed_spectrum": s_norm,
        "peaks": peak_positions,
        "intensities": peak_intensities,
        "figure": fig
    }
