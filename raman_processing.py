import numpy as np
import matplotlib.pyplot as plt
import ramanchada2 as rc2

def process_raman(df):
    s = rc2.spectrum.from_array(df["wavenumber_cm1"].values, df["intensity_a"].values)
    s_corr = s.remove_baseline()
    s_smooth = s_corr.smooth()
    s_norm = s_smooth.normalize()
    peaks = s_norm.find_peaks(threshold_rel=0.05)
    peak_positions = [round(p.pos, 2) for p in peaks]
    peak_intensities = [round(p.height, 3) for p in peaks]
    fig, ax = plt.subplots()
    s_norm.plot(ax=ax, label="Espectro processado")
    peaks.plot(ax=ax, marker="o", color="r", label="Picos")
    ax.set_xlabel("Número de onda (cm⁻¹)")
    ax.set_ylabel("Intensidade (a.u.)")
    ax.legend()
    return {"processed_spectrum": s_norm, "peaks": peak_positions, "intensities": peak_intensities, "figure": fig}
