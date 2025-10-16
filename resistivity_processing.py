import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def process_resistivity(file, thickness_m=200e-9, mode="filme"):
    """
    Lê arquivo .csv com colunas: current_a, voltage_v
    Ajusta curva I x V, calcula R, resistividade, condutividade e classe.
    """
    df = pd.read_csv(file)
    if "current_a" not in df.columns:
        df = df.rename(columns={"I": "current_a", "Current": "current_a"})
    if "voltage_v" not in df.columns:
        df = df.rename(columns={"V": "voltage_v", "Voltage": "voltage_v"})

    I = df["current_a"].values.reshape(-1, 1)
    V = df["voltage_v"].values

    model = LinearRegression().fit(I, V)
    R = model.coef_[0]
    R2 = model.score(I, V)

    if mode == "filme":
        k = np.pi / np.log(2)
        rho = k * R * thickness_m
    else:
        rho = R

    sigma = 1 / rho if rho != 0 else np.nan
    if sigma > 1e4:
        classe = "Condutor"
    elif 1e-2 < sigma <= 1e4:
        classe = "Semicondutor"
    else:
        classe = "Isolante"

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(I, V, label="Dados")
    ax.plot(I, model.predict(I), "r-", label=f"Ajuste Linear (R²={R2:.3f})")
    ax.set_xlabel("Corrente (A)")
    ax.set_ylabel("Tensão (V)")
    ax.legend()

    return {
        "df": df,
        "R": R,
        "rho": rho,
        "sigma": sigma,
        "classe": classe,
        "R2": R2,
        "figure": fig
    }
