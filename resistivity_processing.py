import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def process_resistivity(df, thickness_m=200e-9, mode="filme"):
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
    fig, ax = plt.subplots()
    ax.scatter(I, V, label="Dados")
    ax.plot(I, model.predict(I), "r-", label=f"Ajuste Linear (R²={R2:.3f})")
    ax.set_xlabel("Corrente (A)")
    ax.set_ylabel("Tensão (V)")
    ax.legend()
    return {"R": R, "rho": rho, "sigma": sigma, "classe": classe, "R2": R2, "figure": fig}
