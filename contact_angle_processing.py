import numpy as np
import matplotlib.pyplot as plt

def process_contact_angle(df, fit_order=3):
    t = df["t_seconds"].values
    theta = df["angle_mean_deg"].values
    coef = np.polyfit(t, theta, fit_order)
    poly = np.poly1d(coef)
    theta_fit = poly(t)
    dtheta_dt = np.gradient(theta_fit, t)
    gamma_ratio = np.mean(np.cos(np.radians(theta_fit)))
    fig, ax = plt.subplots()
    ax.plot(t, theta, "bo", label="Experimental")
    ax.plot(t, theta_fit, "r-", label=f"Ajuste polinomial ({fit_order}º grau)")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Ângulo de contato (°)")
    ax.legend()
    return {"coef": coef.tolist(), "gamma_ratio": gamma_ratio, "dtheta_dt": dtheta_dt.tolist(), "figure": fig}
