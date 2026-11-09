"""Regenerate manuscript figures from saved experiment tables."""

from paper_experiments import OUT, pd, plot_uncertainty_error, plt
from utils import plot_privacy_tradeoff


if __name__ == "__main__":
    privacy = pd.read_excel(OUT / "exp6_privacy_utility.xlsx")
    plot_privacy_tradeoff(privacy, "exp6_privacy_tradeoff.pdf")

    samples = pd.read_csv(OUT / "exp7_uncertainty_error_samples.csv")
    plot_uncertainty_error(samples)

    matched = pd.read_csv(OUT / "exp8_matched_fl_controller.csv")
    main = pd.read_csv(OUT / "exp1_metrics_5seed.csv")
    prox = main.loc[main["Method"] == "FedUMPC"].iloc[0].copy()
    prox["Method"] = "Prox-FL"
    matched = pd.concat(
        [matched.loc[matched["Method"] != "Prox-FL"], prox.to_frame().T],
        ignore_index=True,
    )
    matched.to_csv(OUT / "exp8_matched_fl_controller.csv", index=False)

    participation = pd.read_csv(OUT / "exp9_participation_stress.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0))
    labels = ["1.00", "0.75", "0.50"]
    panels = [("Final_MSE", "Validation MSE"),
              ("SR_mean_pct", "Success rate (%)"),
              ("SC_mean_pct", "Safety compliance (%)")]
    for ax, (key, ylabel) in zip(axes, panels):
        ax.plot(labels, participation[key], marker="o", linewidth=2.2,
                color="#7D4AB3")
        ax.set_xlabel("Participation rate", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(OUT / "exp9_participation_stress.pdf", bbox_inches="tight")
    plt.close(fig)
