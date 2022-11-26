import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from sklearn.metrics import auc

plt.style.use(hep.style.ROOT)


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


if __name__ == "__main__":

    model_names = [
        "gnn_max_npv_15_best",
        "gnn_max_npv_15_just_tracks_best",
        "gnn_max_npv_15_just_svs_best",
        "projector_linear_max_npv_15_best",
        "projector_finetune_max_npv_15_best",
    ]
    model_labels = [
        "IN(particles, SVs)",
        "IN(particles)",
        "IN(SVs)",
        "VICReg+Linear(particles, SVs)",
        "Finetuned VICReg+Linear(particles, SVs)",
    ]
    lines = ["-", "--", "-.", ":", "--."]
    pu_label = "min_npv_15"  # max_npv_15
    min_msd = 40.0
    max_msd = 200.0
    min_pt = 300.0
    max_pt = 2000.0

    if pu_label == "max_npv_15":
        pu_tag = r"$n_{{PV}} < 15$"
    else:
        pu_tag = r"$n_{{PV}} \geq 15$"

    plt.figure()
    for model_name, model_label, line in zip(model_names, model_labels, lines):
        model_perf_loc = "models/model_performances"

        fpr = np.load(f"{model_perf_loc}/{model_name}_test_fpr_{pu_label}.npy")
        tpr = np.load(f"{model_perf_loc}/{model_name}_test_tpr_{pu_label}.npy")
        plabel = (
            f"{model_label}\n"
            + f"AUC = {auc(fpr, tpr)*100:.1f}%, "
            + f"$\epsilon_S(\epsilon_B=10^{{-2}})$ = {tpr[find_nearest(fpr, 0.01)]*100:.1f}%"  # noqa: W605
        )
        plt.plot(tpr, fpr, label=plabel, ls=line)
    plt.semilogy()
    plt.legend(
        title=(
            f"${min_msd:.0f} < m_{{SD}} < {max_msd:.0f}$ GeV, "
            + f"${min_pt:.0f} < p_{{T}} < {max_pt:.0f}$ GeV, "
            + f"{pu_tag}"
        ),
        loc="upper left",
        fontsize=18,
        title_fontsize=18,
    )
    plt.xlabel(r"$H(b\bar{b})$ identification probability")
    plt.ylabel("QCD misidentification probability")
    plt.xlim([0, 1])
    plt.ylim([1e-5, 100])
    plt.savefig(f"roc_{pu_label}.pdf")
    plt.savefig(f"roc_{pu_label}.png")
