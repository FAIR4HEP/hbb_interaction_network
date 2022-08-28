from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import auc, precision_recall_curve, roc_curve

project_dir = Path(__file__).resolve().parents[2]
train_path = f"{project_dir}/data/processed/train/"
definitions = f"{project_dir}/src/data/definitions.yml"
with open(definitions) as yaml_file:
    defn = yaml.load(yaml_file, Loader=yaml.FullLoader)

params = defn["features_2"]
params_sv = defn["features_3"]

softmax = torch.nn.Softmax(dim=1)
Ntracks = defn["nobj_2"]
Nverts = defn["nobj_3"]


def eval(  # noqa: C901
    model,
    data,
    drop_pfeatures=torch.tensor([], dtype=torch.long),
    drop_svfeatures=torch.tensor([], dtype=torch.long),
    mask_pfeatures=torch.tensor([], dtype=torch.long),
    mask_svfeatures=torch.tensor([], dtype=torch.long),
    mask_tracks=torch.tensor([], dtype=torch.long),
    mask_vertices=torch.tensor([], dtype=torch.long),
    sort_vertices=False,
    sort_tracks=False,
    track_column_shuffle=torch.tensor(np.arange(Ntracks), dtype=torch.long),
    vertex_column_shuffle=torch.tensor(np.arange(Nverts), dtype=torch.long),
    training_all=[],
    training_sv_all=[],
    save_data=False,
):

    lst = []
    correct = []

    with torch.no_grad():
        for sub_X, sub_Y, _ in data.generate_data():
            training = sub_X[2]
            training_sv = sub_X[3]
            if save_data:
                training_all.append(training)
                training_sv_all.append(training_sv)
            target = sub_Y[0]
            trainingv = (torch.FloatTensor(training)).cuda()

            if sort_tracks:  # sorting tracks by energy => index 1 of the feature list
                _, inds = torch.sort(torch.tensor(trainingv[:, 1, :]), descending=True)
                for ii in range(trainingv.shape[0]):
                    trainingv[ii] = trainingv[ii][:, inds[ii]]
            elif len(mask_tracks) == 0:
                trainingv = trainingv[:, :, track_column_shuffle]

            if len(mask_tracks) > 0:
                trainingv[:, :, mask_tracks] *= 0

            if len(drop_pfeatures) > 0:
                keep_features = [i for i in np.arange(0, len(params), 1, dtype=int) if i not in drop_pfeatures]
                trainingv = trainingv[:, keep_features, :]
            elif len(mask_pfeatures) > 0:
                trainingv[:, mask_pfeatures, :] *= 0

            trainingv_sv = (torch.FloatTensor(training_sv)).cuda()

            if sort_vertices:
                _, inds = torch.sort(torch.tensor(trainingv_sv[:, 1, :]), descending=True)
                for ii in range(trainingv.shape[0]):
                    trainingv[ii] = trainingv[ii][:, inds[ii]]
            elif len(mask_vertices) == 0:
                trainingv_sv = trainingv_sv[:, :, vertex_column_shuffle]

            if len(mask_vertices) > 0:
                trainingv[:, :, mask_vertices] *= 0

            if len(drop_svfeatures) > 0:
                keep_features = [i for i in np.arange(0, len(params_sv), 1, dtype=int) if i not in drop_svfeatures]
                trainingv_sv = trainingv_sv[:, keep_features, :]
            elif len(mask_svfeatures) > 0:
                trainingv_sv[:, mask_svfeatures, :] *= 0

            # targetv = (torch.from_numpy(np.argmax(target, axis = 1)).long()).cuda()

            out = model.forward(trainingv.cuda(), trainingv_sv.cuda())
            lst.append(softmax(out).cpu().data.numpy())
            correct.append(target)

    predicted = np.concatenate(lst)
    val_targetv = np.concatenate(correct)

    return predicted, val_targetv


class ModelComparison:
    def __init__(self, preds, targets, model_tags):
        self.preds = preds
        self.targets = targets
        self.n_models = len(preds)
        self.model_tags = model_tags
        self.aucs_roc = []
        self.aucs_prc = []
        self.fidelity = []

    def plot_roc(self, fname):
        if len(self.preds) > 4:
            plt.figure(figsize=(8, 8))
        else:
            plt.figure(figsize=(4, 4))
        for ii in range(self.n_models):
            fpr, tpr, _ = roc_curve(self.targets[ii], self.preds[ii])
            self.aucs_roc.append(auc(fpr, tpr))
            # print(self.model_tags[ii])
            plt.plot(
                fpr,
                tpr,
                label=self.model_tags[ii] + " ({:.2f}%)".format(self.aucs_roc[ii] * 100),
            )
        plt.yscale("log")
        plt.xscale("log")
        plt.xlim([1e-5, 1.00])
        plt.ylim([1e-2, 1.00])
        plt.xlabel("FPR", fontsize=20)
        plt.ylabel("TPR", fontsize=20)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()
        return self.aucs_roc

    def plot_prc(self, fname):
        if len(self.preds) > 4:
            plt.figure(figsize=(8, 8))
        else:
            plt.figure(figsize=(4, 4))
        for ii in range(self.n_models):
            precision, recall, _ = precision_recall_curve(self.targets[ii], self.preds[ii])
            self.aucs_prc.append(auc(recall, precision))
            plt.plot(
                recall,
                precision,
                label=self.model_tags[ii] + " ({:.2f}%)".format(self.aucs_prc[ii] * 100),
            )
        plt.xlabel("Recall", fontsize=20)
        plt.ylabel("Precision", fontsize=20)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()
        return self.aucs_prc

    def get_fidelity(self, pred_0):
        for ii in range(self.n_models):
            self.fidelity.append(1.0 - np.mean(np.abs(pred_0 - self.preds[ii])))
        return self.fidelity


def print_scores(vals, model_tags, mode="roc"):
    if mode == "roc":
        statement = "ROC AUC for {}: {:.2f}"
    if mode == "prc":
        statement = "PRC AUC for {}: {:.2f}"
    if mode == "fidelity":
        statement = "Fidelity for {}: {:.2f}"
    for ii in range(len(model_tags)):
        print(statement.format(model_tags[ii], vals[ii] * 100))


def dAUC_chart(dAUC_vals_roc, tags, fname, dAUC_vals_prc=[]):
    plot_auc_rpc = False
    if len(dAUC_vals_prc) == len(dAUC_vals_roc):
        kfact = 2
        plot_auc_rpc = True
    else:
        kfact = 1
    pos = kfact * np.arange(len(dAUC_vals_roc))
    if len(tags) < 10:
        plt.figure(figsize=(4, 4))
    else:
        plt.figure(figsize=(8, 8))
    plt.bar(pos, dAUC_vals_roc, align="center", label="ROC curve")
    if plot_auc_rpc:
        plt.bar(pos + 1, dAUC_vals_roc, align="center", label="Precision-Recall Curve")
    plt.xticks(pos + 0.5 * plot_auc_rpc, tags, rotation="vertical")
    plt.ylabel("Percent Drop in AUC")
    if plot_auc_rpc:
        plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()


def fidelity_chart(fidelity_vals, tags, fname):
    if len(tags) < 10:
        plt.figure(figsize=(4, 4))
    else:
        plt.figure(figsize=(8, 8))
    pos = np.arange(len(fidelity_vals))
    plt.bar(pos, 1.0 - np.array(fidelity_vals), align="center")
    plt.xticks(pos, tags, rotation="vertical")
    plt.ylabel("1 - F")
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()


def weight_modifier(weights, mode, gamma, alpha, beta):
    func_dict = {
        "zero": lambda w: w,
        "eps": lambda w: w,
        "gamma": lambda w: w + gamma * torch.ones(w.shape, device=w.device) * (w > 0) * w,
        "gamma+": lambda w: w + gamma * torch.ones(w.shape, device=w.device) * (w > 0) * w,
        "gamma-": lambda w: w + gamma * torch.ones(w.shape, device=w.device) * (w <= 0) * w,
        "ab": lambda w: w
        + alpha * torch.ones(w.shape, device=w.device) * (w > 0) * w
        + beta * torch.ones(w.shape, device=w.device) * (w <= 0) * w,
    }

    return func_dict[mode](weights)


def LRP(  # noqa: C901
    Rin,
    weights,
    biases,
    activations,
    include_bias=False,
    mode="zero",
    eps=1.0,
    gamma=2.0,
    beta=1.0,
    extend_dendrop=False,
    dendrop_threshold=1.0,
):
    """expected sizes:
    Rin : (Nb, N_next, N_candidates)
    weights : (N_prev, N_next)
    biases: (N_next)
    activations: (Nb, N_prev, N_candidates)
    returns Rout: (Nb, N_prev, N_candidates)
    eps = 0.25 * torch.std(Rin, dim=(0,2), unbiased = False)
    shape (N_prev, N_next).T * (Nb, N_prev, N_candidates) + (N_next, 1) = (Nb, N_next, N_candidates)
    """

    if mode not in ["zero", "eps", "gamma", "gamma+", "gamma-", "ab"]:
        print("Unrecognized mode! Defaulting to LRP-0")
        mode = "zero"

    alpha = beta + 1
    weights = weight_modifier(weights, mode, gamma, alpha, beta)
    biases = weight_modifier(biases, mode, gamma, alpha, beta)
    denominator = torch.matmul(torch.transpose(weights, 0, 1), activations) + include_bias * biases.reshape(-1, 1)
    denominator[denominator == 0] = eps / 10.0
    if mode == "eps":
        denominator += eps * torch.sign(denominator)

    if extend_dendrop:
        dendropat = dendrop_threshold
    else:
        dendropat = 0.0
    den2drop = torch.abs(denominator) <= dendropat

    fs_Rin = Rin.sum().item() / (Rin.sum().item() - Rin[den2drop].sum().item())
    Rin[den2drop] = 0.0
    denominator[den2drop] = 1.0
    Rin = Rin * fs_Rin

    scaledR = Rin / denominator
    Rout = torch.matmul(weights, scaledR) * activations

    return Rout


def LRPEvaluator(  # noqa: C901
    model,
    x,
    y,
    def_state_dict,
    weighted_firing=False,
    target=0,
    LRP_mode="eps",
    eps=1.0,
    gamma=1.0,
    beta=1.0,
    dendrop_threshold=0.0,
    include_bias=False,
):

    Nb = x.shape[0]
    particle_relevances = torch.zeros(x.shape)
    vertex_relevances = torch.zeros(y.shape)
    target = int(target)
    hidden_relevance = []
    tags = []

    # PF Candidate - PF Candidate
    Orr = model.tmul(x, model.Rr)
    Ors = model.tmul(x, model.Rs)
    B = torch.cat([Orr, Ors], 1)
    # First MLP
    B = torch.transpose(B, 1, 2).contiguous()
    B1 = nn.functional.relu(model.fr1(B.view(-1, 2 * model.P + model.Dr)))
    B2 = nn.functional.relu(model.fr2(B1))
    E = nn.functional.relu(model.fr3(B2))
    E = E.view(-1, model.Nr, model.De)
    Epp = torch.transpose(E, 1, 2).contiguous()
    Ebar_pp = model.tmul(Epp, torch.transpose(model.Rr, 0, 1).contiguous())
    del E

    # Secondary Vertex - PF Candidate
    Ork = model.tmul(x, model.Rk)
    Orv = model.tmul(y, model.Rv)
    Bpv = torch.cat([Ork, Orv], 1)
    # First MLP
    Bpv = torch.transpose(Bpv, 1, 2).contiguous()
    Bpv1 = nn.functional.relu(model.fr1_pv(Bpv.view(-1, model.S + model.P + model.Dr)))
    Bpv2 = nn.functional.relu(model.fr2_pv(Bpv1))
    E = nn.functional.relu(model.fr3_pv(Bpv2))
    E = E.view(-1, model.Nt, model.De)

    Epv = torch.transpose(E, 1, 2).contiguous()
    Ebar_pv = model.tmul(Epv, torch.transpose(model.Rk, 0, 1).contiguous())

    del E

    # Final output matrix for particles
    C = torch.cat([x, Ebar_pp, Ebar_pv], 1)
    C = torch.transpose(C, 1, 2).contiguous()
    # Second MLP
    C1 = nn.functional.relu(model.fo1(C.view(-1, model.P + model.Dx + (2 * model.De))))
    C2 = nn.functional.relu(model.fo2(C1))
    Omatrix = nn.functional.relu(model.fo3(C2))
    Omatrix = Omatrix.view(-1, model.N, model.Do)

    # Taking the sum of over each particle/vertex
    N_in = torch.sum(Omatrix, dim=1).reshape(Nb, 1, model.Do)

    # Classification MLP
    N_out = model.fc_fixed(N_in)

    # This is where we start calculating the LRP for for different layers

    # First calculating the total relevance of the desired jet class
    Relevances = N_out[:, :, target].reshape(Nb, 1, -1)

    # step-1: relevance for fc_fixed

    rel_fc_fixed = LRP(
        Rin=Relevances,
        weights=torch.transpose(def_state_dict["fc_fixed.weight"], 0, 1)[:, target].reshape(-1, 1),
        biases=def_state_dict["fc_fixed.bias"][target].reshape(-1),
        activations=torch.transpose(N_in, 1, 2),
        mode=LRP_mode,
        eps=eps,
        gamma=gamma,
        include_bias=include_bias,
    )

    hidden_relevance.append((rel_fc_fixed / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fo3")

    # step-2: relevance propagation from O_bar -> O

    Omatrix = torch.transpose((Omatrix + 0) / (N_in + 1.0e-5), 1, 2)
    rel_O = rel_fc_fixed * Omatrix

    # step-3: relevance propagation across fo

    rel_fo3 = LRP(
        Rin=rel_O,
        weights=torch.transpose(def_state_dict["fo3.weight"], 0, 1),
        biases=def_state_dict["fo3.bias"],
        activations=torch.transpose(C2.reshape(Nb, model.N, model.hidden), 1, 2),
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )

    hidden_relevance.append((rel_fo3 / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fo2")

    rel_fo2 = LRP(
        Rin=rel_fo3,
        weights=torch.transpose(def_state_dict["fo2.weight"], 0, 1),
        biases=def_state_dict["fo2.bias"],
        activations=torch.transpose(C1.reshape(Nb, model.N, model.hidden), 1, 2),
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )

    hidden_relevance.append((rel_fo2 / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fo1")

    rel_fo1 = LRP(
        Rin=rel_fo2,
        weights=torch.transpose(def_state_dict["fo1.weight"], 0, 1),
        biases=def_state_dict["fo1.bias"],
        activations=torch.transpose(C, 1, 2),
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )

    # Step-4: relevance redistribution for Interaction Matrices

    rel_Epp = rel_fo1[:, model.P : model.P + model.De, :]
    rel_Epv = rel_fo1[:, model.P + model.De : model.P + 2 * model.De, :]
    particle_relevances = particle_relevances + (rel_fo1[:, : model.P, :] / Relevances.reshape(Nb, 1, 1)).detach().cpu()
    Ebar_pp = torch.transpose(C, 1, 2)[:, model.P : model.P + model.De, :]
    Ebar_pv = torch.transpose(C, 1, 2)[:, model.P + model.De : model.P + 2 * model.De, :]
    rel_Epp = torch.matmul(rel_Epp / (Ebar_pp + 1.0e-5), model.Rr) * Epp
    rel_Epv = torch.matmul(rel_Epv / (Ebar_pv + 1.0e-5), model.Rk) * Epv

    # Step-5: relevance distribution across fr network (PC-PC)

    hidden_relevance.append((rel_Epp / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fr3")

    rel_fr3 = LRP(
        Rin=rel_Epp,
        weights=torch.transpose(def_state_dict["fr3.weight"], 0, 1),
        biases=def_state_dict["fr3.bias"],
        activations=torch.transpose(B2.reshape(Nb, model.Nr, model.hidden), 1, 2),
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )

    hidden_relevance.append((rel_fr3 / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fr2")

    rel_fr2 = LRP(
        Rin=rel_fr3,
        weights=torch.transpose(def_state_dict["fr2.weight"], 0, 1),
        biases=def_state_dict["fr2.bias"],
        activations=torch.transpose(B1.reshape(Nb, model.Nr, model.hidden), 1, 2),
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )

    hidden_relevance.append((rel_fr2 / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fr1")

    rel_fr1 = LRP(
        Rin=rel_fr2,
        weights=torch.transpose(def_state_dict["fr1.weight"], 0, 1),
        biases=def_state_dict["fr1.bias"],
        activations=torch.transpose(B, 1, 2),
        extend_dendrop=True,
        dendrop_threshold=dendrop_threshold,
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )

    # Step-6: relevance distribution across Rr and Rs network matrices (PC-PC)

    rel_Rr = rel_fr1[:, : model.P, :]
    rel_Rs = rel_fr1[:, model.P :, :]
    rel_Rr = torch.matmul(rel_Rr, torch.transpose(model.Rr, 0, 1))
    rel_Rs = torch.matmul(rel_Rs, torch.transpose(model.Rs, 0, 1))

    particle_relevances = (
        particle_relevances
        + (rel_Rr / Relevances.reshape(Nb, 1, 1)).detach().cpu()
        + (rel_Rs / Relevances.reshape(Nb, 1, 1)).detach().cpu()
    )

    # Step-7: relevance distribution across fr_pv network (PC-SV)

    hidden_relevance.append((rel_Epv / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fr3_pv")

    rel_fr3_pv = LRP(
        Rin=rel_Epv,
        weights=torch.transpose(def_state_dict["fr3_pv.weight"], 0, 1),
        biases=def_state_dict["fr3_pv.bias"],
        activations=torch.transpose(Bpv2.reshape(Nb, model.Nt, model.hidden), 1, 2),
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )

    hidden_relevance.append((rel_fr3_pv / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fr2_pv")

    rel_fr2_pv = LRP(
        Rin=rel_fr3_pv,
        weights=torch.transpose(def_state_dict["fr2_pv.weight"], 0, 1),
        biases=def_state_dict["fr2_pv.bias"],
        activations=torch.transpose(Bpv1.reshape(Nb, model.Nt, model.hidden), 1, 2),
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )

    hidden_relevance.append((rel_fr2_pv / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fr1_pv")

    rel_fr1_pv = LRP(
        Rin=rel_fr2_pv,
        weights=torch.transpose(def_state_dict["fr1_pv.weight"], 0, 1),
        biases=def_state_dict["fr1_pv.bias"],
        activations=torch.transpose(Bpv, 1, 2),
        extend_dendrop=True,
        dendrop_threshold=dendrop_threshold,
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )

    # Step-8: relevance distribution across Rr and Rs network matrices (PC-PC)

    rel_Rk = rel_fr1_pv[:, : model.P, :]
    rel_Rv = rel_fr1_pv[:, model.P :, :]
    rel_Rk = torch.matmul(rel_fr1_pv[:, : model.P, :], torch.transpose(model.Rk, 0, 1))
    rel_Rv = torch.matmul(rel_fr1_pv[:, model.P :, :], torch.transpose(model.Rv, 0, 1))
    particle_relevances = particle_relevances + (rel_Rk / Relevances.reshape(Nb, 1, 1)).detach().cpu()
    vertex_relevances = vertex_relevances + (rel_Rv / Relevances.reshape(Nb, 1, 1)).detach().cpu()

    return N_out, particle_relevances, vertex_relevances, hidden_relevance, tags
