import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, precision_recall_curve, roc_curve

params = [
    "track_ptrel",
    "track_erel",
    "track_phirel",
    "track_etarel",
    "track_deltaR",
    "track_drminsv",
    "track_drsubjet1",
    "track_drsubjet2",
    "track_dz",
    "track_dzsig",
    "track_dxy",
    "track_dxysig",
    "track_normchi2",
    "track_quality",
    "track_dptdpt",
    "track_detadeta",
    "track_dphidphi",
    "track_dxydxy",
    "track_dzdz",
    "track_dxydz",
    "track_dphidxy",
    "track_dlambdadz",
    "trackBTag_EtaRel",
    "trackBTag_PtRatio",
    "trackBTag_PParRatio",
    "trackBTag_Sip2dVal",
    "trackBTag_Sip2dSig",
    "trackBTag_Sip3dVal",
    "trackBTag_Sip3dSig",
    "trackBTag_JetDistVal",
]

params_sv = [
    "sv_ptrel",
    "sv_erel",
    "sv_phirel",
    "sv_etarel",
    "sv_deltaR",
    "sv_pt",
    "sv_mass",
    "sv_ntracks",
    "sv_normchi2",
    "sv_dxy",
    "sv_dxysig",
    "sv_d3d",
    "sv_d3dsig",
    "sv_costhetasvpv",
]

softmax = torch.nn.Softmax(dim=1)
Ntracks = 60
Nverts = 5


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


def LRP(  # noqa: C901
    Rin,
    weights,
    biases,
    activations,
    debug=False,
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
    if mode == "gamma+" or mode == "gamma":
        weights = weights + gamma * torch.ones(weights.shape, device="cuda") * (weights > 0)
    elif mode == "gamma-":
        weights = weights + gamma * torch.ones(weights.shape, device="cuda") * (weights <= 0)
    elif mode == "ab":
        weights = (
            weights
            + alpha * torch.ones(weights.shape, device="cuda") * (weights > 0)
            + beta * torch.ones(weights.shape, device="cuda") * (weights <= 0)
        )
    if debug:
        print("\nNext layer relevances sum", Rin.sum())
        print("Next layer relevances size", Rin.size())
        print("Weight matrix size", weights.size())
        print("Biases size", biases.size())
        print("Prev layer activation size", activations.size())

    denominator = torch.matmul(torch.transpose(weights, 0, 1), activations) + include_bias * biases.reshape(-1, 1)
    denominator[denominator == 0] = eps / 10.0
    if mode == "eps":
        denominator += eps * torch.sign(denominator)

    if extend_dendrop:
        dendropat = dendrop_threshold
    else:
        dendropat = 0.0
    den2drop = torch.abs(denominator) <= dendropat

    if debug:
        print("Denominator drop threshold", dendropat)
        print("How many denominators are zero?", torch.sum(den2drop).item())
        print("Cumulative relevance of zeroed instances:", Rin[den2drop].sum().item())

    if True:
        fs_Rin = Rin.sum().item() / (Rin.sum().item() - Rin[den2drop].sum().item())
        if debug:
            print("Rescaling Factor for Rin:", fs_Rin)
        Rin[den2drop] = 0.0
        denominator[den2drop] = 1.0
        Rin = Rin * fs_Rin

    scaledR = Rin / denominator
    Rout = torch.matmul(weights, scaledR) * activations

    if debug:
        print("Prev layer relevances size", Rout.size())
        print("Prev layer relevances sum", Rout.sum(), "\n")

    return Rout


def LRPEvaluator(  # noqa: C901
    model,
    x,
    y,
    def_state_dict,
    weighted_firing=False,
    debug=False,
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

    if debug:
        print("target = ", target)

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
    if debug:
        print("\nDoing LRP for fc_fixed)")

    rel_fc_fixed = LRP(
        Rin=Relevances,
        weights=torch.transpose(def_state_dict["fc_fixed.weight"], 0, 1)[:, target].reshape(-1, 1),
        biases=def_state_dict["fc_fixed.bias"][target].reshape(-1),
        activations=torch.transpose(N_in, 1, 2),
        debug=debug,
        mode=LRP_mode,
        eps=eps,
        gamma=gamma,
        include_bias=include_bias,
    )

    hidden_relevance.append((rel_fc_fixed / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fo3")

    # step-2: relevance propagation from O_bar -> O

    Omatrix = torch.transpose((Omatrix + 0) / (N_in + 1.0e-5), 1, 2)
    if debug:
        print("\nActivation distribution shape", Omatrix.shape)
        print(Omatrix[:, 0:5, :])
        print(torch.sum(Omatrix[:, 0:5, :], dim=2))
    rel_O = rel_fc_fixed * Omatrix

    if debug:
        print("\nFixed FC size after sum", rel_fc_fixed.size())
        print(rel_fc_fixed.reshape(-1))
        print("Fixed FC size before sum", rel_O.size())
        print("Fixed FC relevance before sum", rel_O.sum())
        print("Fo:", rel_O.max(), rel_O.min())

    # step-3: relevance propagation across fo

    if debug:
        print("\nDoing LRP for fo")

    rel_fo3 = LRP(
        Rin=rel_O,
        weights=torch.transpose(def_state_dict["fo3.weight"], 0, 1),
        biases=def_state_dict["fo3.bias"],
        activations=torch.transpose(C2.reshape(Nb, model.N, model.hidden), 1, 2),
        debug=debug,
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )

    hidden_relevance.append((rel_fo3 / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fo2")

    if debug:
        print("fo3", rel_fo3.max(), rel_fo3.min())

    rel_fo2 = LRP(
        Rin=rel_fo3,
        weights=torch.transpose(def_state_dict["fo2.weight"], 0, 1),
        biases=def_state_dict["fo2.bias"],
        activations=torch.transpose(C1.reshape(Nb, model.N, model.hidden), 1, 2),
        debug=debug,
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )

    hidden_relevance.append((rel_fo2 / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fo1")

    if debug:
        print("fo2", rel_fo2.max(), rel_fo2.min())

    rel_fo1 = LRP(
        Rin=rel_fo2,
        weights=torch.transpose(def_state_dict["fo1.weight"], 0, 1),
        biases=def_state_dict["fo1.bias"],
        activations=torch.transpose(C, 1, 2),
        debug=debug,
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )

    if debug:
        print("fo1", rel_fo1.max(), rel_fo1.min())

    # Step-4: relevance redistribution for Interaction Matrices

    rel_Epp = rel_fo1[:, model.P : model.P + model.De, :]
    rel_Epv = rel_fo1[:, model.P + model.De : model.P + 2 * model.De, :]
    particle_relevances = particle_relevances + (rel_fo1[:, : model.P, :] / Relevances.reshape(Nb, 1, 1)).detach().cpu()

    if debug:
        print("\nParticle Relevances Sum: ", particle_relevances.sum())
        print("Particle Relevances Max: ", particle_relevances.max())
        print("Particle Relevances Min: ", particle_relevances.min())

    if debug:
        print("\nrel_Epp shape before IN: ", rel_Epp.size())
        print("rel_Epp sum before IN: ", rel_Epp.sum())
        print("rel_Epv shape before IN: ", rel_Epv.size())
        print("rel_Epv sum before IN: ", rel_Epv.sum())
        print("Remaining relevance sum before IN: ", rel_fo1[:, : model.P, :].sum())
        print("Ebar_pp shape:", Ebar_pp.shape)
        print("Epp shape:", Epp.shape)
        print("Ebar_pv shape:", Ebar_pv.shape)
        print("Epv shape:", Epv.shape)

    Ebar_pp = torch.transpose(C, 1, 2)[:, model.P : model.P + model.De, :]
    Ebar_pv = torch.transpose(C, 1, 2)[:, model.P + model.De : model.P + 2 * model.De, :]
    rel_Epp = torch.matmul(rel_Epp / (Ebar_pp + 1.0e-5), model.Rr) * Epp
    rel_Epv = torch.matmul(rel_Epv / (Ebar_pv + 1.0e-5), model.Rk) * Epv

    if debug:
        print("\nEpp relevance shape after IN: ", rel_Epp.size())
        print("Epv relevance shape after IN: ", rel_Epv.size())
        print("Epp relevance sum before IN: ", rel_Epp.sum())
        print("Epv relevance sum before IN: ", rel_Epv.sum())
        print("Epp", rel_Epp.max(), rel_Epp.min())
        print("Epv", rel_Epv.max(), rel_Epv.min())

    if debug:
        print("\nDoing LRP for fr")

    # Step-5: relevance distribution across fr network (PC-PC)

    hidden_relevance.append((rel_Epp / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fr3")

    rel_fr3 = LRP(
        Rin=rel_Epp,
        weights=torch.transpose(def_state_dict["fr3.weight"], 0, 1),
        biases=def_state_dict["fr3.bias"],
        activations=torch.transpose(B2.reshape(Nb, model.Nr, model.hidden), 1, 2),
        debug=debug,
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )
    if debug:
        print("fr3", rel_fr3.max(), rel_fr3.min())

    hidden_relevance.append((rel_fr3 / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fr2")

    rel_fr2 = LRP(
        Rin=rel_fr3,
        weights=torch.transpose(def_state_dict["fr2.weight"], 0, 1),
        biases=def_state_dict["fr2.bias"],
        activations=torch.transpose(B1.reshape(Nb, model.Nr, model.hidden), 1, 2),
        debug=debug,
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )

    if debug:
        print("fr2", rel_fr2.max(), rel_fr2.min())

    hidden_relevance.append((rel_fr2 / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fr1")

    rel_fr1 = LRP(
        Rin=rel_fr2,
        weights=torch.transpose(def_state_dict["fr1.weight"], 0, 1),
        biases=def_state_dict["fr1.bias"],
        activations=torch.transpose(B, 1, 2),
        debug=debug,
        extend_dendrop=True,
        dendrop_threshold=dendrop_threshold,
        mode=LRP_mode,
        gamma=gamma,
        eps=eps,
        include_bias=include_bias,
    )

    if debug:
        print("fr1", rel_fr1.max(), rel_fr1.min())

    # Step-6: relevance distribution across Rr and Rs network matrices (PC-PC)

    rel_Rr = rel_fr1[:, : model.P, :]
    rel_Rs = rel_fr1[:, model.P :, :]
    if debug:
        print("\nRr relevance shape before IN: ", rel_Rr.size())
        print("Rs relevance shape before IN: ", rel_Rs.size())
        print("Rr relevance sum before IN: ", rel_Rr.sum())
        print("Rs relevance sum before IN: ", rel_Rs.sum())

    rel_Rr = torch.matmul(rel_Rr, torch.transpose(model.Rr, 0, 1))
    rel_Rs = torch.matmul(rel_Rs, torch.transpose(model.Rs, 0, 1))

    if debug:
        print("\nRr relevance shape after IN: ", rel_Rr.size())
        print("Rs relevance shape after IN: ", rel_Rs.size())
        print("Rr relevance sum after IN: ", rel_Rr.sum())
        print("Rs relevance sum after IN: ", rel_Rs.sum())
        print("rel_Rr", rel_Rr.max(), rel_Rr.min())
        print("rel_Rs", rel_Rs.max(), rel_Rs.min())

    particle_relevances = (
        particle_relevances
        + (rel_Rr / Relevances.reshape(Nb, 1, 1)).detach().cpu()
        + (rel_Rs / Relevances.reshape(Nb, 1, 1)).detach().cpu()
    )

    if debug:
        print("\nParticle Relevances Sum: ", particle_relevances.sum())

    # Step-7: relevance distribution across fr_pv network (PC-SV)

    if debug:
        print("\nDoing LRP for fr_pv")

    hidden_relevance.append((rel_Epv / Relevances.reshape(Nb, 1, 1)).sum(dim=(0, 2)).detach().cpu().numpy())
    tags.append("fr3_pv")

    rel_fr3_pv = LRP(
        Rin=rel_Epv,
        weights=torch.transpose(def_state_dict["fr3_pv.weight"], 0, 1),
        biases=def_state_dict["fr3_pv.bias"],
        activations=torch.transpose(Bpv2.reshape(Nb, model.Nt, model.hidden), 1, 2),
        debug=debug,
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
        debug=debug,
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
        debug=debug,
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
    if debug:
        print("\nRk relevance shape before IN: ", rel_Rk.size())
        print("Rv relevance shape before IN: ", rel_Rv.size())
        print("Rk relevance sum before IN: ", rel_Rk.sum())
        print("Rv relevance sum before IN: ", rel_Rv.sum())
    rel_Rk = torch.matmul(rel_fr1_pv[:, : model.P, :], torch.transpose(model.Rk, 0, 1))
    rel_Rv = torch.matmul(rel_fr1_pv[:, model.P :, :], torch.transpose(model.Rv, 0, 1))

    if debug:
        print("\nRk relevance shape after IN: ", rel_Rk.size())
        print("Rv relevance shape after IN: ", rel_Rv.size())
        print("Rk relevance sum after IN: ", rel_Rk.sum())
        print("Rv relevance sum after IN: ", rel_Rv.sum())

    particle_relevances = particle_relevances + (rel_Rk / Relevances.reshape(Nb, 1, 1)).detach().cpu()
    vertex_relevances = vertex_relevances + (rel_Rv / Relevances.reshape(Nb, 1, 1)).detach().cpu()

    if debug:
        print("\nParticle Relevances Sum: ", particle_relevances.sum())
        print("Vertex Relevances Sum: ", vertex_relevances.sum())
        print("Outsput shape:", N_out.shape)

    return N_out, particle_relevances, vertex_relevances, hidden_relevance, tags
