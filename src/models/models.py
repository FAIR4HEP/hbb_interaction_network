import itertools

import torch
import torch.nn as nn


class GraphNet(nn.Module):
    def __init__(
        self,
        n_constituents,
        n_targets,
        params,
        hidden,
        n_vertices,
        params_v,
        vv_branch=False,
        De=5,
        Do=6,
        softmax=False,
        device="cpu",
    ):
        super(GraphNet, self).__init__()
        self.hidden = int(hidden)
        self.P = params
        self.N = n_constituents
        self.S = params_v
        self.Nv = n_vertices
        self.Nr = self.N * (self.N - 1)
        self.Nt = self.N * self.Nv
        self.Ns = self.Nv * (self.Nv - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.device = device
        self.n_targets = n_targets
        self.assign_matrices()
        self.assign_matrices_SV()
        self.vv_branch = vv_branch
        self.softmax = softmax
        if self.vv_branch:
            self.assign_matrices_SVSV()

        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden).to(self.device)
        self.fr2 = nn.Linear(self.hidden, int(self.hidden)).to(self.device)
        self.fr3 = nn.Linear(int(self.hidden), self.De).to(self.device)
        self.fr1_pv = nn.Linear(self.S + self.P + self.Dr, self.hidden).to(self.device)
        self.fr2_pv = nn.Linear(self.hidden, int(self.hidden)).to(self.device)
        self.fr3_pv = nn.Linear(int(self.hidden), self.De).to(self.device)
        if self.vv_branch:
            self.fr1_vv = nn.Linear(2 * self.S + self.Dr, self.hidden).to(self.device)
            self.fr2_vv = nn.Linear(self.hidden, int(self.hidden)).to(self.device)
            self.fr3_vv = nn.Linear(int(self.hidden), self.De).to(self.device)
        self.fo1 = nn.Linear(self.P + self.Dx + (2 * self.De), self.hidden).to(self.device)
        self.fo2 = nn.Linear(self.hidden, int(self.hidden)).to(self.device)
        self.fo3 = nn.Linear(int(self.hidden), self.Do).to(self.device)
        if self.vv_branch:
            self.fo1_v = nn.Linear(self.S + self.Dx + (2 * self.De), self.hidden).to(self.device)
            self.fo2_v = nn.Linear(self.hidden, int(self.hidden)).to(self.device)
            self.fo3_v = nn.Linear(int(self.hidden), self.Do).to(self.device)

        if self.vv_branch:
            self.fc_fixed = nn.Linear(2 * self.Do, self.n_targets).to(self.device)
        else:
            self.fc_fixed = nn.Linear(self.Do, self.n_targets).to(self.device)

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0] != i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = (self.Rr).to(self.device)
        self.Rs = (self.Rs).to(self.device)

    def assign_matrices_SV(self):
        self.Rk = torch.zeros(self.N, self.Nt)
        self.Rv = torch.zeros(self.Nv, self.Nt)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.Nv))]
        for i, (k, v) in enumerate(receiver_sender_list):
            self.Rk[k, i] = 1
            self.Rv[v, i] = 1
        self.Rk = (self.Rk).to(self.device)
        self.Rv = (self.Rv).to(self.device)

    def assign_matrices_SVSV(self):
        self.Rl = torch.zeros(self.Nv, self.Ns)
        self.Ru = torch.zeros(self.Nv, self.Ns)
        receiver_sender_list = [i for i in itertools.product(range(self.Nv), range(self.Nv)) if i[0] != i[1]]
        for i, (l, u) in enumerate(receiver_sender_list):
            self.Rl[l, i] = 1
            self.Ru[u, i] = 1
        self.Rl = (self.Rl).to(self.device)
        self.Ru = (self.Ru).to(self.device)

    def forward(self, x, y):
        # PF Candidate - PF Candidate
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        # First MLP
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
        B = nn.functional.relu(self.fr2(B))
        E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pp = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E

        # Secondary Vertex - PF Candidate
        Ork = self.tmul(x, self.Rk)
        Orv = self.tmul(y, self.Rv)
        B = torch.cat([Ork, Orv], 1)
        # First MLP
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1_pv(B.view(-1, self.S + self.P + self.Dr)))
        B = nn.functional.relu(self.fr2_pv(B))
        E = nn.functional.relu(self.fr3_pv(B).view(-1, self.Nt, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pv = self.tmul(E, torch.transpose(self.Rk, 0, 1).contiguous())
        Ebar_vp = self.tmul(E, torch.transpose(self.Rv, 0, 1).contiguous())
        del E

        # Secondary vertex - secondary vertex
        if self.vv_branch:
            Orl = self.tmul(y, self.Rl)
            Oru = self.tmul(y, self.Ru)
            B = torch.cat([Orl, Oru], 1)
            # First MLP
            B = torch.transpose(B, 1, 2).contiguous()
            B = nn.functional.relu(self.fr1_vv(B.view(-1, 2 * self.S + self.Dr)))
            B = nn.functional.relu(self.fr2_vv(B))
            E = nn.functional.relu(self.fr3_vv(B).view(-1, self.Ns, self.De))
            del B
            E = torch.transpose(E, 1, 2).contiguous()
            Ebar_vv = self.tmul(E, torch.transpose(self.Rl, 0, 1).contiguous())
            del E

        # Final output matrix for particles
        C = torch.cat([x, Ebar_pp, Ebar_pv], 1)
        del Ebar_pp
        del Ebar_pv
        C = torch.transpose(C, 1, 2).contiguous()
        # Second MLP
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + (2 * self.De))))
        C = nn.functional.relu(self.fo2(C))
        Omatrix = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C

        if self.vv_branch:
            # Final output matrix for particles
            C = torch.cat([y, Ebar_vv, Ebar_vp], 1)
            del Ebar_vv
            del Ebar_vp
            C = torch.transpose(C, 1, 2).contiguous()
            # Second MLP
            C = nn.functional.relu(self.fo1_v(C.view(-1, self.S + self.Dx + (2 * self.De))))
            C = nn.functional.relu(self.fo2_v(C))
            O_v = nn.functional.relu(self.fo3_v(C).view(-1, self.Nv, self.Do))
            del C

        # Taking the sum of over each particle/vertex
        N = torch.sum(Omatrix, dim=1)
        del Omatrix
        if self.vv_branch:
            N_v = torch.sum(O_v, dim=1)
            del O_v

        # Classification MLP
        if self.vv_branch:
            N = self.fc_fixed(torch.cat([N, N_v], 1))
        else:
            N = self.fc_fixed(N)

        if self.softmax:
            N = nn.Softmax(dim=-1)(N)

        return N

    def tmul(self, x, y):  # Takes (I * J * K)(K * L) -> I * J * L
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.reshape(-1, x_shape[2]), y).reshape(-1, x_shape[1], y_shape[1])


class GraphNetEmbedding(nn.Module):
    def __init__(
        self,
        n_constituents,
        n_features,
        fr,
        fo,
        De=5,
        Do=6,
        device="cpu",
    ):
        super(GraphNetEmbedding, self).__init__()
        self.P = n_features
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.De = De
        self.Do = Do
        self.device = device
        self.assign_matrices()
        self.fr = fr
        self.fo = fo

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0] != i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = self.Rr.to(self.device)
        self.Rs = self.Rs.to(self.device)

    def edge_conv(self, x):
        Orr = torch.matmul(x, self.Rr)  # [batch, P, Nr]
        Ors = torch.matmul(x, self.Rs)  # [batch, P, Nr]
        B = torch.cat([Orr, Ors], dim=-2)  # [batch, 2*P, Nr]
        B = B.transpose(-1, -2).contiguous()  # [batch, Nr, 2*P]
        E = self.fr(B.view(-1, 2 * self.P)).view(-1, self.Nr, self.De)  # [batch, Nr, De]
        E = E.transpose(-1, -2).contiguous()  # [batch, De, Nr]
        Ebar_pp = torch.einsum("bij,kj->bik", E, self.Rr)  # [batch, De, N]
        return Ebar_pp

    def forward(self, x):  # [batch, P, N]
        # pf - pf
        Ebar_pp = self.edge_conv(x)  # [batch, De, N]

        # Final output matrix
        C = torch.cat([x, Ebar_pp], dim=-2)  # [batch, P + De, N]
        C = C.transpose(-1, -2).contiguous()  # [batch, N, P + De]
        Omatrix = self.fo(C.view(-1, self.P + self.De)).view(-1, self.N, self.Do)  # [batch, N, Do]
        Omatrix = Omatrix.transpose(-1, -2).contiguous()  # [batch, Do, N]

        # Taking the sum of over each particle/vertex
        N = torch.sum(Omatrix, dim=-1)  # [batch, Do]

        return N
