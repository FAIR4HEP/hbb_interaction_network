import itertools

import torch
import torch.nn as nn


class InteractionNetTagger(nn.Module):
    def __init__(self, pf_dims, sv_dims, num_classes, pf_features_dims, sv_features_dims, hidden, De, Do, **kwargs):
        super().__init__(**kwargs)
        self.P = pf_features_dims
        self.N = pf_dims
        self.S = sv_features_dims
        self.Nv = sv_dims
        self.Nr = self.N * (self.N - 1)
        self.Nt = self.N * self.Nv
        self.De = De
        self.Do = Do
        self.n_targets = num_classes
        self.hidden = hidden
        self.assign_matrices()
        self.assign_matrices_SV()
        self.batchnorm_x = nn.BatchNorm1d(self.P)
        self.batchnorm_y = nn.BatchNorm1d(self.S)

        self.fr = nn.Sequential(
            nn.Linear(2 * self.P, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.De),
            nn.BatchNorm1d(self.De),
            nn.ReLU(),
        )

        self.fr_pv = nn.Sequential(
            nn.Linear(self.S + self.P, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.De),
            nn.BatchNorm1d(self.De),
            nn.ReLU(),
        )

        self.fo = nn.Sequential(
            nn.Linear(self.P + (2 * self.De), self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.Do),
            nn.BatchNorm1d(self.Do),
            nn.ReLU(),
        )

        self.fc_fixed = nn.Linear(self.Do, self.n_targets)

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0] != i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1

    def assign_matrices_SV(self):
        self.Rk = torch.zeros(self.N, self.Nt)
        self.Rv = torch.zeros(self.Nv, self.Nt)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.Nv))]
        for i, (k, v) in enumerate(receiver_sender_list):
            self.Rk[k, i] = 1
            self.Rv[v, i] = 1

    def edge_conv(self, x):
        Orr = torch.matmul(x, self.Rr.to(device=x.device))  # [batch, P, Nr]
        Ors = torch.matmul(x, self.Rs.to(device=x.device))  # [batch, P, Nr]
        B = torch.cat([Orr, Ors], dim=-2)  # [batch, 2*P, Nr]
        B = B.transpose(-1, -2).contiguous()  # [batch, Nr, 2*P]
        E = self.fr(B.view(-1, 2 * self.P)).view(-1, self.Nr, self.De)  # [batch, Nr, De]
        E = E.transpose(-1, -2).contiguous()  # [batch, De, Nr]
        Ebar_pp = torch.einsum("bij,kj->bik", E, self.Rr.to(device=x.device))  # [batch, De, N]
        return Ebar_pp

    def edge_conv_SV(self, x, y):
        Ork = torch.matmul(x, self.Rk.to(device=x.device))  # [batch, P, Nt]
        Orv = torch.matmul(y, self.Rv.to(device=x.device))  # [batch, S, Nt]
        B = torch.cat([Ork, Orv], dim=-2)  # [batch, P+S, Nt]
        B = B.transpose(-1, -2).contiguous()  # [batch, Nt, P+S]
        E = self.fr_pv(B.view(-1, self.P + self.S)).view(-1, self.Nt, self.De)  # [batch, Nt, De]
        E = E.transpose(-1, -2).contiguous()  # [batch, De, Nt]
        Ebar_pv = torch.einsum("bij,kj->bik", E, self.Rk.to(device=x.device))  # [batch, De, N]
        return Ebar_pv

    def forward(self, x, y):
        x = self.batchnorm_x(x)  # [batch, P, N]
        y = self.batchnorm_y(y)  # [batch, S, Nv]

        # pf - pf
        Ebar_pp = self.edge_conv(x)  # [batch, De, N]

        # sv - pf
        Ebar_pv = self.edge_conv_SV(x, y)  # [batch, De, N]

        # Final output matrix
        C = torch.cat([x, Ebar_pp, Ebar_pv], dim=-2)  # [batch, P + 2*De, N]
        C = C.transpose(-1, -2).contiguous()  # [batch, N, P + 2*De]
        Omatrix = self.fo(C.view(-1, self.P + 2 * self.De)).view(-1, self.N, self.Do)  # [batch, N, Do]
        Omatrix = Omatrix.transpose(-1, -2).contiguous()  # [batch, Do, N]

        # Taking the sum of over each particle/vertex
        N = torch.sum(Omatrix, dim=-1)  # [batch, Do]

        # Classification MLP
        N = self.fc_fixed(N)  # [batch, Do]

        return N


class InteractionNetMergedTagger(nn.Module):
    def __init__(
        self, pf_dims, sv_dims, num_classes, pf_features_dims, sv_features_dims, hidden, De, Do, transform_dims, **kwargs
    ):
        super().__init__(**kwargs)
        self.P = pf_features_dims
        self.pf_dims = pf_dims
        self.sv_dims = sv_dims
        self.N = pf_dims + sv_dims
        self.S = sv_features_dims
        self.Nr = self.N * (self.N - 1)
        self.De = De
        self.Do = Do
        self.n_targets = num_classes
        self.hidden = hidden
        self.transform_dims = transform_dims
        self.assign_matrices()

        self.x_transform = nn.Sequential(
            nn.BatchNorm1d(self.P),
            nn.Linear(self.P, self.transform_dims),
            nn.BatchNorm1d(self.transform_dims),
            nn.ReLU(),
        )
        self.y_transform = nn.Sequential(
            nn.BatchNorm1d(self.S),
            nn.Linear(self.S, self.transform_dims),
            nn.BatchNorm1d(self.transform_dims),
            nn.ReLU(),
        )

        self.fr = nn.Sequential(
            nn.Linear(2 * self.transform_dims, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.De),
            nn.BatchNorm1d(self.De),
            nn.ReLU(),
        )

        self.fo = nn.Sequential(
            nn.Linear(self.transform_dims + self.De, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.Do),
            nn.BatchNorm1d(self.Do),
            nn.ReLU(),
        )

        self.fc_fixed = nn.Linear(self.Do, self.n_targets)

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0] != i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1

    def edge_conv(self, x):
        Orr = torch.matmul(x, self.Rr.to(device=x.device))  # [batch, transform_dims, Nr]
        Ors = torch.matmul(x, self.Rs.to(device=x.device))  # [batch, transform_dims, Nr]
        B = torch.cat([Orr, Ors], dim=-2)  # [batch, 2*P, Nr]
        B = B.transpose(-1, -2).contiguous()  # [batch, Nr, 2*P]
        E = self.fr(B.view(-1, 2 * self.P)).view(-1, self.Nr, self.De)  # [batch, Nr, De]
        E = E.transpose(-1, -2).contiguous()  # [batch, De, Nr]
        Ebar_pp = torch.einsum("bij,kj->bik", E, self.Rr.to(device=x.device))  # [batch, De, N]
        return Ebar_pp

    def forward(self, x, y):
        # x: [batch, P, pf_dims]
        # y: (batch, S, sv_dims]
        x = x.transpose(-1, -2).contiguous()  # [batch, pf_dims, P]
        y = y.transpose(-1, -2).contiguous()  # [batch, sv_dims, S]
        x = self.x_transform(x.view(-1, self.P)).view(-1, self.pf_dims, self.transform_dims)  # [batch, N, transform_dims]
        y = self.y_transform(y.view(-1, self.S)).view(-1, self.sv_dims, self.transform_dims)  # [batch, Nv, transform_dims]
        x = x.transpose(-1, -2).contiguous()  # [batch, transform_dims, pf_dims]
        y = y.transpose(-1, -2).contiguous()  # [batch, transform_dims, sv_dims]

        x = torch.cat((x, y), axis=-1)  # [batch, transform_dims, N]

        # pf - pf
        Ebar_pp = self.edge_conv(x)  # [batch, De, N]

        # Final output matrix
        C = torch.cat([x, Ebar_pp], dim=-2)  # [batch, transform_dims + De, N]
        C = C.transpose(-1, -2).contiguous()  # [batch, N + Nv, transform_dims + De]
        Omatrix = self.fo(C.view(-1, self.P + 2 * self.De)).view(-1, self.N, self.Do)  # [batch, N, Do]
        Omatrix = Omatrix.transpose(-1, -2).contiguous()  # [batch, Do, N]

        # Taking the sum of over each particle/vertex
        N = torch.sum(Omatrix, dim=-1)  # [batch, Do]

        # Classification MLP
        N = self.fc_fixed(N)  # [batch, Do]

        return N


class InteractionNetSingleTagger(nn.Module):
    def __init__(self, dims, num_classes, features_dims, hidden, De, Do, **kwargs):
        super().__init__(**kwargs)
        self.P = features_dims
        self.N = dims
        self.Nr = self.N * (self.N - 1)
        self.De = De
        self.Do = Do
        self.n_targets = num_classes
        self.hidden = hidden
        self.assign_matrices()
        self.batchnorm_x = nn.BatchNorm1d(self.P)

        self.fr = nn.Sequential(
            nn.Linear(2 * self.P, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.De),
            nn.BatchNorm1d(self.De),
            nn.ReLU(),
        )

        self.fo = nn.Sequential(
            nn.Linear(self.P + self.De, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.Do),
            nn.BatchNorm1d(self.Do),
            nn.ReLU(),
        )

        self.fc_fixed = nn.Linear(self.Do, self.n_targets)

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0] != i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1

    def edge_conv(self, x):
        Orr = torch.matmul(x, self.Rr.to(device=x.device))  # [batch, P, Nr]
        Ors = torch.matmul(x, self.Rs.to(device=x.device))  # [batch, P, Nr]
        B = torch.cat([Orr, Ors], dim=-2)  # [batch, 2*P, Nr]
        B = B.transpose(-1, -2).contiguous()  # [batch, Nr, 2*P]
        E = self.fr(B.view(-1, 2 * self.P)).view(-1, self.Nr, self.De)  # [batch, Nr, De]
        E = E.transpose(-1, -2).contiguous()  # [batch, De, Nr]
        Ebar_pp = torch.einsum("bij,kj->bik", E, self.Rr.to(device=x.device))  # [batch, De, N]
        return Ebar_pp

    def forward(self, x):
        x = self.batchnorm_x(x)  # [batch, P, N]

        # pf - pf
        Ebar_pp = self.edge_conv(x)  # [batch, De, N]

        # Final output matrix
        C = torch.cat([x, Ebar_pp], dim=-2)  # [batch, P + De, N]
        C = C.transpose(-1, -2).contiguous()  # [batch, N, P + De]
        Omatrix = self.fo(C.view(-1, self.P + 2 * self.De)).view(-1, self.N, self.Do)  # [batch, N, Do]
        Omatrix = Omatrix.transpose(-1, -2).contiguous()  # [batch, Do, N]

        # Taking the sum of over each particle/vertex
        N = torch.sum(Omatrix, dim=-1)  # [batch, Do]

        # Classification MLP
        N = self.fc_fixed(N)  # [batch, Do]

        return N


class InteractionNetTaggerEmbedding(nn.Module):
    def __init__(
        self,
        dims,
        features_dims,
        fr,
        fo,
        De,
        Do,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.P = features_dims
        self.N = dims
        self.Nr = self.N * (self.N - 1)
        self.De = De
        self.Do = Do
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
