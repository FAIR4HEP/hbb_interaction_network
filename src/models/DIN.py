import itertools

import torch
import torch.nn as nn

### an interaction layer that learnss a representation from input and interactions for each node node
class interaction(nn.Module):
    def __init__(
        self,
        n_p,
        n_v,
        params_p,
        params_v,
        hidden,
        n_pout,
        n_vout,
        device="cpu",
    }:
        super(interaction, self).__init__()

        ### input
        self.P = params_p
        self.N = n_p
        self.S = params_v
        self.Nv = n_v

        ### interpreted from input

        self.Nr = self.N*(self.N-1)
        self.Nt = self.N*self.Nv
        self.Ns = self.S*(self.N-1)

        ### hidden
        self.hidden = int(hidden)

        ### representation
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do

        ### target
        self.n_pout = n_pout
        self.n_vout = n_vout

        ### device
        self.device = device

        ### initialize graph
        self.


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
            # Final output matrix for particles###
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
