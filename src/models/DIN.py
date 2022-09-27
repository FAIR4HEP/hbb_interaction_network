import itertools

import torch
import torch.nn as nn

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
        self.hidden = int(hidden)
        self.P = params_p



