import torch
import torch.nn as nn
from ltn.fuzzy_ops import BinaryConnectiveOperator, pi_0, pi_1
import math


class ImpliesReichenbachSigmoidal(BinaryConnectiveOperator):
    def __init__(self, s, stable=True):
        # b = -0.5
        self.s = s
        self.stable = stable
        self.sigmoid = nn.Sigmoid()
        self.pre_comp_factor1 = 1 + math.exp(self.s / 2)
        self.pre_comp_factor2 = math.exp(self.s / 2) - 1

    def __repr__(self):
        return "ImpliesReichenbachSigmoidal()"

    def __call__(self, x, y):
        # compute Reichenbach
        if self.stable:
            x, y = pi_0(x), pi_1(y)
        I_out = 1.0 - x + torch.mul(x, y)

        sigm_out = (
            self.pre_comp_factor1 * self.sigmoid(self.s * (I_out - 1 / 2)) - 1
        ) / self.pre_comp_factor2
        return sigm_out
