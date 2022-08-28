""" Adapted from ODE-LSTM https://github.com/mlech26l/ode-lstms/. """
# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import torch
import torch.nn as nn
from torchdyn.models import NeuralDE
import pytorch_lightning as pl
from torchmetrics.functional import accuracy


class mmRNNCell(nn.Module):
    def __init__(self, d_model, d_hidden,solver_type):
        super(mmRNNCell, self).__init__()
        self.solver_type = solver_type
        self.lstm = nn.LSTMCell(d_model, d_hidden)
        # 1 hidden layer NODE
        self.f_node = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.d_model = d_model
        self.d_hidden = d_hidden
        options = {
            "fixed_euler": self.euler,
            "fixed_heun": self.heun,
            "fixed_rk4": self.rk4,
        }
        if not solver_type in options.keys():
            raise ValueError("Unknown solver type '{:}'".format(solver_type))
        self.node = options[self.solver_type]

    def forward(self, input, hx, ts):
        new_h, new_c = self.lstm(input, hx)
        new_h = self.solve_fixed(new_h, ts)
        return (new_h, new_c)

    def solve_fixed(self, x, ts):
        ts = ts.view(-1, 1)
        for i in range(3):  # 3 unfolds
            x = self.node(x, ts * (1.0 / 3))
        return x

    def euler(self, y, delta_t):
        dy = self.f_node(y)
        return y + delta_t * dy

    def heun(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + delta_t * k1)
        return y + delta_t * 0.5 * (k1 + k2)

    def rk4(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + k1 * delta_t * 0.5)
        k3 = self.f_node(y + k2 * delta_t * 0.5)
        k4 = self.f_node(y + k3 * delta_t)

        return y + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


class mmRNN(nn.Module):
    def __init__(
        self,
        d_model,
        d_output=None,
        d_hidden=None,
        return_sequences=True,
        solver_type="heun",
        **kwargs,
    ):
        super(mmRNN, self).__init__()
        d_output = d_output or d_model
        d_hidden = d_hidden or d_model
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.return_sequences = return_sequences

        self.rnn_cell = mmRNN(d_model, d_hidden, solver_type=solver_type)
        self.fc = nn.Linear(self.d_hidden, self.d_output)

    def forward(self, x, timespans=None):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_state = [
            torch.zeros((batch_size, self.d_hidden), device=device),
            torch.zeros((batch_size, self.d_hidden), device=device),
        ]
        outputs = []
        last_output = torch.zeros((batch_size, self.d_output), device=device)

        if timespans is None:
            timespans = x.new_ones(x.shape[:-1]+(1,)) / x.shape[1]

        for t in range(seq_len):
            inputs = x[:, t]
            ts = timespans[:, t].squeeze()
            hidden_state = self.rnn_cell.forward(inputs, hidden_state, ts)
            current_output = self.fc(hidden_state[0])
            outputs.append(current_output)
            last_output = current_output
        if self.return_sequences:
            outputs = torch.stack(outputs, dim=1)  # return entire sequence
        else:
            outputs = last_output  # only last item
        return outputs, hidden_state