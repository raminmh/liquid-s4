""" Adapted from ODE-LSTM https://github.com/mlech26l/ode-lstms/. """
# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import torch
import torch.nn as nn

class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)

class mmRNNCell(nn.Module):
    def __init__(self, d_model, d_hidden,solver_type):
        super(mmRNNCell, self).__init__()
        self.solver_type = solver_type
        self.lstm = nn.LSTMCell(d_model, d_hidden)
        # 1 hidden layer NODE
        self.f_node = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            LeCun(),
            nn.Linear(d_hidden, d_hidden),
            LeCun(),
        )
        self.d_model = d_model
        self.d_hidden = d_hidden
        options = {
            "euler": self.euler,
            "heun": self.heun,
            "rk4": self.rk4,
        }
        if not solver_type in options.keys():
            raise ValueError("Unknown solver type '{:}'".format(solver_type))
        self.node = options[self.solver_type]

    def forward(self, input, hx, ts):
        new_h, new_c = self.lstm(input, hx)
        new_h = self.solve_fixed(new_h, ts)
        return (new_h, new_c)

    def solve_fixed(self, x, rate):
        N = 3
        for i in range(N):  # 3 unfolds
            x = self.node(x, rate * (1.0 / N))
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
        solver_type="euler",
        **kwargs,
    ):
        super(mmRNN, self).__init__()
        d_output = d_output or d_model
        d_hidden = d_hidden or d_model
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.return_sequences = return_sequences

        self.rnn_cell = mmRNNCell(d_model, d_hidden, solver_type=solver_type)
        self.fc = nn.Linear(self.d_hidden, self.d_output)

    # def forward(self, x, state=None,lengths=None,timespans=None):
    def forward(self, u, state=None, rate=1.0, lengths=None, **kwargs):
        if state is not None:
            print("state is not None -> breakpoint")
            breakpoint()
        # if lengths is not None:
        #     print("lengths is not None -> breakpoint")
        #     breakpoint()
        #
        L = u.size(-1)
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, u.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device) < lengths[:, None, None], 1., 0.)
            u = u * mask

        device = u.device
        batch_size = u.size(0)
        seq_len = u.size(1)
        hidden_state = [
            torch.zeros((batch_size, self.d_hidden), device=device),
            torch.zeros((batch_size, self.d_hidden), device=device),
        ]
        outputs = []
        last_output = torch.zeros((batch_size, self.d_output), device=device)

        for t in range(seq_len):
            inputs = u[:, t]
            hidden_state = self.rnn_cell.forward(inputs, hidden_state, rate)
            current_output = self.fc(hidden_state[0])
            outputs.append(current_output)
            last_output = current_output
        # if self.return_sequences:
        #     outputs = torch.stack(outputs, dim=1)  # return entire sequence
        # else:
        #     outputs = last_output  # only last item
        outputs = torch.stack(outputs, dim=1)  # return entire sequence
        return outputs, hidden_state