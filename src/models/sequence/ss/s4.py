import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from functools import partial
from einops import rearrange, repeat
import opt_einsum as oe
import itertools

optimized = True

if optimized:
    contract = oe.contract
else:
    contract = torch.einsum

from src.models.sequence.ss.kernel import SSKernel, _conj
from src.models.nn import LinearActivation, Activation, DropoutNd

class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class S4(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=None,
            channels=1,
            bidirectional=False,
            # Arguments for position-wise feedforward components
            activation='gelu',
            postact='glu',
            initializer=None,
            weight_norm=False,
            hyper_act=None,
            dropout=0.0, tie_dropout=False,
            bottleneck=None,
            gate=None,
            transposed=True,
            verbose=False,
            shift=False,
            linear=False,
            liquid_kernel=None,
            liquid_degree=2,
            allcombs=True,
            lcontract=None,
            # SSM Kernel arguments
            **kernel_args,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum kernel length, also denoted by L. Set l_max=None to always use a global kernel
        channels: can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this unless desperate for things to tune; instead, increase d_model for larger models
        bidirectional: if True, convolution kernel will be two-sided

        Position-wise feedforward components:
        --------------------
        activation: activation in between SS and FF
        postact: activation after FF
        initializer: initializer on FF
        weight_norm: weight normalization on FF
        hyper_act: use a "hypernetwork" multiplication (experimental)
        dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

        Other arguments:
        --------------------
        transposed: choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=hidden dimension]
        gate: add gated activation (GSS)
        bottleneck: reduce SSM dimension (GSS)
        shift: experimental option, shouldn't affect results
        linear: Remove pointwise components so that the entire module is a linear SSM

        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        import src.utils.train
        log = src.utils.train.get_logger(__name__)
        if verbose:
            log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")
            log = src.utils.train.get_logger(__name__)
        if liquid_degree <= 1:
            raise ValueError(f"Illegal argument for liquid_degree ({liquid_degree}). Valid options are >= 2")
        if liquid_kernel is not None:
            log.info(f"Constructing liquid-S4 with liquid kernel '{liquid_kernel}' and degree {liquid_degree} (allcombs={allcombs})")
        else:
            log.info(
                f"Using plain S4 (to enable liquid-S4 run with model.layer.liquid_degree='polyb'|'kb')"
            )
        if liquid_kernel not in [None, "polyb","kb"]:
            raise ValueError(f"Invalid argument for liquid_kernel ('{liquid_kernel}'). Use 'polyb', 'kb'")
        self.liquid_kernel = liquid_kernel
        self.liquid_degree = liquid_degree

        self.d_model = d_model
        self.H = d_model
        self.N = d_state
        self.L = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.shift = shift
        self.linear = linear
        self.linear = linear
        if lcontract=="lecun":
            self.lcontract = LeCun()
        elif lcontract=="tanh":
            self.lcontract = nn.Tanh()
        else:
            self.lcontract = nn.Identity()


        self.allcombs = allcombs

        self.gate = gate
        self.bottleneck = bottleneck

        if bottleneck is not None:
            self.H = self.H // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.H,
                transposed=self.transposed,
                initializer=initializer,
                activation=activation,
                activate=True,
                weight_norm=weight_norm,
            )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                transposed=self.transposed,
                initializer=initializer,
                activation=activation,
                activate=True,
                weight_norm=weight_norm,
            )
            self.output_gate = LinearActivation(
                self.d_model * gate,
                self.d_model,
                transposed=self.transposed,
                initializer=initializer,
                activation=None,
                activate=False,
                weight_norm=weight_norm,
            )

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.H))

        if self.bidirectional:
            channels *= 2


        # SSM Kernel
        self.kernel = SSKernel(self.H, N=self.N, L=self.L, channels=channels, verbose=verbose, **kernel_args)
        log.info(f"Using S4 kernel {self.kernel.mode}")
        # Pointwise
        if not self.linear:
            self.activation = Activation(activation)
            # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
            dropout_fn = DropoutNd if tie_dropout else nn.Dropout
            self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        # position-wise output transform to mix features
        if not self.linear:
            self.output_linear = LinearActivation(
                self.H*self.channels,
                self.d_model*(1 if self.gate is None else self.gate),
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )
        self._allcombs_index_cache = None
        self._allcombs_cache_L = None

    def get_combs_cache(self,seq_len, i ):
        if self._allcombs_cache_L != seq_len:
            self._allcombs_index_cache = []
            self._allcombs_cache_L = seq_len
            for p in range(2, self.liquid_degree + 1):
                selected_count = 1
                for n in range(2, seq_len):
                    count = math.comb(n, p)
                    if count >= seq_len:
                        selected_count = n
                        break
                indices = range(seq_len - selected_count, seq_len)
                indices = list(itertools.combinations(indices, p))
                # print(f"p={p}, seq_len={seq_len}, selected_count={selected_count}",)
                # print(f"{len(indices)=}")
                if len(indices) != seq_len:
                    # select exactly amount to match sequence length dimension
                    indices = indices[-seq_len:]
                indices = torch.LongTensor(indices)
                self._allcombs_index_cache.append((p, indices))
        return self._allcombs_index_cache[i]

    def upgrade_degree(self,us,u,i):
        seq_len = u.size(-1)
        if self.allcombs:
            p, indices = self.get_combs_cache(seq_len,i)
            us = u[..., indices[:, 0]]
            for j in range(1, p):
                us = us * u[..., indices[:, j]]
            if us.size(-1) != u.size(-1):
                us = F.pad(us, (0, u.size(-1) - us.size(-1)))
        else:
            us_shift = torch.nn.functional.pad(us[..., :-1], (1, 0), "constant", 0)
            us = us * us_shift
        return us


    def forward(self, u, state=None, rate=1.0, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Mask out padding tokens
        # TODO handle option for mask - instead of lengths, which assumes suffix padding
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=u.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, u.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device) < lengths[:, None, None], 1., 0.)
            u = u * mask

        if self.gate is not None:
            v = self.input_gate(u)
        if self.bottleneck is not None:
            u = self.input_linear(u)

        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state = self.kernel(L=L_kernel, rate=rate, state=state) # (C H L) (B C H L)

        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0)) \

        # Convolution
        if self.shift:
            # Try flip and pad to correct for potential off-by-one
            k_f = torch.fft.rfft(F.pad(k.flip(-1), (L, 0)), n=2*L) # (C H L)
            u_f = torch.fft.rfft(F.pad(u.flip(-1), (L, 0)), n=2*L) # (B H L)
            y_f = contract('bhl,chl->bchl', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
            y = torch.fft.irfft(y_f, n=L_kernel+L)[..., L:].flip(-1) # (B C H L)
            if self.liquid_kernel == "kb":
                raise NotImplementedError()
        else:
            k_f = torch.fft.rfft(k, n=L_kernel+L) # (C H L)
            u_f = torch.fft.rfft(u, n=L_kernel+L) # (B H L)
            y_f = contract('bhl,chl->bchl', u_f, k_f)

            y_sum = y_f
            if self.liquid_kernel == "kb":
                # Approximates the liquid kernel in the fourier space by the product of K-bar and B
                k_b = k
                us = u
                dt = torch.exp(self.kernel.log_dt.to(u.device))
                B = _conj(self.kernel.B).to(u.device)
                w = _conj(self.kernel.w).to(u.device)
                dB = torch.diag_embed(1.0 / (1.0 - 0.5 * dt[:, None] * w))  # (256,64,64)
                dB = dt[:, None] * contract("dab,db->da", dB, B)
                dB = dB.unsqueeze(0).unsqueeze(-1)

                for i in range(self.liquid_degree-1):
                    us = self.upgrade_degree(us,u,i)
                    k_b = k_b.unsqueeze(2)
                    k_b = contract('abcd,abcd->abd', k_b, dB) # Kbar times B
                    u_corr = self.lcontract(us)
                    u_corr = u_corr.flip(dims=[-1])
                    k_b_f = torch.fft.fft(k_b, n=(L_kernel + L)//2 + 1)  # complex FFT
                    u_corr_f = torch.fft.rfft(u_corr, n=L_kernel + L)  # real-valued FFT
                    y_corr_f = contract('bhl,chl->bchl', u_corr_f, k_b_f) # Convolution (Multiplication in FFT domain)
                    y_sum = y_sum + y_corr_f

            y = torch.fft.irfft(y_sum, n=L_kernel+L)[..., :L] # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        if self.liquid_kernel == "polyb":
            # Approximates the liquid kernel by computing only the polynomials involving B up to a certain degree
            dt = torch.exp(self.kernel.log_dt.to(u.device))
            B = _conj(self.kernel.B).to(u.device)
            dC = _conj(self.kernel.C).to(u.device)
            w = _conj(self.kernel.w).to(u.device)
            dB = torch.diag_embed(1.0 / (1.0 - 0.5 * dt[:, None] * w))  #  (256,64,64)

            dB = dt[:, None] * contract("dab,db->da", dB, B)
            us = u
            for i in range(self.liquid_degree-1):
                # print(f"[Liquid={self.liquid}] Generating degree {i+1} input polynomial")
                us = self.upgrade_degree(us, u, i)
                u_corr = self.lcontract(us)
                us_corr = torch.flip(u_corr,[-1])
                dB1 = dB.unsqueeze(2)
                dB2 = dB.unsqueeze(1)
                dB = (dB1 * dB2).sum(2)
                dCB = contract("abc,bc->ab", dC, dB).unsqueeze(2)
                if self.bidirectional:
                    fwd, bwd = dCB.unbind(0)
                    fwd, bwd = fwd.unsqueeze(0), bwd.unsqueeze(0)
                    y = (
                        y
                        + (us_corr * fwd).unsqueeze(1).float()
                        + (us_corr.flip(2) * bwd).unsqueeze(1).float()
                    )
                else:

                    y = y + (us_corr * dCB).unsqueeze(1).float()

        # Compute state update
        if state is not None:
            assert not self.bidirectional, "Bidirectional not supported with state forwarding"
            y = y + k_state #
            next_state = self.kernel.forward_state(u, state)
        else:
            next_state = None

        # Optional hyper-network multiplication
        if self.hyper:
            y, yh = rearrange(y, 'b (s c) h l -> s b c h l', s=2)
            y = self.hyper_activation(yh) * y

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        if not self.linear:
            y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        if not self.linear:
            y = self.output_linear(y)

        if self.gate is not None:
            y = self.output_gate(y * v)

        return y, next_state

    def setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state) # (B C H)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, 'b c h -> b (c h)')
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.H * self.N

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)