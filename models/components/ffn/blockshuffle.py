"""
Arxiv paper: https://arxiv.org/html/2406.16450v1
Implementation based on: https://github.com/CLAIRE-Labo/StructuredFFN/tree/main
"""

import torch.nn as nn
import torch

from models.components.utils.feedforward_utils import LinearTempDecay, CosineTempDecay

import numpy as np
from einops import rearrange

class BasicLinear(nn.Module):

    def __init__(
        self, in_features, out_features, bias, return_bias, config, init_config, device
    ):
        super().__init__()
        # config: method part, and model init
        self.device = device
        self.config = config
        self.init_config = init_config
        self.training_config = self.config.training
        # model part
        self.in_features = in_features
        self.out_features = out_features
        # otherwise, we need to fuse the bias into the ops
        assert return_bias is True
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, device=device))
        else:
            self.bias = None

        if self.training_config.enabled:
            self.guide_linear = nn.Parameter(
                torch.empty(self.out_features, self.in_features, device=device)
            )
            self.register_buffer("count", torch.tensor(0).cuda(), persistent=True)
            self.register_buffer("ratio", torch.tensor(1.0).cuda(), persistent=True)
            guide_scheduler = {
                "linear": LinearTempDecay,
                "cosine": CosineTempDecay,
            }
            self.guide_scheduler = guide_scheduler[self.training_config.scheduler](
                t_max=self.training_config.max_step
            )

    @torch.no_grad()
    def _update_ratio(
        self,
    ):
        self.count += 1
        self.ratio = self.guide_scheduler(self.count)

    def _check_guide_layer(
        self,
    ):
        if not self.training_config.enabled:
            return False
        if (
            self.training_config.reduce_flop
            and torch.rand_like(self.ratio) >= self.ratio
        ):
            return False
        return True

    def forward_guide_layer(self, input, out):
        if self._check_guide_layer():
            guide_out = torch.matmul(input, self.guide_linear.transpose(-1, -2))
            out = self.ratio * guide_out + (1.0 - self.ratio) * out
        return out

    def get_weights(
        self,
    ):
        pass

    @torch.no_grad()
    def _init_weights(
        self,
    ):
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        for para in self.get_weights():
            if self.init_config.weight_init == "xavier":
                nn.init.normal_(para, mean=0.0, std=(para.shape[-1] ** -0.5))
            elif self.init_config.weight_init == "fixed":
                nn.init.normal_(para, std=self.init_config.initializer_range)
            else:
                raise NotImplementedError


class BlockShuffleCustom(torch.autograd.Function):
    # Paste from monarch repo
    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, x, w1_bfly, w2_bfly):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        assert k * p == n
        assert l * r == k * q
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(
            0, 1
        )
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
        out1 = (
            out1.transpose(0, 1)
            .reshape(batch_dim, r, l)
            .transpose(-1, -2)
            .contiguous()
            .transpose(0, 1)
        )
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(
            0, 1
        )
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        # assert k * p == n
        # assert l * r == k * q
        dx, dw1_bfly, dw2_bfly = None, None, None
        # dout_reshaped = dout.reshape(batch_dim, sqrtn, sqrtn).permute(2, 1, 0).contiguous()
        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous()
        dout_reshaped = dout_reshaped.transpose(0, 1)
        if ctx.needs_input_grad[2]:
            # dw2_bfly = torch.empty(l, s, r, device=w2_bfly.device, dtype=w2_bfly.dtype)
            # dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1, out=dw2_bfly)
            dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.empty(
                batch_dim, l, r, device=x.device, dtype=x.dtype
            ).transpose(0, 1)
            dout1 = torch.bmm(dout_reshaped, w2_bfly, out=dout1)
            dout1 = (
                dout1.transpose(0, 1)
                .transpose(-1, -2)
                .contiguous()
                .reshape(batch_dim, k, q)
                .transpose(0, 1)
            )
            # dout1 = dout1.permute(1, 2, 0).contiguous().transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.empty(batch_dim, k, p, device=x.device, dtype=x.dtype)
                dx = (
                    torch.bmm(dout1, w1_bfly, out=dx.transpose(0, 1))
                    .transpose(0, 1)
                    .reshape(*batch_shape, n)
                )
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped)
        return dx, dw1_bfly, dw2_bfly


block_shuffle_custom = BlockShuffleCustom.apply



class BlockShuffleLayer(BasicLinear):

    def __init__(
        self,
        in_features,
        out_features,
        bias,
        return_bias,
        config,
        init_config,
        device="cuda",
    ):
        super().__init__(
            in_features, out_features, bias, return_bias, config, init_config, device
        )
        self.nblocks = config["nblocks"]
        assert self.in_features % self.nblocks == 0
        assert self.out_features % self.nblocks == 0

        in_blksz = self.in_features // self.nblocks
        out_blksz = self.out_features // self.nblocks

        if self.in_features < self.out_features:
            self.blkdiag1 = nn.Parameter(
                torch.empty(self.nblocks, in_blksz, in_blksz, device=device)
            )
            self.blkdiag2 = nn.Parameter(
                torch.empty(self.nblocks, out_blksz, in_blksz, device=device)
            )
        else:
            self.blkdiag1 = nn.Parameter(
                torch.empty(self.nblocks, out_blksz, in_blksz, device=device)
            )
            self.blkdiag2 = nn.Parameter(
                torch.empty(self.nblocks, out_blksz, out_blksz, device=device)
            )
        self._init_weights()
        self.post_init()

    def get_weights(
        self,
    ):
        return [self.blkdiag1, self.blkdiag2]

    @torch.no_grad()
    def post_init(
        self,
    ):
        if self.config.init.post_init == "ortho":
            for i in range(self.nblocks):
                U, S, Vh = torch.linalg.svd(self.blkdiag1.data[i], full_matrices=False)
                self.blkdiag1.data[i] = torch.mm(U, Vh)
                U, S, Vh = torch.linalg.svd(self.blkdiag2.data[i], full_matrices=False)
                self.blkdiag2.data[i] = torch.mm(U, Vh)

        # init guide linear
        if hasattr(self, "guide_linear"):
            self.guide_linear.data = torch.mm(
                torch.block_diag(*torch.unbind(self.blkdiag2.data, dim=0)),
                torch.block_diag(*torch.unbind(self.blkdiag1.data, dim=0)),
            )

    def forward(self, input):
        out = block_shuffle_custom(input, self.blkdiag1, self.blkdiag2)
        return self.forward_guide_layer(input, out)

    def extra_repr(self) -> str:
        return f"blockdiag1={self.blkdiag1.shape}, blockdiag2={self.blkdiag2.shape}, bias={self.bias is not None}, guide={self.training_config.enabled}"