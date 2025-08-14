import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
    x_shift = [
        torch.roll(x_c, shift, dim)
        for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))
    ]
    x_cat = torch.cat(x_shift, 1)
    x_cat = torch.narrow(x_cat, 2, self.pad, H)
    x_cat = torch.narrow(x_cat, 3, self.pad, W)
    return x_cat


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def swish(x):

    return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.main(x)
        return x


class kan(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = Swish
        grid_eps = 0.02
        grid_range = [-1, 1]

        self.fc1 = KANLinear(
            in_features,
            hidden_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()

        return x


class shiftedBlock(nn.Module):
    def __init__(
        self, dim, mlp_ratio=4.0, drop_path=0.0, norm_layer=nn.LayerNorm, tdim=256
    ):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, dim),
        )

        self.kan = kan(in_features=dim, hidden_features=mlp_hidden_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, temb):

        temb = self.temb_proj(temb)
        x = self.drop_path(self.kan(self.norm2(x), H, W))
        x = x + temb.unsqueeze(1)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.GroupNorm(32, dim)
        # self.relu = Swish()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = swish(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class SingleConv(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
        )

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )

    def forward(self, input, temb):
        return self.conv(input) + self.temb_proj(temb)[:, :, None, None]


class DoubleConv(nn.Module):
    def __init__(self, in_ch, h_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
            nn.GroupNorm(32, h_ch),
            Swish(),
            nn.Conv2d(h_ch, h_ch, 3, padding=1),
            nn.GroupNorm(32, h_ch),
            Swish(),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, h_ch),
        )

    def forward(self, input, temb):
        return self.conv(input) + self.temb_proj(temb)[:, :, None, None]


class D_SingleConv(nn.Module):
    def __init__(self, in_ch, h_ch, tdim=256):
        super(D_SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, h_ch),
        )

    def forward(self, input, temb):
        return self.conv(input) + self.temb_proj(temb)[:, :, None, None]


class D_DoubleConv(nn.Module):
    def __init__(self, in_ch, h_ch, tdim=256):
        super(D_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
            nn.GroupNorm(32, h_ch),
            Swish(),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, h_ch),
        )

    def forward(self, input, temb):
        return self.conv(input) + self.temb_proj(temb)[:, :, None, None]


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, h_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, h_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, h_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, h_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(h_ch, h_ch, 3, stride=1, padding=1),
        )
        if in_ch != h_ch:
            self.shortcut = nn.Conv2d(in_ch, h_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(h_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UKan_Hybrid(nn.Module):
    def __init__(
        self,
        in_channels,
        T,
        ch,
        ch_mult,
        attn,
        num_res_blocks,
        dropout,
        sample_size: int = 16,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.ch_mult = ch_mult
        assert all([i < len(ch_mult) for i in attn]), "attn index h of bound"
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        attn = []
        self.head = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record input channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            h_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(
                        in_ch=now_ch,
                        h_ch=h_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn),
                    )
                )
                now_ch = h_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        # Dynamic calculation of embed_dims based on channel structure
        deepest_ch = ch * ch_mult[-1]

        # Create embedding dimensions that scale with the model size
        embed_dims = [
            deepest_ch,  # First KAN level - same as deepest UNet
            deepest_ch * 3 // 2,  # Second level - 1.5x deeper channels
            deepest_ch * 2,  # Third level - 2x deeper channels
        ]

        # Store for potential lazy initialization
        self.embed_dims = embed_dims
        self.tdim = tdim

        # Flag to track if KAN blocks should be used
        self.use_kan_blocks = None  # Will be determined during first forward pass
        self.kan_components_initialized = False

        # UNet upsampling part - this is always used
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            h_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(
                        in_ch=chs.pop() + now_ch,
                        h_ch=h_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn),
                    )
                )
                now_ch = h_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, in_channels, 3, stride=1, padding=1),
        )

        self.initialize()
        self._init_kan_components()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def _init_kan_components(self):
        """Initialize KAN components based on feature map size"""
        H, W = self.sample_size // 2 ** len(self.ch_mult), self.sample_size // 2 ** len(
            self.ch_mult
        )
        # Check if feature map is large enough for KAN blocks
        # Need at least 8x8 to support two levels of KAN blocks with stride 2
        if H < 4 or W < 4:
            print(
                f"Feature map size {H}x{W} is too small for KAN blocks. Using UNet backbone only."
            )
            self.use_kan_blocks = False
            return

        self.use_kan_blocks = True
        norm_layer = nn.LayerNorm
        dpr = [0.0, 0.0, 0.0]

        # Initialize first level KAN components
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=H,
            patch_size=3,
            stride=2,
            in_chans=self.embed_dims[0],
            embed_dim=self.embed_dims[1],
        )

        self.kan_block1 = nn.ModuleList(
            [
                shiftedBlock(
                    dim=self.embed_dims[1],
                    mlp_ratio=1,
                    drop_path=dpr[0],
                    norm_layer=norm_layer,
                    tdim=self.tdim,
                )
            ]
        )

        self.norm3 = norm_layer(self.embed_dims[1])

        # Calculate size after first patch embedding (approximately half with stride=2)
        H2, W2 = H // 2, W // 2

        # Initialize second level KAN components
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=H2,
            patch_size=3,
            stride=2,
            in_chans=self.embed_dims[1],
            embed_dim=self.embed_dims[2],
        )

        self.kan_block2 = nn.ModuleList(
            [
                shiftedBlock(
                    dim=self.embed_dims[2],
                    mlp_ratio=1,
                    drop_path=dpr[1],
                    norm_layer=norm_layer,
                    tdim=self.tdim,
                )
            ]
        )

        self.norm4 = norm_layer(self.embed_dims[2])

        # Initialize decoder components
        self.decoder1 = D_SingleConv(
            self.embed_dims[2], self.embed_dims[1], tdim=self.tdim
        )

        self.kan_dblock1 = nn.ModuleList(
            [
                shiftedBlock(
                    dim=self.embed_dims[1],
                    mlp_ratio=1,
                    drop_path=dpr[0],
                    norm_layer=norm_layer,
                    tdim=self.tdim,
                )
            ]
        )

        self.dnorm3 = norm_layer(self.embed_dims[1])

        self.decoder2 = D_SingleConv(
            self.embed_dims[1], self.embed_dims[0], tdim=self.tdim
        )

        self.kan_components_initialized = True

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)

        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        # Process with KAN blocks if possible
        if not self.kan_components_initialized:
            self._init_kan_components(h)

        if self.use_kan_blocks:
            # Store the output of UNet encoder for skip connection
            t3 = h

            # First patch embedding and KAN block
            B = h.shape[0]
            h, H, W = self.patch_embed3(h)
            for i, blk in enumerate(self.kan_block1):
                h = blk(h, H, W, temb)
            h = self.norm3(h)
            h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            # Store output for skip connection
            t4 = h

            # Second patch embedding and KAN block
            h, H, W = self.patch_embed4(h)
            for i, blk in enumerate(self.kan_block2):
                h = blk(h, H, W, temb)
            h = self.norm4(h)
            h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            # Stage 4 - Upsample and add skip connection
            h = self.decoder1(h, temb)
            h = swish(F.interpolate(h, size=t4.shape[2:], mode="bilinear"))
            h = torch.add(h, t4)

            # KAN decoder block
            _, _, H, W = h.shape
            h = h.flatten(2).transpose(1, 2)
            for i, blk in enumerate(self.kan_dblock1):
                h = blk(h, H, W, temb)

            # Stage 3 - Upsample and add skip connection
            h = self.dnorm3(h)
            h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            h = self.decoder2(h, temb)
            h = swish(F.interpolate(h, size=t3.shape[2:], mode="bilinear"))
            h = torch.add(h, t3)

        # Upsampling - UNet decoder
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == "__main__":
    import sys

    # import os
    sys.path.append("/home/march/masters-degree/thesis-prep/wavelet-diffusion")
    from wavelet.dwt import DiscreteWaveletTransform

    batch_size = 32
    model = UKan_Hybrid(
        in_channels=12,
        T=1000,
        ch=64,
        ch_mult=[1, 2, 2, 2, 4],
        attn=[],
        num_res_blocks=2,
        dropout=0.1,
    ).to("cuda")
    encdec = DiscreteWaveletTransform(level=1)
    x = torch.randn(batch_size, 3, 256, 256).to("cuda")
    enc = encdec.encode(x)
    t = torch.randint(0, 1000, (batch_size,)).to("cuda")
    out = model(enc, t)
    print(out.shape)  # Expected output shape: (batch_size, 3, 64, 64)
