import torch
import torch.nn as nn
import math
import pywt
import torch.nn.functional as F


class DWT2D:
    def __init__(self, in_channels=3):
        # super(DWT2D, self).__init__()

        # # Define Haar wavelet filters
        hw = pywt.Wavelet("haar")
        dec_hi = torch.tensor(hw.dec_hi)
        dec_lo = torch.tensor(hw.dec_lo)
        rec_hi = torch.tensor(hw.rec_hi)
        rec_lo = torch.tensor(hw.rec_lo)
        self.filters = torch.stack(
            [
                dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
            ],
            dim=0,
        )

        self.inv_filters = torch.stack(
            [
                rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
            ],
            dim=0,
        )

        # # Combine filters into a single kernel tensor
        # self.filters = self.filters.repeat(3, 1, 1).to("cuda")
        # self.filters = self.filters.unsqueeze(1).repeat(1, in_channels, 1, 1)
        # self.register_buffer("filters", filters)

        # # Define Conv2d with appropriate groups for multi-channel input
        # self.conv = nn.Conv2d(
        #     in_channels=in_channels,
        #     out_channels=4 * in_channels,  # 4 filters (LL, LH, HL, HH)
        #     kernel_size=2,
        #     stride=2,  # Downsample
        #     padding=0,  # No padding for DWT
        #     groups=in_channels,  # Apply DWT to each channel
        #     bias=False,
        # )
        # # filters = filters.repeat(in_channels, 1, 1, 1)
        # with torch.no_grad():
        #     self.conv.weight.data = filters
        #     self.conv.weight.requires_grad = False

    def forward(self, x):
        # out = self.conv(x)
        out = F.conv2d(
            x,
            self.filters[:, None],
            stride=2,
            padding=0,
            groups=3,
            # groups=x.size(1),
        )
        return out


def haar2_conv2d_grouped(x):
    # Define Haar wavelet filters for a single channel
    filters = torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5]],  # Approximation (a0)
            [[0.5, 0.5], [-0.5, -0.5]],  # Horizontal detail (a1)
            [[0.5, -0.5], [0.5, -0.5]],  # Vertical detail (a2)
            [[0.5, -0.5], [-0.5, 0.5]],  # Diagonal detail (a3)
        ]
    ).to(
        "cuda"
    )  # Shape: (4, 2, 2)

    # Expand filters to handle 3 input channels
    # Create one set of filters for each input channel
    filters = filters.repeat(3, 1, 1, 1)  # Shape: (12, 1, 2, 2)

    # Reshape for grouped convolution
    filters = filters.view(
        12, 1, 2, 2
    )  # Shape: (12 filters, 1 in_channel per group, 2, 2)

    # Apply grouped convolution
    output = F.conv2d(
        x, filters, groups=3, stride=2
    )  # Groups = 3 (one per input channel)

    return output


def inverse_haar2_conv2d(y):
    """
    Performs the inverse Haar wavelet transform.
    Args:
        y: The output tensor from haar2_conv2d_grouped (shape: [1, 12, H/2, W/2]).
        h: Original height of the input.
        w: Original width of the input.
    Returns:
        Reconstructed tensor (shape: [1, 3, h, w]).
    """
    # Split coefficients into 4 groups for each channel
    a0, a1, a2, a3 = torch.chunk(y, 4, dim=1)  # Each has shape: [1, 3, H/2, W/2]

    # Create an empty tensor to store the reconstructed result
    output = torch.zeros((1, 3, y.shape[2] * 2, y.shape[3] * 2), device=y.device)

    # Perform inverse transform for each channel
    output[:, :, 0::2, 0::2] = a0 + a1 + a2 + a3
    output[:, :, 0::2, 1::2] = a0 + a1 - a2 - a3
    output[:, :, 1::2, 0::2] = a0 - a1 + a2 - a3
    output[:, :, 1::2, 1::2] = a0 - a1 - a2 + a3

    return output
