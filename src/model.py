import torch
import torch.nn as nn
import torch.nn.functional as F


class DSResBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()


        # Depthwise convolution
        
        self.dw = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False
        )
        self.dw_bn = nn.BatchNorm2d(in_ch)

        # Pointwise convolution
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)

        # Residual shortcut
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.dw(x)
        out = self.dw_bn(out)
        out = self.act(out)

        out = self.pw(out)
        out = self.pw_bn(out)

        out = self.act(out + identity)
        return out


class SeqMLPHead(nn.Module):
    """
    Sequence-based MLP head.
    Input: sequence of length 3 with embedding size C -> shape [B, 3, C]
    The order of elements is preserved by concatenation.
    """
    def __init__(self, c: int, hidden: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * c, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, seq):
        # seq: [B, 3, C] -> [B, 3C]
        x = seq.reshape(seq.size(0), 3 * seq.size(2))
        return self.mlp(x)


class SudokuNet(nn.Module):
    """
    CNN for SudokuMNIST.

    - Input is the full 84x84 image (no preprocessing split)
    - Backbone embeds the image into a feature map
    - Feature map is pooled to a 3x3 grid using AdaptiveAvgPool2d
    - Task-specific heads operate on the grid structure
    """
    def __init__(self, feat_dim: int = 48, head_hidden: int = 32):
        super().__init__()

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),  # 84 -> 42
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # CNN backbone
        self.backbone = nn.Sequential(
            DSResBlock(16, 16),
            DSResBlock(16, 24, stride=2),  # 42 -> 21
            DSResBlock(24, 24),
            DSResBlock(24, 32, stride=2),  # 21 -> 11
            DSResBlock(32, 32),
            DSResBlock(32, feat_dim),
        )

        # Pool feature map to a 3x3 grid (represents Sudoku cells)
        self.pool_to_grid = nn.AdaptiveAvgPool2d((3, 3))

        # Task 1: missing digit (global information, order not important)
        self.head_missing = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, 10),
        )

        # Task 2: sorted rows/columns (order-sensitive)
        self.head_sorted = SeqMLPHead(
            c=feat_dim,
            hidden=head_hidden,
            out_dim=3
        )

        # Task 3: row/column sums (regression)
        self.head_sum = SeqMLPHead(
            c=feat_dim,
            hidden=head_hidden,
            out_dim=1
        )

    def _grid_to_sequences(self, grid_feat):

        b, c, _, _ = grid_feat.shape

        # Flatten grid cells
        cells = grid_feat.permute(0, 2, 3, 1).reshape(b, 9, c)

        # Rows (left to right)
        row0 = cells[:, 0:3]
        row1 = cells[:, 3:6]
        row2 = cells[:, 6:9]

        # Columns (top to bottom)
        col0 = cells[:, [0, 3, 6]]
        col1 = cells[:, [1, 4, 7]]
        col2 = cells[:, [2, 5, 8]]

        sequences = torch.stack(
            [row0, row1, row2, col0, col1, col2],
            dim=1
        )
        return sequences

    def forward(self, x):
        x = self.stem(x)
        feat = self.backbone(x)  # [B, C, 11, 11]

        # Missing digit head
        missing_logits = self.head_missing(feat)

        if feat.shape[-1] % 3 != 0 or feat.shape[-2] % 3 != 0:
            feat = F.interpolate(feat, size=(12, 12), mode="bilinear", align_corners=False)
            
        grid = self.pool_to_grid(feat)              # [B, C, 3, 3]
        seqs = self._grid_to_sequences(grid)        # [B, 6, 3, C]

        b = seqs.size(0)
        seqs_flat = seqs.reshape(b * 6, 3, -1)

        # Sorted head
        sorted_logits = self.head_sorted(seqs_flat)
        sorted_logits = sorted_logits.reshape(b, 6, 3)

        # Sum head
        sum_pred = self.head_sum(seqs_flat)
        sum_pred = sum_pred.reshape(b, 6)

        return {
            "missing_logits": missing_logits,
            "sorted_logits": sorted_logits,
            "sum_pred": sum_pred,
        }
