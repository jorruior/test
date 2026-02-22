class PosTransModelTCN_Refine(nn.Module):
    """
    Two-pass model for Ribo-seq prediction with iterative refinement.

    Pass 1 (backbone): identical to PosTransModelTCN
        Input:  (B, seq_len, 5+1)  →  DNA one-hot + RNA-seq
        Output: (B, nbins)         →  initial Ribo-seq prediction

    Pass 2 (refinement): lightweight causal TCN
        Input:  (B, seq_len, 5+1+1) → original features + upsampled pass-1 prediction
        Output: (B, nbins)          →  refined Ribo-seq prediction (additive residual)

    Final output = pass1 + pass2  (residual refinement)

    The refinement network is deliberately smaller (~25% of backbone params)
    to avoid overfitting — it only needs to learn corrections, not the full
    mapping.
    """

    def __init__(self, num_genomic_features, target_length, nbins,
                 seqno=False, mid_hidden=None, **kwargs):
        super().__init__()
        self.target_length = target_length
        self.nbins = nbins

        input_channels = num_genomic_features if seqno else 5 + num_genomic_features

        # ═══════════════════════════════════════════════════════════════
        # Pass 1: Backbone (same as PosTransModelTCN)
        # ═══════════════════════════════════════════════════════════════

        # Multi-kernel causal motif detectors
        self.conv_k3 = CausalConv1d(input_channels, 64, kernel_size=3)
        self.conv_k6 = CausalConv1d(input_channels, 64, kernel_size=6)
        self.conv_k25 = CausalConv1d(input_channels, 64, kernel_size=25)
        self.conv_gn = nn.GroupNorm(8, 64 * 3)
        self.conv_relu = nn.ReLU()

        # Causal dilated TCN
        self.tcn = nn.Sequential(
            CausalResidualBlock(192, 256, dilation=1),
            CausalResidualBlock(256, 256, dilation=4),
            CausalResidualBlock(256, 384, dilation=16),
            CausalResidualBlock(384, 384, dilation=64),
            CausalResidualBlock(384, 384, dilation=128),
            CausalResidualBlock(384, 384, dilation=256),
        )

        # Decoder
        self.final_conv = nn.Sequential(
            nn.Conv1d(384, 256, kernel_size=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Conv1d(256, 1, kernel_size=1),
        )

        # Downsampling
        if target_length == nbins:
            self.bin_pool = nn.Identity()
            self.pool_k = 1
        else:
            self.pool_k = target_length // nbins
            self.bin_pool = nn.AvgPool1d(
                kernel_size=self.pool_k, stride=self.pool_k,
            )

        # ═══════════════════════════════════════════════════════════════
        # Pass 2: Refinement network (lightweight)
        # ═══════════════════════════════════════════════════════════════
        # Input: original features + 1 extra channel (pass-1 prediction)
        refine_in = input_channels + 1

        # Smaller multi-kernel conv (half the channels)
        self.ref_conv_k3 = CausalConv1d(refine_in, 32, kernel_size=3)
        self.ref_conv_k6 = CausalConv1d(refine_in, 32, kernel_size=6)
        self.ref_conv_k25 = CausalConv1d(refine_in, 32, kernel_size=25)
        self.ref_conv_gn = nn.GroupNorm(8, 96)
        self.ref_conv_relu = nn.ReLU()

        # Lighter TCN — 4 blocks, narrower channels, focus on long-range
        # (uORF effects propagate over long distances)
        self.ref_tcn = nn.Sequential(
            CausalResidualBlock(96, 128, dilation=1),
            CausalResidualBlock(128, 128, dilation=16),
            CausalResidualBlock(128, 128, dilation=64),
            CausalResidualBlock(128, 128, dilation=256),
        )

        # Refinement decoder -> residual correction
        self.ref_final = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1),
        )

        # Learnable gate: controls how much refinement to apply
        # Initialized near 0 so early training behaves like base model
        self.refine_gate = nn.Parameter(torch.tensor(0.0))

    def _backbone_forward(self, x_channels):
        """Pass 1: standard backbone. x_channels is (B, C, seq_len)."""
        x1 = self.conv_k3(x_channels)
        x2 = self.conv_k6(x_channels)
        x3 = self.conv_k25(x_channels)

        min_len = min(x1.size(2), x2.size(2), x3.size(2))
        x = torch.cat(
            [x1[:, :, :min_len], x2[:, :, :min_len], x3[:, :, :min_len]],
            dim=1,
        )
        x = self.conv_gn(x)
        x = self.conv_relu(x)
        x = self.tcn(x)
        x = self.final_conv(x)
        x = self.bin_pool(x)
        return x.squeeze(1)  # (B, nbins)

    def _refine_forward(self, x_channels):
        """Pass 2: lightweight refinement. x_channels is (B, C+1, seq_len)."""
        r1 = self.ref_conv_k3(x_channels)
        r2 = self.ref_conv_k6(x_channels)
        r3 = self.ref_conv_k25(x_channels)

        min_len = min(r1.size(2), r2.size(2), r3.size(2))
        r = torch.cat(
            [r1[:, :, :min_len], r2[:, :, :min_len], r3[:, :, :min_len]],
            dim=1,
        )
        r = self.ref_conv_gn(r)
        r = self.ref_conv_relu(r)
        r = self.ref_tcn(r)
        r = self.ref_final(r)
        r = self.bin_pool(r)
        return r.squeeze(1)  # (B, nbins)

    def forward(self, x):
        """
        x: (B, seq_len, features) — DNA one-hot (4ch) + N-mask (1ch) + RNA-seq (1ch)
        returns: (B, nbins) — predicted Ribo-seq profile
        """
        # (B, seq_len, C) -> (B, C, seq_len)
        x_ch = x.permute(0, 2, 1).float()

        # --- Pass 1: backbone prediction ---
        pred1 = self._backbone_forward(x_ch)  # (B, nbins)

        # --- Upsample pred1 back to seq_len for pass 2 ---
        # (B, nbins) -> (B, 1, nbins) -> (B, 1, seq_len) via repeat
        pred1_up = pred1.unsqueeze(1)  # (B, 1, nbins)
        pred1_up = pred1_up.repeat_interleave(self.pool_k, dim=2)  # (B, 1, seq_len)

        # Trim or pad to match input length exactly
        seq_len = x_ch.size(2)
        if pred1_up.size(2) > seq_len:
            pred1_up = pred1_up[:, :, :seq_len]
        elif pred1_up.size(2) < seq_len:
            pred1_up = F.pad(pred1_up, (0, seq_len - pred1_up.size(2)))

        # --- Pass 2: refinement with original input + pass-1 prediction ---
        x_refine = torch.cat([x_ch, pred1_up.detach()], dim=1)
        correction = self._refine_forward(x_refine)  # (B, nbins)

        # Gated residual: output = pred1 + sigmoid(gate) * correction
        gate = torch.sigmoid(self.refine_gate)
        return pred1 + gate * correction
