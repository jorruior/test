#!/usr/bin/env python3
"""
RiboTransPred — Tissue-conditioned Ribo-seq prediction (FiLM)
=============================================================
Predicts Ribo-seq profiles from RNA-seq + DNA sequence, conditioned on
tissue identity via FiLM (Feature-wise Linear Modulation).

Architecture:
  PosTransModelTCN backbone is shared across all tissues (generalised
  translational regulation). Tissue conditioning is injected via FiLM
  layers that produce per-tissue (gamma, beta) scale/shift parameters
  after each TCN residual block and before the decoder.

Prediction modes:
  - Known tissue:  pass tissue index → uses learned tissue embedding
  - New tissue:    pass index num_tissues → uses average of all tissue
                   embeddings (generalised cross-tissue prediction)

Checkpoint contains: tissue_vocab (tissue→index mapping) stored in
hparams so the same ckpt can be loaded for prediction on any tissue.

3-node / 24-GPU DDP training via SLURM + PyTorch Lightning.
Author: Jorge Ruiz-Orera (rewritten)
"""

import argparse
import datetime
import gc
import json
import logging
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

_RANK       = int(os.environ.get("SLURM_PROCID",  os.environ.get("RANK", 0)))
_LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
_WORLD_SIZE = int(os.environ.get("SLURM_NTASKS",  os.environ.get("WORLD_SIZE", 1)))
_NNODES     = int(os.environ.get("SLURM_NNODES",  1))


def _dbg(msg):
    sys.stderr.write(f"[rank {_RANK}] {msg}\n")
    sys.stderr.flush()


if _RANK != 0:
    import builtins
    builtins._original_print = builtins.print
    builtins.print = lambda *a, **k: None
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

warnings.filterwarnings("ignore", message="Trying to infer the `batch_size`")

_NCPU = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ[_v] = str(max(1, _NCPU))

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler
from torchmetrics import Metric

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import SLURMEnvironment
import pytorch_lightning.callbacks as plc
from transformers import get_cosine_schedule_with_warmup


# ═══════════════════════════════════════════════════════════════════════════
# § 1  Model: PosTransModelTCN with FiLM tissue conditioning
# ═══════════════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              dilation=dilation, padding=0)

    def forward(self, x):
        pad = self.dilation * (self.kernel_size - 1)
        x = F.pad(x, (pad, 0))
        return self.conv(x)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Takes a tissue embedding and produces (gamma, beta) to scale/shift
    feature maps: output = gamma * x + beta

    Initialised to identity (gamma=1, beta=0) so an untrained FiLM
    layer is a no-op, preserving the base model's behaviour.
    """

    def __init__(self, tissue_emb_dim, num_channels):
        super().__init__()
        self.fc = nn.Linear(tissue_emb_dim, num_channels * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        with torch.no_grad():
            self.fc.bias[:num_channels] = 1.0   # gamma = 1

    def forward(self, x, tissue_emb):
        """
        x:          (B, C, L)  feature maps
        tissue_emb: (B, D)     tissue embedding
        Returns:    (B, C, L)  modulated feature maps
        """
        params = self.fc(tissue_emb)           # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)   # each (B, C)
        return gamma.unsqueeze(2) * x + beta.unsqueeze(2)


class CausalResidualBlockFiLM(nn.Module):
    """Causal dilated residual block with FiLM after norm2."""

    def __init__(self, in_ch, out_ch, dilation, tissue_emb_dim):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size=3, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.residual = (nn.Conv1d(in_ch, out_ch, kernel_size=1)
                         if in_ch != out_ch else nn.Identity())
        self.film = FiLMLayer(tissue_emb_dim, out_ch)

    def forward(self, x, tissue_emb):
        res = self.residual(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.film(x, tissue_emb)
        return self.relu(x + res)


class PosTransModelTCN_FiLM(nn.Module):
    """
    PosTransModelTCN with FiLM tissue conditioning.

    Convolutional weights are shared across tissues (generalised patterns).
    FiLM layers learn per-tissue modulation of those patterns.

    For unseen tissues: pass tissue_ids = num_tissues (the +1 slot) which
    is initialised to the mean of all learned embeddings after training.
    """

    def __init__(self, num_genomic_features, target_length, nbins,
                 num_tissues, tissue_emb_dim=64, seqno=False, **kwargs):
        super().__init__()
        self.seqno = seqno
        self.num_tissues = num_tissues
        self.tissue_emb_dim = tissue_emb_dim

        input_channels = num_genomic_features if seqno else 5 + num_genomic_features

        # Tissue embedding: num_tissues known + 1 "mean/unknown" slot
        self.tissue_embedding = nn.Embedding(num_tissues + 1, tissue_emb_dim)

        # --- 1. Multi-kernel causal motif detectors ---
        self.conv_k3  = CausalConv1d(input_channels, 64, kernel_size=3)
        self.conv_k6  = CausalConv1d(input_channels, 64, kernel_size=6)
        self.conv_k25 = CausalConv1d(input_channels, 64, kernel_size=25)
        self.conv_gn   = nn.GroupNorm(8, 192)
        self.conv_relu = nn.ReLU()
        self.film_conv = FiLMLayer(tissue_emb_dim, 192)

        # --- 2. TCN with FiLM ---
        self.tcn_blocks = nn.ModuleList([
            CausalResidualBlockFiLM(192, 256, dilation=1,   tissue_emb_dim=tissue_emb_dim),
            CausalResidualBlockFiLM(256, 256, dilation=4,   tissue_emb_dim=tissue_emb_dim),
            CausalResidualBlockFiLM(256, 384, dilation=16,  tissue_emb_dim=tissue_emb_dim),
            CausalResidualBlockFiLM(384, 384, dilation=64,  tissue_emb_dim=tissue_emb_dim),
            CausalResidualBlockFiLM(384, 384, dilation=128, tissue_emb_dim=tissue_emb_dim),
            CausalResidualBlockFiLM(384, 384, dilation=256, tissue_emb_dim=tissue_emb_dim),
        ])

        # --- 3. Decoder (split so we can insert FiLM) ---
        self.dec_conv1 = nn.Conv1d(384, 256, kernel_size=1)
        self.dec_norm  = nn.GroupNorm(8, 256)
        self.dec_relu  = nn.ReLU()
        self.dec_conv2 = nn.Conv1d(256, 1, kernel_size=1)
        self.film_dec  = FiLMLayer(tissue_emb_dim, 256)

        # --- 4. Downsampling ---
        if target_length == nbins:
            self.bin_pool = nn.Identity()
        else:
            stride = target_length // nbins
            self.bin_pool = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def set_mean_embedding(self):
        """Copy the mean of known tissue embeddings into the +1 slot."""
        with torch.no_grad():
            mean = self.tissue_embedding.weight[:self.num_tissues].mean(dim=0)
            self.tissue_embedding.weight[self.num_tissues] = mean

    def forward(self, x, tissue_ids):
        """
        x:          (B, seq_len, features)
        tissue_ids: (B,) LongTensor — index into tissue_embedding
        """
        t_emb = self.tissue_embedding(tissue_ids)   # (B, D)
        x = x.permute(0, 2, 1).float()              # (B, C, L)

        # Multi-kernel conv
        x1, x2, x3 = self.conv_k3(x), self.conv_k6(x), self.conv_k25(x)
        ml = min(x1.size(2), x2.size(2), x3.size(2))
        x = torch.cat([x1[:,:,:ml], x2[:,:,:ml], x3[:,:,:ml]], dim=1)
        x = self.conv_relu(self.conv_gn(x))
        x = self.film_conv(x, t_emb)

        # TCN
        for blk in self.tcn_blocks:
            x = blk(x, t_emb)

        # Decoder
        x = self.dec_conv1(x)
        x = self.dec_norm(x)
        x = self.film_dec(x, t_emb)
        x = self.dec_relu(x)
        x = self.dec_conv2(x)

        x = self.bin_pool(x)
        return x.squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════
# § 2  File conversion
# ═══════════════════════════════════════════════════════════════════════════

def _pt_to_npy(pt_path):
    npy_path = pt_path[:-3] + ".npy"
    if os.path.exists(npy_path):
        return npy_path
    try:
        t = torch.load(pt_path, map_location="cpu", weights_only=True)
        tmp = npy_path + f".tmp.{os.getpid()}"
        np.save(tmp, t.numpy())
        del t; gc.collect()
        os.replace(tmp, npy_path)
        print(f"  converted {os.path.basename(pt_path)}")
        return npy_path
    except Exception as e:
        print(f"  WARNING: cannot convert {pt_path}: {e}")
        return None


def convert_all_tracks(training_tracks, args):
    if _LOCAL_RANK == 0:
        print(f"\n  Converting .pt -> .npy on node "
              f"{os.environ.get('SLURM_NODEID', '?')} ...")
        for t in training_tracks:
            sp, ti = t["species"], t["tissue"]
            bn = f"{sp}_{ti}"
            td = args.tracks_dir
            inp = f"{td}/{sp}/{ti}/{bn}_rna_{args.region_len}_log_rnaseq_final.pt"
            tag = "ribo.psites" if args.psites else "ribo"
            out = (f"{td}/{sp}/{ti}/{bn}_{tag}_"
                   f"{args.region_len}_{args.nBins}_log_riboseq_final.pt")
            if os.path.exists(inp): _pt_to_npy(inp)
            if os.path.exists(out): _pt_to_npy(out)
        print("  Conversions done.")

    if dist.is_initialized():
        dist.barrier()
    elif _LOCAL_RANK != 0:
        time.sleep(30)


# ═══════════════════════════════════════════════════════════════════════════
# § 3  Coordinates + chromosome splitting
# ═══════════════════════════════════════════════════════════════════════════

def _npy_paths(tracks_dir, species, tissue, region_len, nbins, psites):
    bn = f"{species}_{tissue}"
    d = f"{tracks_dir}/{species}/{tissue}"
    inp = f"{d}/{bn}_rna_{region_len}_log_rnaseq_final.npy"
    tag = "ribo.psites" if psites else "ribo"
    out = f"{d}/{bn}_{tag}_{region_len}_{nbins}_log_riboseq_final.npy"
    return inp, out


def load_coords(species):
    """Load ALL transcripts (all biotypes) to keep mmap row alignment."""
    path = f"coordinates/{species}_coordinates.txt"
    if not os.path.exists(path):
        _dbg(f"ERROR: coordinate file not found: {path}")
        return None
    rows = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 8:
                continue
            rows.append({
                "chr": p[0], "start": int(p[1]), "end": int(p[2]),
                "id": p[3], "biotype": p[5], "sequence": p[7],
                "species": species,
            })
    return pd.DataFrame(rows) if rows else None


def _norm_chr(c):
    return c[3:] if c.lower().startswith("chr") else c


def split_chromosomes(chroms, trial=None, fixed_val=None):
    chroms = [c for c in chroms if len(c) <= 7 and "MT" not in c.upper()]
    normed = {c: _norm_chr(c) for c in chroms}

    if trial is not None:
        hits = [c for c in chroms if normed[c] == _norm_chr(str(trial))]
        return ([hits[0]], [hits[0]], [hits[0]]) if hits else ([], [], [])

    test = []
    for want in ("16", "1", "X"):
        for c in chroms:
            if normed[c] == want and c not in test:
                test.append(c); break
    remaining = [c for c in chroms if c not in test]
    test += remaining[:max(0, 3 - len(test))]
    remaining = [c for c in chroms if c not in test]

    if fixed_val:
        fv = set(fixed_val)
        val = [c for c in remaining if normed[c] in fv]
        remaining = [c for c in remaining if c not in val]
        if len(val) < 3:
            val += remaining[:3 - len(val)]
            remaining = [c for c in remaining if c not in val]
    else:
        val, remaining = remaining[:3], remaining[3:]
    return remaining, val, test


def _biotype_mask(df_all, biotype):
    if biotype == "protein_coding":
        return (df_all["biotype"] == "protein_coding").values
    elif biotype == "non_coding":
        return df_all["biotype"].isin(["lncRNA", "lincRNA"]).values
    return np.ones(len(df_all), dtype=bool)


def compute_shared_val_chromosomes(training_tracks, args, biotype):
    job_id = os.environ.get("SLURM_JOB_ID", "0")
    chrom_file = f"_val_chroms_tissue_{job_id}.json"

    if _RANK == 0:
        per_species = {}
        for t in training_tracks:
            df = load_coords(t["species"])
            if df is not None:
                per_species.setdefault(t["species"], set()).update(
                    _norm_chr(c) for c in df["chr"].unique())
        # Test chromosomes are 16, 1, X — exclude from val candidates
        test_chroms = {"16", "1", "X"}
        shared = []
        for n in range(1, 23):
            s = str(n)
            if s in test_chroms:
                continue
            if all(s in v for v in per_species.values()):
                shared.append(s)

        # Randomly sample 3 from shared non-test chromosomes
        import random
        rng = random.Random(args.seed)
        if len(shared) >= 3:
            candidates = sorted(rng.sample(shared, 3))
        else:
            candidates = shared
        tmp = f"{chrom_file}.tmp.{os.getpid()}"
        with open(tmp, "w") as f: json.dump(candidates, f)
        os.replace(tmp, chrom_file)
        print(f"  Validation chromosomes: {candidates}")

    deadline = time.time() + 120
    while not os.path.exists(chrom_file):
        if time.time() > deadline: return ["1", "2", "3"]
        time.sleep(1)
    with open(chrom_file) as f:
        result = json.load(f)

    if dist.is_initialized(): dist.barrier()
    if _RANK == _WORLD_SIZE - 1:
        try: time.sleep(2); os.remove(chrom_file)
        except OSError: pass
    return result


# ═══════════════════════════════════════════════════════════════════════════
# § 4  Dataset — returns tissue_id
# ═══════════════════════════════════════════════════════════════════════════

_OHE_TABLE = np.zeros((256, 5), dtype=np.float32)
for _i, _b in enumerate("ATCGN"):  _OHE_TABLE[ord(_b), _i] = 1.0
for _i, _b in enumerate("atcgn"):  _OHE_TABLE[ord(_b), _i] = 1.0


def _encode_seq_fast(seq, seq_len):
    s = seq[:seq_len]
    codes = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    ohe = _OHE_TABLE[codes].copy()
    if len(ohe) < seq_len:
        ohe = np.concatenate([ohe, np.zeros((seq_len - len(ohe), 5),
                                            dtype=np.float32)])
    return ohe


def _make_mask_fast(seq, seq_len, pool_k):
    s = seq[:seq_len]
    codes = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    valid = ((codes != ord("N")) & (codes != ord("n"))).astype(np.float32)
    raw = np.zeros(seq_len, dtype=np.float32)
    raw[:len(valid)] = valid
    if pool_k > 1:
        L = (seq_len // pool_k) * pool_k
        raw = raw[:L].reshape(-1, pool_k).mean(axis=1)
    return raw


class TissueTranscriptDataset(Dataset):
    """Mmap-backed dataset that returns tissue_id and track_id for FiLM + per-track testing."""

    def __init__(self, sequences, index_array, inp_mmap, out_mmap,
                 tissue_id, seq_len, nbins, nosequence=False, track_id=0):
        self.sequences  = sequences
        self.inp_mmap   = inp_mmap
        self.out_mmap   = out_mmap
        self.tissue_id  = tissue_id
        self.track_id   = track_id
        self.seq_len    = seq_len
        self.nbins      = nbins
        self.nosequence = nosequence
        self.pool_k     = max(1, seq_len // nbins)

        valid = []
        for idx in index_array:
            seq = sequences[int(idx)]
            if seq and any(c not in ("N", "n") for c in seq[:seq_len]):
                valid.append(idx)
        self.index_array = np.array(valid, dtype=np.int64)

    def __len__(self):
        return len(self.index_array)

    def __getitem__(self, i):
        row = int(self.index_array[i])
        seq = self.sequences[row]
        if self.nosequence:
            seq = "N" * min(len(seq), self.seq_len)

        ohe = _encode_seq_fast(seq, self.seq_len)
        rna = self.inp_mmap[row].astype(np.float32)
        features = np.concatenate([ohe, rna[:, np.newaxis]], axis=1)
        target = self.out_mmap[row].astype(np.float32)
        mask = _make_mask_fast(seq, self.seq_len, self.pool_k)

        return {
            "features":  torch.from_numpy(features),
            "target":    torch.from_numpy(target),
            "mask":      torch.from_numpy(mask),
            "tissue_id": torch.tensor(self.tissue_id, dtype=torch.long),
            "track_id":  torch.tensor(self.track_id, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════
# § 5  DataModule — tissue-aware
# ═══════════════════════════════════════════════════════════════════════════

class TissueRiboDataModule(LightningDataModule):

    def __init__(self, args, training_tracks, val_chroms, tissue_vocab):
        super().__init__()
        self.args            = args
        self.training_tracks = training_tracks
        self.val_chroms_norm = set(val_chroms)
        self.tissue_vocab    = tissue_vocab
        self._train_ds = None
        self._val_ds   = None
        self._test_ds  = None
        self._mmap_refs = []
        self.track_names = []  # track_id -> (species, tissue)

    def setup(self, stage=None):
        if self._train_ds is not None:
            return

        _dbg(f"setup() n_tracks={len(self.training_tracks)} "
             f"n_tissues={len(self.tissue_vocab)}")

        # Wait for npy files
        deadline = time.time() + 300
        while time.time() < deadline:
            all_ready = True
            for t in self.training_tracks:
                inp_npy, out_npy = _npy_paths(
                    self.args.tracks_dir, t["species"], t["tissue"],
                    self.args.region_len, self.args.nBins, self.args.psites)
                if not os.path.exists(inp_npy) or not os.path.exists(out_npy):
                    all_ready = False; break
            if all_ready: break
            time.sleep(2)

        train_parts, val_parts, test_parts = [], [], []

        for track in self.training_tracks:
            sp, ti = track["species"], track["tissue"]
            tissue_id = self.tissue_vocab.get(ti, len(self.tissue_vocab))
            track_id = len(self.track_names)
            self.track_names.append((sp, ti))

            inp_npy, out_npy = _npy_paths(
                self.args.tracks_dir, sp, ti,
                self.args.region_len, self.args.nBins, self.args.psites)
            if not os.path.exists(inp_npy) or not os.path.exists(out_npy):
                _dbg(f"SKIP {sp}/{ti}: missing npy"); continue

            df_all = load_coords(sp)
            if df_all is None or df_all.empty:
                _dbg(f"SKIP {sp}/{ti}: no coordinates"); continue

            inp_mm = np.load(inp_npy, mmap_mode="r")
            out_mm = np.load(out_npy, mmap_mode="r")
            self._mmap_refs.extend([inp_mm, out_mm])

            if inp_mm.shape[0] != len(df_all):
                _dbg(f"SKIP {sp}/{ti}: shape mismatch "
                     f"mmap={inp_mm.shape[0]} coords={len(df_all)}")
                continue

            bt_ok = _biotype_mask(df_all, self.args.biotype)
            sequences = df_all["sequence"].values
            norm_chroms = np.array([_norm_chr(c) for c in df_all["chr"].values])

            train_chr, val_chr, test_chr = split_chromosomes(
                df_all["chr"].unique().tolist(),
                trial=self.args.trial,
                fixed_val=list(self.val_chroms_norm))
            tr_set = {_norm_chr(c) for c in train_chr}
            vl_set = {_norm_chr(c) for c in val_chr}
            te_set = {_norm_chr(c) for c in test_chr}

            chr_tr = np.isin(norm_chroms, list(tr_set))
            chr_vl = np.isin(norm_chroms, list(vl_set))
            chr_te = np.isin(norm_chroms, list(te_set))
            tr_idx = np.where(chr_tr & bt_ok)[0]
            vl_idx = np.where(chr_vl & bt_ok)[0]
            te_idx = np.where(chr_te & bt_ok)[0]

            if len(tr_idx):
                ds = TissueTranscriptDataset(
                    sequences, tr_idx, inp_mm, out_mm,
                    tissue_id, self.args.region_len, self.args.nBins,
                    self.args.nosequence, track_id=track_id)
                train_parts.append(ds)

            if len(vl_idx):
                ds = TissueTranscriptDataset(
                    sequences, vl_idx, inp_mm, out_mm,
                    tissue_id, self.args.region_len, self.args.nBins,
                    self.args.nosequence, track_id=track_id)
                val_parts.append(ds)

            if len(te_idx):
                ds = TissueTranscriptDataset(
                    sequences, te_idx, inp_mm, out_mm,
                    tissue_id, self.args.region_len, self.args.nBins,
                    self.args.nosequence, track_id=track_id)
                test_parts.append(ds)

            n_tr = len(train_parts[-1]) if len(tr_idx) else 0
            n_vl = len(val_parts[-1]) if len(vl_idx) else 0
            n_te = len(test_parts[-1]) if len(te_idx) else 0
            _dbg(f"LOADED {sp}/{ti} (tissue_id={tissue_id} track_id={track_id}): "
                 f"train={n_tr} val={n_vl} test={n_te}")

        self._train_ds = ConcatDataset(train_parts) if train_parts else None
        self._val_ds   = ConcatDataset(val_parts)   if val_parts   else None
        self._test_ds  = ConcatDataset(test_parts)  if test_parts  else None

        if self._train_ds is None:
            raise RuntimeError("No training data loaded.")

        total_tr = len(self._train_ds)
        total_vl = len(self._val_ds) if self._val_ds else 0
        total_te = len(self._test_ds) if self._test_ds else 0
        _dbg(f"Total: {total_tr} train, {total_vl} val, {total_te} test")
        print(f"\n  Total dataset: {total_tr} train, {total_vl} val, {total_te} test")
        print(f"  Tissues: {len(self.tissue_vocab)}\n")

    def _make_loader(self, dataset, shuffle, drop_last):
        sampler = None
        try:    world = self.trainer.world_size
        except: world = _WORLD_SIZE
        if world > 1:
            sampler = DistributedSampler(dataset, shuffle=shuffle,
                                         drop_last=drop_last)
            shuffle = False
        nw = self.args.num_workers
        return DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=shuffle,
            sampler=sampler, num_workers=nw,
            pin_memory=torch.cuda.is_available(), drop_last=drop_last,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None, timeout=600)

    def train_dataloader(self):
        if self._train_ds is None:
            raise RuntimeError("setup() not called")
        return self._make_loader(self._train_ds, shuffle=True, drop_last=True)

    def val_dataloader(self):
        if self._val_ds is None or len(self._val_ds) == 0:
            return None
        return self._make_loader(self._val_ds, shuffle=False, drop_last=True)

    def test_dataloader(self):
        if self._test_ds is None or len(self._test_ds) == 0:
            return None
        return self._make_loader(self._test_ds, shuffle=False, drop_last=True)


# ═══════════════════════════════════════════════════════════════════════════
# § 6  Metrics and loss
# ═══════════════════════════════════════════════════════════════════════════

class StreamingPearsonR(Metric):
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for name in ("sum_xy", "sum_x", "sum_y", "sum_x2", "sum_y2", "count"):
            self.add_state(name,
                           default=torch.tensor(0.0, dtype=torch.float64),
                           dist_reduce_fx="sum")

    def update(self, pred, target, mask):
        m = mask.reshape(-1).bool()
        p = (pred * mask).reshape(-1)[m].double()
        t = (target * mask).reshape(-1)[m].double()
        n = p.numel()
        if n < 2: return
        self.sum_xy += (p * t).sum()
        self.sum_x  += p.sum()
        self.sum_y  += t.sum()
        self.sum_x2 += (p * p).sum()
        self.sum_y2 += (t * t).sum()
        self.count  += n

    def compute(self):
        n = self.count.clamp(min=2)
        cov = self.sum_xy - self.sum_x * self.sum_y / n
        vx  = self.sum_x2 - self.sum_x ** 2 / n
        vy  = self.sum_y2 - self.sum_y ** 2 / n
        return (cov / torch.sqrt((vx * vy).clamp(min=1e-16))).float()


def _per_sample_pcc_loss(pred, target, mask):
    p, t = pred * mask, target * mask
    n = mask.sum(dim=1, keepdim=True).clamp(min=1)
    p_c = (p - p.sum(dim=1, keepdim=True) / n) * mask
    t_c = (t - t.sum(dim=1, keepdim=True) / n) * mask
    cov = (p_c * t_c).sum(dim=1)
    denom = torch.sqrt(((p_c**2).sum(dim=1) * (t_c**2).sum(dim=1)).clamp(min=1e-12))
    r = cov / denom
    valid = (n.squeeze(1) >= 10)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return 1.0 - r[valid].mean()


# ═══════════════════════════════════════════════════════════════════════════
# § 7  LightningModule — tissue-conditioned
# ═══════════════════════════════════════════════════════════════════════════

class TissueRiboModel(LightningModule):

    def __init__(self, args, tissue_vocab):
        super().__init__()
        self.save_hyperparameters({
            "model_type":     "PosTransModelTCN_FiLM",
            "region_len":     args.region_len,
            "nBins":          args.nBins,
            "dropout":        args.dropout,
            "learning_rate":  args.learning_rate,
            "weight_decay":   args.weight_decay,
            "warmup_steps":   args.warmup_steps,
            "batch_size":     args.batch_size,
            "grad_accum":     args.grad_accum,
            "nosequence":     args.nosequence,
            "seed":           args.seed,
            "tissue_emb_dim": args.tissue_emb_dim,
            "num_tissues":    len(tissue_vocab),
            "tissue_vocab":   tissue_vocab,
        })
        self.args = args
        self.tissue_vocab = tissue_vocab

        self.net = PosTransModelTCN_FiLM(
            num_genomic_features=1,
            target_length=args.region_len,
            nbins=args.nBins,
            num_tissues=len(tissue_vocab),
            tissue_emb_dim=args.tissue_emb_dim,
        )

        self.train_pcc = StreamingPearsonR()
        self.val_pcc   = StreamingPearsonR()
        self._tr_loss_sum = 0.0
        self._tr_loss_n   = 0
        self._vl_loss_sum = 0.0
        self._vl_loss_n   = 0
        self._history     = []

        # Per-track test accumulators
        self._test_per_track = {}
        self._track_names    = None
        self._zero_w      = 0.1 if args.nBins < args.region_len else 0.3
        self._pcc_loss_w  = 0.2

    def forward(self, x, tissue_ids):
        return self.net(x, tissue_ids)

    def _compute_loss(self, pred, target, mask):
        w = torch.where(target.abs() < 1e-6,
                        torch.full_like(target, self._zero_w),
                        torch.ones_like(target)) * mask
        mse = ((pred - target)**2 * w).sum() / (w.sum() + 1e-8)
        return mse + self._pcc_loss_w * _per_sample_pcc_loss(pred, target, mask)

    def training_step(self, batch, batch_idx):
        pred = self(batch["features"].float(), batch["tissue_id"])
        target, mask = batch["target"].float(), batch["mask"].float()
        loss = self._compute_loss(pred, target, mask)

        if torch.isnan(loss) or torch.isinf(loss):
            _dbg(f"WARNING: NaN/Inf loss at step {self.global_step}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        self._tr_loss_sum += loss.detach().item()
        self._tr_loss_n   += 1
        with torch.no_grad():
            self.train_pcc.update(pred.detach(), target, mask)
        if batch_idx % 50 == 0:
            self.log("train/loss_step", loss, prog_bar=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch["features"].float(), batch["tissue_id"])
        target, mask = batch["target"].float(), batch["mask"].float()
        loss = self._compute_loss(pred, target, mask)
        self._vl_loss_sum += loss.item()
        self._vl_loss_n   += 1
        self.val_pcc.update(pred, target, mask)

    def _get_test_track(self, tid):
        tid = int(tid)
        if tid not in self._test_per_track:
            self._test_per_track[tid] = {
                "loss_sum": 0.0, "loss_n": 0,
                "pcc": StreamingPearsonR().to(self.device),
            }
        return self._test_per_track[tid]

    def test_step(self, batch, batch_idx):
        features  = batch["features"].float()
        tissue_ids = batch["tissue_id"]
        target    = batch["target"].float()
        mask      = batch["mask"].float()
        track_ids = batch["track_id"]

        pred = self(features, tissue_ids)
        loss = self._compute_loss(pred, target, mask)

        for tid_val in track_ids.unique().tolist():
            sel = (track_ids == tid_val)
            acc = self._get_test_track(tid_val)
            acc["loss_sum"] += loss.item() * sel.sum().item() / len(track_ids)
            acc["loss_n"]   += sel.sum().item()
            acc["pcc"].update(pred[sel], target[sel], mask[sel])

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            print(f"\n  {'='*60}")
            print(f"  Test results per species / tissue")
            print(f"  {'='*60}")

        results = []
        for tid in sorted(self._test_per_track.keys()):
            acc = self._test_per_track[tid]
            avg_loss = acc["loss_sum"] / max(1, acc["loss_n"])
            pcc = acc["pcc"].compute().item()

            if self._track_names and tid < len(self._track_names):
                sp, ti = self._track_names[tid]
                label = f"{sp}/{ti}"
            else:
                label = f"track_{tid}"

            self.log(f"test/loss_{label}", avg_loss, sync_dist=False)
            self.log(f"test/pcc_{label}",  pcc,      sync_dist=False)

            if self.global_rank == 0:
                print(f"  {label:30s} | loss={avg_loss:.5f}  PCC={pcc:.4f}  "
                      f"(n={acc['loss_n']})")
            results.append({"species_tissue": label, "loss": avg_loss,
                            "pcc": pcc, "n_samples": acc["loss_n"]})

        total_n = sum(a["loss_n"] for a in self._test_per_track.values())
        if total_n > 0:
            total_loss = sum(a["loss_sum"] for a in self._test_per_track.values()) / total_n
            if self.global_rank == 0:
                print(f"  {'-'*60}")
                print(f"  {'OVERALL':30s} | loss={total_loss:.5f}  "
                      f"(n={total_n})")

        if self.global_rank == 0 and results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(self.args.save_dir, "test_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"\n  Test results saved -> {csv_path}")

        self._test_per_track = {}

    def on_train_epoch_end(self):
        avg = self._tr_loss_sum / max(1, self._tr_loss_n)
        pcc = self.train_pcc.compute().item()
        self.log("train/loss_epoch", avg, sync_dist=False, prog_bar=True)
        self.log("train/pcc_epoch",  pcc, sync_dist=False)
        if self.global_rank == 0:
            print(f"\n  Epoch {self.current_epoch} train | "
                  f"loss={avg:.5f}  PCC={pcc:.4f}")
        self.train_pcc.reset()
        self._tr_loss_sum = 0.0
        self._tr_loss_n   = 0

    def on_validation_epoch_end(self):
        avg = self._vl_loss_sum / max(1, self._vl_loss_n)
        pcc = self.val_pcc.compute().item()
        self.log("val/loss_epoch", avg, sync_dist=False, prog_bar=True)
        self.log("val/pcc_epoch",  pcc, sync_dist=False)
        if self.global_rank == 0:
            print(f"  Epoch {self.current_epoch} val   | "
                  f"loss={avg:.5f}  PCC={pcc:.4f}")
            self._history.append({
                "epoch": self.current_epoch,
                "val_loss": avg, "val_pcc": pcc})
        self.val_pcc.reset()
        self._vl_loss_sum = 0.0
        self._vl_loss_n   = 0

    def on_train_end(self):
        # Set mean embedding in the +1 slot for generalised prediction
        self.net.set_mean_embedding()

        if self.global_rank == 0:
            if self._history:
                df = pd.DataFrame(self._history)
                p = os.path.join(self.args.save_dir, "training_history.csv")
                df.to_csv(p, index=False)
                print(f"\n  History -> {p}")

            vocab_path = os.path.join(self.args.save_dir, "tissue_vocab.json")
            with open(vocab_path, "w") as f:
                json.dump(self.tissue_vocab, f, indent=2)
            print(f"  Tissue vocab -> {vocab_path}")

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            if "bias" in name or "norm" in name or "bn" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        opt = torch.optim.AdamW([
            {"params": decay,    "weight_decay": self.args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=self.args.learning_rate, betas=(0.9, 0.999), eps=1e-8)

        total_steps  = self.trainer.estimated_stepping_batches
        warmup_steps = min(self.args.warmup_steps, max(100, total_steps // 10))
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

# ═══════════════════════════════════════════════════════════════════════════
# § 8  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("RiboTransPred - Tissue FiLM")
    p.add_argument("--seed",            type=int,   default=4)
    p.add_argument("--save_path",       default="results_tissues")
    p.add_argument("--tracks",          default="tracks.txt")
    p.add_argument("--tracks_dir",      default="tracks")
    p.add_argument("--region_len",      type=int,   default=6000)
    p.add_argument("--nBins",           type=int,   default=1000)
    p.add_argument("--biotype",         default="protein_coding",
                   choices=["protein_coding", "non_coding", "all"])
    p.add_argument("--psites",          action="store_true")
    p.add_argument("--trial",           type=str,   default=None)
    p.add_argument("--nosequence",      action="store_true")
    p.add_argument("--patience",        type=int,   default=8)
    p.add_argument("--max-epochs",      type=int,   default=80)
    p.add_argument("--save-top-n",      type=int,   default=5)
    p.add_argument("--batch-size",      type=int,   default=4)
    p.add_argument("--num-workers",     type=int,   default=4)
    p.add_argument("--dropout",         type=float, default=0.3)
    p.add_argument("--learning_rate",   type=float, default=1e-5)
    p.add_argument("--weight_decay",    type=float, default=5e-4)
    p.add_argument("--warmup_steps",    type=int,   default=2000)
    p.add_argument("--grad_accum",      type=int,   default=3)
    p.add_argument("--grad_clip",       type=float, default=0.5)
    p.add_argument("--tissue_emb_dim",  type=int,   default=64,
                   help="Dimension of tissue embedding for FiLM")
    p.add_argument("--checkpoint",      type=str,   default=None)
    p.add_argument("--test",            action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# § 9  Track parser + tissue vocabulary
# ═══════════════════════════════════════════════════════════════════════════

def parse_tracks(path):
    tracks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 4: continue
            bam, species, tissue, dataset = parts[:4]
            if "ribo" in bam.lower():
                tracks.append({"bam_file": bam, "species": species,
                                "tissue": tissue, "dataset": dataset})
    return tracks


def build_tissue_vocab(tracks):
    """Deterministic tissue vocabulary (sorted alphabetically)."""
    tissues = sorted({t["tissue"] for t in tracks})
    return {tissue: idx for idx, tissue in enumerate(tissues)}


# ═══════════════════════════════════════════════════════════════════════════
# § 10  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    all_tracks      = parse_tracks(args.tracks)
    training_tracks = [t for t in all_tracks if t["dataset"] == "training"]
    if not training_tracks:
        print("ERROR: No training tracks found in", args.tracks)
        sys.exit(1)

    tissue_vocab = build_tissue_vocab(training_tracks)

    suffix = ""
    if args.trial:      suffix += f"_trial{args.trial}"
    if args.nosequence: suffix += "_nosequence"
    save_dir = (
        f"{args.save_path}/tissue_film_{args.region_len}_{args.nBins}"
        f"_bs{args.batch_size}"
        f"_lr{args.learning_rate}_wd{args.weight_decay}"
        f"_ws{args.warmup_steps}_ga{args.grad_accum}"
        f"_gc{args.grad_clip}_dp{args.dropout}"
        f"_emb{args.tissue_emb_dim}"
        f"_nt{len(tissue_vocab)}{suffix}"
    )
    args.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if _RANK == 0:
        print(f"\n{'='*60}")
        print(f"  RiboTransPred — Tissue FiLM")
        print(f"  nodes={_NNODES}  gpus/node={n_gpus}  "
              f"world_size={_WORLD_SIZE}")
        print(f"  region_len={args.region_len}  nBins={args.nBins}")
        print(f"  batch={args.batch_size}  accum={args.grad_accum}  "
              f"eff_batch={args.batch_size * args.grad_accum * _WORLD_SIZE}")
        print(f"  tissue_emb_dim={args.tissue_emb_dim}")
        print(f"  tissues ({len(tissue_vocab)}):")
        for tname, tid in tissue_vocab.items():
            print(f"    {tid}: {tname}")
        species = sorted({t['species'] for t in training_tracks})
        print(f"  species: {species}")
        print(f"  biotype: {args.biotype}")
        print(f"{'='*60}\n")

    # Init dist early for barriers
    if _WORLD_SIZE > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=_WORLD_SIZE, rank=_RANK,
            timeout=datetime.timedelta(seconds=7200))

    convert_all_tracks(training_tracks, args)

    if args.trial:
        val_chroms = [str(args.trial)]
    else:
        val_chroms = compute_shared_val_chromosomes(
            training_tracks, args, args.biotype)
    if _RANK == 0:
        print(f"  Validation chromosomes: {val_chroms}\n")

    dm    = TissueRiboDataModule(args, training_tracks, val_chroms, tissue_vocab)
    model = TissueRiboModel(args, tissue_vocab)
    ckpt  = (args.checkpoint
             if args.checkpoint and os.path.exists(args.checkpoint)
             else None)

    callbacks = [
        plc.EarlyStopping(monitor="val/loss_epoch", patience=args.patience,
                          mode="min", verbose=(_RANK == 0)),
        plc.LearningRateMonitor(logging_interval="step"),
        plc.ModelCheckpoint(
            dirpath=save_dir,
            filename="epoch={epoch}-val_loss={val/loss_epoch:.4f}",
            monitor="val/loss_epoch", mode="min",
            save_top_k=args.save_top_n, save_last=True,
            auto_insert_metric_name=False),
    ]
    if _RANK == 0:
        callbacks.append(plc.RichProgressBar())

    world_size = _NNODES * n_gpus
    if world_size > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=False,
            process_group_backend="nccl",
            timeout=datetime.timedelta(seconds=7200),
            cluster_environment=SLURMEnvironment(auto_requeue=False))
    else:
        strategy = "auto"

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=n_gpus,
        num_nodes=_NNODES,
        strategy=strategy,
        max_epochs=args.max_epochs,
        precision="bf16-mixed",
        gradient_clip_val=args.grad_clip,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.grad_accum,
        callbacks=callbacks,
        logger=pl.loggers.CSVLogger(save_dir=f"{save_dir}/csv"),
        log_every_n_steps=50,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=2,
        sync_batchnorm=False,
        enable_checkpointing=True,
        enable_progress_bar=(_RANK == 0),
        enable_model_summary=(_RANK == 0),
        deterministic=False)

    if not args.test:
        if _RANK == 0: print("=== Starting Training ===\n")
        trainer.fit(model, dm, ckpt_path=ckpt)
    else:
        if _RANK == 0: print("=== Test Mode ===\n")
        dm.setup()
        model._track_names = dm.track_names
        trainer.test(model, dm, ckpt_path=ckpt)

    if _RANK == 0:
        print("\nDone!")


if __name__ == "__main__":
    main()
  
