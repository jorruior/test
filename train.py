#!/usr/bin/env python3
"""
RiboTransPred — Predict Ribo-seq profiles from RNA-seq + DNA sequence
=====================================================================
Input:  RNA-seq signal (length=region_len) + DNA sequence (one-hot)
Output: Ribo-seq signal (binned to nbins)

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
    """Debug print that works on ALL ranks (writes to stderr)."""
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

import model.models as models

if torch.cuda.is_available():
    torch.cuda.set_device(_LOCAL_RANK)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision("high")
    print(f"Rank {_RANK}: cuda:{_LOCAL_RANK} -> "
          f"{torch.cuda.get_device_name(_LOCAL_RANK)}")
else:
    print(f"Rank {_RANK}: WARNING - running on CPU")

os.environ.update({
    "LOCAL_RANK":  str(_LOCAL_RANK),
    "RANK":        str(_RANK),
    "WORLD_SIZE":  str(_WORLD_SIZE),
    "NODE_RANK":   str(os.environ.get("SLURM_NODEID", "0")),
    "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", "ibp23s0"),
})


# ═══════════════════════════════════════════════════════════════════════════
# § 1  File conversion:  .pt -> .npy
# ═══════════════════════════════════════════════════════════════════════════

def _pt_to_npy(pt_path):
    npy_path = pt_path.rsplit(".", 1)[0] + ".npy"
    if os.path.exists(npy_path):
        return npy_path
    if not os.path.exists(pt_path):
        return None
    try:
        t = torch.load(pt_path, map_location="cpu", weights_only=True)
        tmp = f"{npy_path}.tmp.{os.getpid()}"
        np.save(tmp, t.numpy())
        del t
        gc.collect()
        os.replace(tmp, npy_path)
        print(f"  converted {os.path.basename(pt_path)}")
        return npy_path
    except Exception as e:
        _dbg(f"WARNING: cannot convert {pt_path}: {e}")
        return None


def _npy_paths(tracks_dir, species, tissue, region_len, nbins, psites):
    bn = f"{species}_{tissue}"
    d  = f"{tracks_dir}/{species}/{tissue}"
    inp = f"{d}/{bn}_rna_{region_len}_log_rnaseq_final.npy"
    tag = "ribo.psites" if psites else "ribo"
    out = f"{d}/{bn}_{tag}_{region_len}_{nbins}_log_riboseq_final.npy"
    return inp, out


def _pt_paths(tracks_dir, species, tissue, region_len, nbins, psites):
    bn = f"{species}_{tissue}"
    d  = f"{tracks_dir}/{species}/{tissue}"
    inp = f"{d}/{bn}_rna_{region_len}_log_rnaseq_final.pt"
    tag = "ribo.psites" if psites else "ribo"
    out = f"{d}/{bn}_{tag}_{region_len}_{nbins}_log_riboseq_final.pt"
    return inp, out


def convert_all_tracks(training_tracks, args):
    if _LOCAL_RANK == 0:
        print(f"\nNode {os.environ.get('SLURM_NODEID', '?')}: "
              f"converting .pt -> .npy ...")
        for t in training_tracks:
            inp_pt, out_pt = _pt_paths(
                args.tracks_dir, t["species"], t["tissue"],
                args.region_len, args.nBins, args.psites,
            )
            _pt_to_npy(inp_pt)
            _pt_to_npy(out_pt)
        print("  conversions done on this node")

    deadline = time.time() + 300
    while time.time() < deadline:
        all_ready = True
        for t in training_tracks:
            inp_npy, out_npy = _npy_paths(
                args.tracks_dir, t["species"], t["tissue"],
                args.region_len, args.nBins, args.psites,
            )
            if not os.path.exists(inp_npy) or not os.path.exists(out_npy):
                all_ready = False
                break
        if all_ready:
            break
        time.sleep(2)


# ═══════════════════════════════════════════════════════════════════════════
# § 2  Coordinates and chromosome splitting
# ═══════════════════════════════════════════════════════════════════════════

def load_coords(species):
    """
    Load ALL transcripts from the coordinate file.
    Returns DataFrame with columns: chr, start, end, id, biotype, sequence, species.

    The mmap .npy files were built from the FULL coordinate file (all biotypes),
    so we must load all rows to keep row indices aligned with the mmap.
    Biotype filtering is done later on the indices, not here.
    """
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
                "chr":      p[0],
                "start":    int(p[1]),
                "end":      int(p[2]),
                "id":       p[3],
                "biotype":  p[5],
                "sequence": p[7],
                "species":  species,
            })
    return pd.DataFrame(rows) if rows else None


def _norm_chr(c):
    return c[3:] if c.lower().startswith("chr") else c


def split_chromosomes(chroms, trial=None, fixed_val=None):
    chroms = [c for c in chroms if len(c) <= 7 and "MT" not in c.upper()]
    normed = {c: _norm_chr(c) for c in chroms}

    if trial is not None:
        hits = [c for c in chroms if normed[c] == _norm_chr(str(trial))]
        if not hits:
            return [], [], []
        return [hits[0]], [hits[0]], [hits[0]]

    test = []
    for want in ("16", "1", "X"):
        for c in chroms:
            if normed[c] == want and c not in test:
                test.append(c)
                break
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


def compute_shared_val_chromosomes(training_tracks, args, biotype):
    job_id = os.environ.get("SLURM_JOB_ID", "0")
    chrom_file = f"_val_chroms_{job_id}.json"

    if _RANK == 0:
        per_species = {}
        for t in training_tracks:
            df = load_coords(t["species"])
            if df is not None:
                per_species.setdefault(t["species"], set()).update(
                    _norm_chr(c) for c in df["chr"].unique()
                )
        candidates = []
        for n in range(1, 23):
            s = str(n)
            if all(s in v for v in per_species.values()):
                candidates.append(s)
            if len(candidates) == 3:
                break
        if len(candidates) < 3:
            candidates = ["1", "2", "3"]

        tmp = f"{chrom_file}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(candidates, f)
        os.replace(tmp, chrom_file)
        print(f"  Validation chromosomes: {candidates}")

    deadline = time.time() + 120
    while not os.path.exists(chrom_file):
        if time.time() > deadline:
            _dbg("WARNING: timed out waiting for val chroms, using default")
            return ["1", "2", "3"]
        time.sleep(1)

    time.sleep(1)
    with open(chrom_file) as f:
        result = json.load(f)

    if _RANK == _WORLD_SIZE - 1:
        try:
            time.sleep(2)
            os.remove(chrom_file)
        except OSError:
            pass

    return result


# ═══════════════════════════════════════════════════════════════════════════
# § 3  Dataset
# ═══════════════════════════════════════════════════════════════════════════

_OHE_TABLE = np.zeros((256, 5), dtype=np.float32)
for _i, _b in enumerate("ATCGN"):
    _OHE_TABLE[ord(_b), _i] = 1.0
for _i, _b in enumerate("atcgn"):
    _OHE_TABLE[ord(_b), _i] = 1.0


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


class TranscriptDataset(Dataset):
    """
    Mmap-backed dataset.  Pre-filters all-N transcripts at __init__
    so __getitem__ NEVER returns None.
    """

    def __init__(self, sequences, index_array, inp_mmap, out_mmap,
                 seq_len, nbins, nosequence=False):
        self.sequences  = sequences
        self.inp_mmap   = inp_mmap
        self.out_mmap   = out_mmap
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
            "features": torch.from_numpy(features),
            "target":   torch.from_numpy(target),
            "mask":     torch.from_numpy(mask),
        }


# ═══════════════════════════════════════════════════════════════════════════
# § 4  DataModule
# ═══════════════════════════════════════════════════════════════════════════

def _biotype_mask(df_all, biotype):
    """
    Return a boolean numpy array over df_all rows indicating which
    transcripts match the requested biotype.
    """
    if biotype == "protein_coding":
        return (df_all["biotype"] == "protein_coding").values
    elif biotype == "non_coding":
        return df_all["biotype"].isin(["lncRNA", "lincRNA"]).values
    else:  # "all"
        return np.ones(len(df_all), dtype=bool)


class RiboDataModule(LightningDataModule):

    def __init__(self, args, training_tracks, val_chroms):
        super().__init__()
        self.args            = args
        self.training_tracks = training_tracks
        self.val_chroms_norm = set(val_chroms)
        self._train_ds = None
        self._val_ds   = None
        self._mmap_refs = []

    def setup(self, stage=None):
        if self._train_ds is not None:
            return

        _dbg(f"setup() cwd={os.getcwd()} tracks_dir={self.args.tracks_dir} "
             f"n_tracks={len(self.training_tracks)} biotype={self.args.biotype}")

        # Wait for npy files
        deadline = time.time() + 300
        while time.time() < deadline:
            all_ready = True
            for t in self.training_tracks:
                inp_npy, out_npy = _npy_paths(
                    self.args.tracks_dir, t["species"], t["tissue"],
                    self.args.region_len, self.args.nBins, self.args.psites,
                )
                if not os.path.exists(inp_npy) or not os.path.exists(out_npy):
                    all_ready = False
                    break
            if all_ready:
                break
            time.sleep(2)

        train_parts, val_parts = [], []

        for track in self.training_tracks:
            sp, ti = track["species"], track["tissue"]
            inp_npy, out_npy = _npy_paths(
                self.args.tracks_dir, sp, ti,
                self.args.region_len, self.args.nBins, self.args.psites,
            )
            if not os.path.exists(inp_npy) or not os.path.exists(out_npy):
                _dbg(f"SKIP {sp}/{ti}: missing npy "
                     f"(inp={os.path.exists(inp_npy)} "
                     f"out={os.path.exists(out_npy)})")
                continue

            # ── Load ALL transcripts (mmap was built from full file) ──────
            df_all = load_coords(sp)
            if df_all is None or df_all.empty:
                _dbg(f"SKIP {sp}/{ti}: no coordinates")
                continue

            # ── Open mmap files ───────────────────────────────────────────
            inp_mm = np.load(inp_npy, mmap_mode="r")
            out_mm = np.load(out_npy, mmap_mode="r")
            self._mmap_refs.extend([inp_mm, out_mm])

            if inp_mm.shape[0] != len(df_all):
                _dbg(f"SKIP {sp}/{ti}: shape mismatch "
                     f"mmap={inp_mm.shape[0]} coords_all={len(df_all)}")
                continue

            # ── Biotype filter (on indices, not on the DataFrame) ─────────
            bt_ok = _biotype_mask(df_all, self.args.biotype)
            n_bt  = bt_ok.sum()

            # ── Chromosome split ──────────────────────────────────────────
            sequences   = df_all["sequence"].values
            norm_chroms = np.array([_norm_chr(c) for c in df_all["chr"].values])

            train_chr, val_chr, _ = split_chromosomes(
                df_all["chr"].unique().tolist(),
                trial=self.args.trial,
                fixed_val=list(self.val_chroms_norm),
            )
            tr_set = {_norm_chr(c) for c in train_chr}
            vl_set = {_norm_chr(c) for c in val_chr}

            # Indices must satisfy BOTH chromosome AND biotype filters
            chr_tr = np.isin(norm_chroms, list(tr_set))
            chr_vl = np.isin(norm_chroms, list(vl_set))

            tr_idx = np.where(chr_tr & bt_ok)[0]
            vl_idx = np.where(chr_vl & bt_ok)[0]

            if len(tr_idx):
                ds = TranscriptDataset(
                    sequences, tr_idx, inp_mm, out_mm,
                    self.args.region_len, self.args.nBins, self.args.nosequence,
                )
                train_parts.append(ds)

            if len(vl_idx):
                ds = TranscriptDataset(
                    sequences, vl_idx, inp_mm, out_mm,
                    self.args.region_len, self.args.nBins, self.args.nosequence,
                )
                val_parts.append(ds)

            n_tr = len(train_parts[-1]) if len(tr_idx) else 0
            n_vl = len(val_parts[-1])   if len(vl_idx) else 0
            _dbg(f"LOADED {sp}/{ti}: total={len(df_all)} "
                 f"biotype_match={n_bt} train={n_tr} val={n_vl}")

        self._train_ds = ConcatDataset(train_parts) if train_parts else None
        self._val_ds   = ConcatDataset(val_parts)   if val_parts   else None

        if self._train_ds is None:
            _dbg("FATAL: No training data loaded!")
            for t in self.training_tracks:
                sp, ti = t["species"], t["tissue"]
                inp_npy, out_npy = _npy_paths(
                    self.args.tracks_dir, sp, ti,
                    self.args.region_len, self.args.nBins, self.args.psites,
                )
                _dbg(f"  {sp}/{ti}: inp={os.path.exists(inp_npy)} "
                     f"out={os.path.exists(out_npy)} "
                     f"coords={os.path.exists(f'coordinates/{sp}_coordinates.txt')}")
            raise RuntimeError("No training data loaded.")

        total_tr = len(self._train_ds)
        total_vl = len(self._val_ds) if self._val_ds else 0
        _dbg(f"Total: {total_tr} train, {total_vl} val")
        print(f"\n  Total dataset: {total_tr} train, {total_vl} val\n")

    def _make_loader(self, dataset, shuffle, drop_last):
        sampler = None
        try:
            world = self.trainer.world_size
        except Exception:
            world = _WORLD_SIZE
        if world > 1:
            sampler = DistributedSampler(
                dataset, shuffle=shuffle, drop_last=drop_last,
            )
            shuffle = False

        nw = self.args.num_workers
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
            timeout=600,
        )

    def train_dataloader(self):
        if self._train_ds is None:
            raise RuntimeError("setup() not called or no train data")
        return self._make_loader(self._train_ds, shuffle=True, drop_last=True)

    def val_dataloader(self):
        if self._val_ds is None or len(self._val_ds) == 0:
            return None
        # drop_last=True: all ranks must process same number of batches
        return self._make_loader(self._val_ds, shuffle=False, drop_last=True)

# ═══════════════════════════════════════════════════════════════════════════
# § 5  Pearson R metric
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
        p = (pred   * mask).reshape(-1)[m].double()
        t = (target * mask).reshape(-1)[m].double()
        n = p.numel()
        if n < 2:
            return
        self.sum_xy += (p * t).sum()
        self.sum_x  += p.sum()
        self.sum_y  += t.sum()
        self.sum_x2 += (p * p).sum()
        self.sum_y2 += (t * t).sum()
        self.count  += n

    def compute(self):
        n   = self.count.clamp(min=2)
        cov = self.sum_xy - self.sum_x * self.sum_y / n
        vx  = self.sum_x2 - self.sum_x ** 2 / n
        vy  = self.sum_y2 - self.sum_y ** 2 / n
        denom = torch.sqrt((vx * vy).clamp(min=1e-16))
        return (cov / denom).float()


# ═══════════════════════════════════════════════════════════════════════════
# § 6  Per-sample PCC loss
# ═══════════════════════════════════════════════════════════════════════════

def _per_sample_pcc_loss(pred, target, mask):
    p = pred * mask
    t = target * mask
    n = mask.sum(dim=1, keepdim=True).clamp(min=1)

    p_mean = p.sum(dim=1, keepdim=True) / n
    t_mean = t.sum(dim=1, keepdim=True) / n

    p_c = (p - p_mean) * mask
    t_c = (t - t_mean) * mask

    cov = (p_c * t_c).sum(dim=1)
    vp  = (p_c ** 2).sum(dim=1)
    vt  = (t_c ** 2).sum(dim=1)

    denom = torch.sqrt((vp * vt).clamp(min=1e-12))
    r = cov / denom

    valid = (n.squeeze(1) >= 10)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    return 1.0 - r[valid].mean()


# ═══════════════════════════════════════════════════════════════════════════
# § 7  LightningModule
# ═══════════════════════════════════════════════════════════════════════════

class RiboModel(LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters({
            "model_type":    args.model_type,
            "region_len":    args.region_len,
            "nBins":         args.nBins,
            "n_heads":       args.n_heads,
            "dropout":       args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay":  args.weight_decay,
            "warmup_steps":  args.warmup_steps,
            "batch_size":    args.batch_size,
            "grad_accum":    args.grad_accum,
            "nosequence":    args.nosequence,
            "seed":          args.seed,
        })
        self.args = args

        ModelCls = getattr(models, args.model_type)
        self.net = ModelCls(
            num_genomic_features=1,
            target_length=args.region_len,
            nbins=args.nBins,
            n_heads=args.n_heads,
            dropout=args.dropout,
        )

        self.train_pcc = StreamingPearsonR()
        self.val_pcc   = StreamingPearsonR()

        self._tr_loss_sum = 0.0
        self._tr_loss_n   = 0
        self._vl_loss_sum = 0.0
        self._vl_loss_n   = 0
        self._history     = []

        self._zero_w     = 0.1 if args.nBins < args.region_len else 0.3
        self._pcc_loss_w = 0.2

    def forward(self, x):
        return self.net(x)

    def _compute_loss(self, pred, target, mask):
        w = torch.where(
            target.abs() < 1e-6,
            torch.full_like(target, self._zero_w),
            torch.ones_like(target),
        )
        w = w * mask
        mse = ((pred - target) ** 2 * w).sum() / (w.sum() + 1e-8)
        pcc_loss = _per_sample_pcc_loss(pred, target, mask)
        return mse + self._pcc_loss_w * pcc_loss

    # ------------------------------------------------------------------
    # ALL self.log() use sync_dist=False to avoid NCCL allreduce hangs.
    # Each rank logs its own local value.  Callbacks see it on every rank.
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        features = batch["features"].float()
        target   = batch["target"].float()
        mask     = batch["mask"].float()

        pred = self(features)
        loss = self._compute_loss(pred, target, mask)

        if torch.isnan(loss) or torch.isinf(loss):
            _dbg(f"WARNING: NaN/Inf loss at step {self.global_step}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        self._tr_loss_sum += loss.detach().item()
        self._tr_loss_n   += 1

        with torch.no_grad():
            self.train_pcc.update(pred.detach(), target, mask)

        if batch_idx % 50 == 0:
            self.log("train/loss_step", loss,
                     prog_bar=True, sync_dist=False)

        return loss

    def validation_step(self, batch, batch_idx):
        features = batch["features"].float()
        target   = batch["target"].float()
        mask     = batch["mask"].float()

        pred = self(features)
        loss = self._compute_loss(pred, target, mask)

        self._vl_loss_sum += loss.item()
        self._vl_loss_n   += 1
        self.val_pcc.update(pred, target, mask)

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
                "val_loss": avg,
                "val_pcc": pcc,
            })
        self.val_pcc.reset()
        self._vl_loss_sum = 0.0
        self._vl_loss_n   = 0

    def on_train_end(self):
        if self.global_rank == 0 and self._history:
            df = pd.DataFrame(self._history)
            p = os.path.join(self.args.save_dir, "training_history.csv")
            df.to_csv(p, index=False)
            print(f"\n  History saved -> {p}")

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "bn" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        opt = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": self.args.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        total_steps  = self.trainer.estimated_stepping_batches
        warmup_steps = min(self.args.warmup_steps, max(100, total_steps // 10))
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


# ═══════════════════════════════════════════════════════════════════════════
# § 8  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("RiboTransPred")
    p.add_argument("--seed",           type=int,   default=4)
    p.add_argument("--save_path",      default="results")
    p.add_argument("--tracks",         default="tracks.txt")
    p.add_argument("--tracks_dir",     default="tracks")
    p.add_argument("--model-type",     default="PosTransModelTCN",
                   choices=["PosTransModelTCN", "PosTransModel"])
    p.add_argument("--region_len",     type=int,   default=6000)
    p.add_argument("--nBins",          type=int,   default=1000)
    p.add_argument("--biotype",        default="protein_coding",
                   choices=["protein_coding", "non_coding", "all"])
    p.add_argument("--psites",         action="store_true")
    p.add_argument("--trial",          type=str,   default=None)
    p.add_argument("--nosequence",     action="store_true")
    p.add_argument("--patience",       type=int,   default=8)
    p.add_argument("--max-epochs",     type=int,   default=80)
    p.add_argument("--save-top-n",     type=int,   default=5)
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--num-workers",    type=int,   default=4)
    p.add_argument("--n_heads",        type=int,   default=6)
    p.add_argument("--dropout",        type=float, default=0.3)
    p.add_argument("--learning_rate",  type=float, default=1e-5)
    p.add_argument("--weight_decay",   type=float, default=5e-4)
    p.add_argument("--warmup_steps",   type=int,   default=2000)
    p.add_argument("--grad_accum",     type=int,   default=3)
    p.add_argument("--grad_clip",      type=float, default=0.5)
    p.add_argument("--checkpoint",     type=str,   default=None)
    p.add_argument("--test",           action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# § 9  Track file parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_tracks(path):
    tracks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            bam, species, tissue, dataset = parts[:4]
            if "ribo" in bam.lower():
                tracks.append({
                    "bam_file": bam, "species": species,
                    "tissue": tissue, "dataset": dataset,
                })
    return tracks


# ═══════════════════════════════════════════════════════════════════════════
# § 10  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)

    suffix = ""
    if args.trial:      suffix += f"_trial{args.trial}"
    if args.nosequence: suffix += "_nosequence"
    save_dir = (
        f"{args.save_path}/model_{args.region_len}_{args.nBins}"
        f"_bs{args.batch_size}_h{args.n_heads}"
        f"_lr{args.learning_rate}_wd{args.weight_decay}"
        f"_ws{args.warmup_steps}_ga{args.grad_accum}"
        f"_gc{args.grad_clip}_dp{args.dropout}"
        f"_{args.model_type}{suffix}"
    )
    args.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    all_tracks      = parse_tracks(args.tracks)
    training_tracks = [t for t in all_tracks if t["dataset"] == "training"]
    if not training_tracks:
        print("ERROR: No training tracks found in", args.tracks)
        sys.exit(1)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    _dbg(f"main() all_tracks={len(all_tracks)} "
         f"training_tracks={len(training_tracks)}")

    if _RANK == 0:
        print(f"\n{'='*60}")
        print(f"  RiboTransPred  |  {args.model_type}")
        print(f"  nodes={_NNODES}  gpus/node={n_gpus}  "
              f"world_size={_WORLD_SIZE}")
        print(f"  region_len={args.region_len}  nBins={args.nBins}")
        print(f"  batch={args.batch_size}  accum={args.grad_accum}  "
              f"eff_batch={args.batch_size * args.grad_accum * _WORLD_SIZE}")
        species = sorted({t['species'] for t in training_tracks})
        print(f"  species: {species}")
        print(f"  tracks:  {len(training_tracks)} training")
        print(f"  biotype: {args.biotype}")
        print(f"{'='*60}\n")

    convert_all_tracks(training_tracks, args)

    if args.trial:
        val_chroms = [str(args.trial)]
    else:
        val_chroms = compute_shared_val_chromosomes(
            training_tracks, args, args.biotype,
        )
    if _RANK == 0:
        print(f"  Validation chromosomes: {val_chroms}\n")

    dm = RiboDataModule(args, training_tracks, val_chroms)

    model = RiboModel(args)
    ckpt = (args.checkpoint
            if args.checkpoint and os.path.exists(args.checkpoint)
            else None)

    # ALL callbacks on ALL ranks — Lightning gates I/O to rank-0
    callbacks = [
        plc.EarlyStopping(
            monitor="val/loss_epoch",
            patience=args.patience,
            mode="min",
            verbose=(_RANK == 0),
        ),
        plc.LearningRateMonitor(logging_interval="step"),
        plc.ModelCheckpoint(
            dirpath=save_dir,
            filename="epoch={epoch}-val_loss={val/loss_epoch:.4f}",
            monitor="val/loss_epoch",
            mode="min",
            save_top_k=args.save_top_n,
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]
    if _RANK == 0:
        callbacks.append(plc.RichProgressBar())

    world_size = _NNODES * n_gpus
    if world_size > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            process_group_backend="nccl",
            timeout=datetime.timedelta(seconds=7200),
            cluster_environment=SLURMEnvironment(auto_requeue=False),
        )
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
        deterministic=False,
    )

    if not args.test:
        if _RANK == 0:
            print("=== Starting Training ===\n")
        trainer.fit(model, dm, ckpt_path=ckpt)
    else:
        if _RANK == 0:
            print("=== Test Mode ===\n")
        trainer.test(model, dm, ckpt_path=ckpt)

    if _RANK == 0:
        print("\nDone!")


if __name__ == "__main__":
    main()
