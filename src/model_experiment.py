"""HiEnWrite VGG19 vs ViT-B16 training, cross-validation, and evaluation.

Extracted in notebook cell order from notebooks/00_full_kaggle_experiment.ipynb.
This script intentionally preserves the original notebook workflow and shared variables.
Run in a Kaggle notebook/IPython session with the required dataset and artifacts attached.
"""

from __future__ import annotations
from IPython.display import display


# ================== ORIGINAL NOTEBOOK CELL 34 ==================
import os, sys, json, time, glob, shutil, hashlib, random, gc, warnings, platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

# ======================= KONTROL UTAMA =======================
SMOKE_TEST = True            # Selaraskan dengan default notebook: uji cepat (2 epoch, fold 0); False untuk eksperimen penuh
TIME_BUDGET_HOURS = None    # mis. 10.0 -> berhenti rapi sebelum batas sesi; lanjutkan di sesi berikutnya
ARTIFACT_DIR = None         # mis. "/kaggle/input/nama-input/hienwrite_clean" bila pencarian otomatis gagal

# ======================= SESUAI PROPOSAL (Tabel 6 & 7) =======================
SEED       = 42
N_FOLDS    = 5
MAX_EPOCHS = 50
PATIENCE   = 10             # early stopping pada RMSE validasi
EPS        = 1e-8           # Persamaan (2)
ARCHS      = ["vgg19", "vit_b16"]
OPTIMIZERS = ["adam", "adamw", "sgd"]          # adam = baseline
LR = {"vgg19":   {"adam": 1e-3, "adamw": 1e-3, "sgd": 1e-3},
      "vit_b16": {"adam": 1e-4, "adamw": 1e-4, "sgd": 1e-3}}
WD = {"adam": 0.0, "adamw": 0.01, "sgd": 1e-4}
BETAS, MOMENTUM = (0.9, 0.999), 0.9

# Tabel 7: batch size 16 (kandidat: 16 dan 32). Tahap 1 = matriks 6 konfigurasi pada bs=16.
# Tahap 2 (RUN_BS32=True) = kandidat bs=32 untuk optimizer terbaik tiap arsitektur; bs final dipilih dari RMSE validasi.
BATCH_SIZES_STAGE1 = [16]   # [16, 32] = matriks penuh 12 konfigurasi (2x lebih berat)
RUN_BS32 = True             # False = bs tetap 16 (nyatakan demikian di naskah)

# ======================= SUMBER BOBOT PRETRAINED =======================
WEIGHTS_INPUT_DIR = "/kaggle/input"
ALLOW_WEIGHT_DOWNLOADS = False      # False untuk memakai bobot dari Input/cache saja

# ======================= PILIHAN IMPLEMENTASI =======================
VIT_CKPT = "google/vit-base-patch16-224"   # ViT-B/16, pretrain ImageNet-21k, fine-tune ImageNet-1k
VIT_RENORM_TO_HALF = False  # False = ikuti proposal (ImageNet mean/std untuk kedua model).
                            # True  = ViT menerima mean=std=0.5 (normalisasi asli checkpoint) via konversi di dalam model.
USE_AMP = True              # mixed precision (hanya GPU); mempercepat tanpa mengubah desain eksperimen
SAVE_FINAL_WEIGHTS = True

if SMOKE_TEST:
    MAX_EPOCHS = 2
FOLDS = [0] if SMOKE_TEST else list(range(N_FOLDS))
TRAITS = ["O", "C", "E", "A", "N"]
TRAIT_NAMES = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion", "A": "Agreeableness", "N": "Neuroticism"}
LABEL_MAX = 40.0

# ---- lokasi artefak Bagian A & folder hasil ----
def find_artifact_dir():
    if ARTIFACT_DIR:
        return Path(ARTIFACT_DIR)
    for base in ["/kaggle/input", "/kaggle/working", "."]:
        hits = glob.glob(os.path.join(base, "**", "hienwrite_clean", "split_manifest.csv"), recursive=True)
        if hits:
            return Path(hits[0]).parent
    raise FileNotFoundError("split_manifest.csv tidak ditemukan. Tambahkan output Bagian A sebagai Input atau isi ARTIFACT_DIR.")

ART = OUT_DIR if ("OUT_DIR" in globals() and (OUT_DIR / "split_manifest.csv").exists()) else find_artifact_dir()   # Bagian A bila baru dijalankan; selain itu cari di Input
RUN_DIR = Path("/kaggle/working" if os.path.isdir("/kaggle/working") else ".") / ("hienwrite_runs_smoke" if SMOKE_TEST else "hienwrite_runs")
CV_DIR, FINAL_DIR, TAB_DIR = RUN_DIR / "cv", RUN_DIR / "final", RUN_DIR / "tables"
for d in (CV_DIR, FINAL_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)
print("Artefak data :", ART)
print("Folder hasil :", RUN_DIR, "| SMOKE_TEST =", SMOKE_TEST, "| MAX_EPOCHS =", MAX_EPOCHS, "| folds =", FOLDS)


# ================== ORIGINAL NOTEBOOK CELL 35 ==================
# ---- Lanjutkan dari sesi sebelumnya: salin checkpoint JSON dari output notebook ini yang dipasang sebagai Input ----
def import_previous_runs():
    if SMOKE_TEST:
        return
    copied = 0
    for src in glob.glob("/kaggle/input/**/" + RUN_DIR.name, recursive=True):
        src = Path(src)
        if src.resolve() == RUN_DIR.resolve():
            continue
        for f in src.rglob("*"):
            if f.is_file() and f.suffix != ".pt":                 # bobot tidak diperlukan untuk resume
                dst = RUN_DIR / f.relative_to(src)
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst); copied += 1
    print(f"Checkpoint dari sesi sebelumnya yang diimpor: {copied} berkas")
import_previous_runs()
print("Unit CV yang sudah ada:", len(list(CV_DIR.glob('*.json'))), "| model akhir yang sudah ada:", len(list(FINAL_DIR.glob('*.json'))))


# ================== ORIGINAL NOTEBOOK CELL 37 ==================
manifest = pd.read_csv(ART / "split_manifest.csv")
X = np.load(ART / "images_224.npy")                                   # (N,224,224,3) uint8
assert len(X) == len(manifest) and (manifest.img_idx.values == np.arange(len(manifest))).all()
assert X.dtype == np.uint8 and X.shape[1:] == (224, 224, 3)

Y = manifest[[f"{t}_s" for t in TRAITS]].to_numpy(np.float32)         # target skala 0-1 (urutan O,C,E,A,N)
is_train = (manifest.split == "train").to_numpy()
fold_arr = manifest.fold.to_numpy()
assert Y.min() >= 0 and Y.max() <= 1

# pemeriksaan ulang ringkas anti-kebocoran
tr, te = manifest[is_train], manifest[~is_train]
assert set(tr.writer_id).isdisjoint(set(te.writer_id))
assert tr.groupby("writer_id").fold.nunique().max() == 1 and (te.fold == -1).all()
assert sorted(tr.fold.unique()) == list(range(N_FOLDS))

# normalisasi: ikuti Bagian A bila config.json tersedia
cfg1 = {}
if (ART / "config.json").exists():
    cfg1 = json.loads((ART / "config.json").read_text()).get("config", {})
IMAGENET_MEAN = cfg1.get("IMAGENET_MEAN", [0.485, 0.456, 0.406])
IMAGENET_STD  = cfg1.get("IMAGENET_STD",  [0.229, 0.224, 0.225])
assert not cfg1.get("AUGMENTATION", False), "Proposal V12 tidak memakai augmentasi"
assert cfg1.get("LABEL_MAX", LABEL_MAX) == LABEL_MAX and cfg1.get("SEED", SEED) == SEED, "Konfigurasi berbeda dari Bagian A"
MANIFEST_MD5 = hashlib.md5(manifest[["file", "split", "fold"]].to_csv(index=False).encode()).hexdigest()   # hash isi pembagian data (stabil lintas sesi)

print(f"Citra: {len(manifest)} | latih: {int(is_train.sum())} ({tr.writer_id.nunique()} penulis) | uji: {int((~is_train).sum())} ({te.writer_id.nunique()} penulis)")
print("Ukuran fold (citra):", {int(k): int(v) for k, v in tr.groupby('fold').size().items()})
print("manifest md5:", MANIFEST_MD5[:10])


# ================== ORIGINAL NOTEBOOK CELL 39 ==================
def rmse_from_err(err):
    """err: (n,5) selisih prediksi-aktual. Mengembalikan RMSE keseluruhan (Pers. 3) dan per dimensi (Pers. 4)."""
    mse_k = (err ** 2).mean(axis=0)
    return float(np.sqrt(mse_k.mean())), [float(v) for v in np.sqrt(mse_k)]

base_rows = []
for k in FOLDS:
    trm, vam = is_train & (fold_arr != k), is_train & (fold_arr == k)
    r, rk = rmse_from_err(Y[vam] - Y[trm].mean(0))
    base_rows.append(dict(fold=k, rmse=r, **{f"rmse_{t}": v for t, v in zip(TRAITS, rk)}))
base_cv = pd.DataFrame(base_rows)
BASE = dict(val_mean=base_cv.rmse.mean(), val_sd=base_cv.rmse.std(ddof=1) if len(base_cv) > 1 else float("nan"),
            val_trait=base_cv[[f"rmse_{t}" for t in TRAITS]].mean().values.tolist())
print(f"Baseline rata-rata | RMSE validasi (skala 0-1): {BASE['val_mean']:.4f} ± {BASE['val_sd']:.4f}"
      f"  (= {BASE['val_mean']*LABEL_MAX:.2f} pada skala 0-40)")
print("(Baseline untuk data uji baru dihitung di section B10, agar data uji tidak disentuh sebelum seluruh CV selesai.)")


# ================== ORIGINAL NOTEBOOK CELL 41 ==================
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision.models import VGG19_Weights, vgg19
from functools import lru_cache
from urllib.error import URLError

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ON = bool(USE_AMP and device.type == "cuda")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | device: {device}"
      + (f" ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)" if device.type == "cuda" else ""))
try:
    import transformers; print("transformers", transformers.__version__)
except ImportError:
    print("transformers belum terpasang -> jalankan: !pip install -q transformers")
if device.type != "cuda":
    print("⚠️ GPU tidak terdeteksi. Aktifkan Accelerator GPU; pelatihan di CPU akan sangat lambat.")

# ------------------- Dataset (tanpa augmentasi; Bagian 3 proposal) -------------------
# Setara dengan ToTensor (uint8 -> 0-1) + Normalize ImageNet, dihitung sekali agar pelatihan cepat dan deterministik.
_mean = np.array(IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1); _std = np.array(IMAGENET_STD, np.float32).reshape(1, 3, 1, 1)
X_t = torch.from_numpy(((X.transpose(0, 3, 1, 2).astype(np.float32) / 255.0) - _mean) / _std)
Y_t = torch.from_numpy(Y)
print("Tensor masukan:", tuple(X_t.shape), X_t.dtype, f"({X_t.numel() * 4 / 1e6:.0f} MB)")

def make_loader(idx, bs, shuffle, seed):
    g = torch.Generator(); g.manual_seed(seed); idx = torch.as_tensor(idx)
    return DataLoader(TensorDataset(X_t[idx], Y_t[idx]), batch_size=bs, shuffle=shuffle, generator=g, pin_memory=(device.type == "cuda"))

def cv_loaders(fold, bs):
    return (make_loader(np.flatnonzero(is_train & (fold_arr != fold)), bs, True, SEED + fold),
            make_loader(np.flatnonzero(is_train & (fold_arr == fold)), bs, False, SEED))

def final_loaders(bs):
    return (make_loader(np.flatnonzero(is_train), bs, True, SEED + 99), make_loader(np.flatnonzero(~is_train), bs, False, SEED))

# ------------------- Sumber bobot pretrained -------------------
VGG19_PRETRAINED = VGG19_Weights.IMAGENET1K_V1
VGG19_FILENAME = Path(VGG19_PRETRAINED.url).name
VGG19_SHA256 = "dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44"
VIT_FILES = ("config.json", "model.safetensors")


def nonempty_file(path):
    return path.is_file() and path.stat().st_size > 0


def has_vit_weights(folder):
    return all(nonempty_file(folder / name) for name in VIT_FILES)


def find_weight_input(name, is_ready):
    root = Path(WEIGHTS_INPUT_DIR)
    matches = [path for path in sorted(root.rglob(name)) if is_ready(path)]
    if len(matches) > 1:
        raise ValueError(f"Ada beberapa salinan {name} di Input. Isi WEIGHTS_INPUT_DIR di B1 dengan path dataset bobot yang dipilih.")
    return matches[0] if matches else None


def weight_help(files):
    return (
        f"Tambahkan {files} lewat Add Input sesuai panduan di bagian atas notebook, "
        "atau aktifkan Internet di Session options dan set ALLOW_WEIGHT_DOWNLOADS = True di B1."
    )


@lru_cache(maxsize=1)
def vgg19_weights_file():
    path = find_weight_input(VGG19_FILENAME, nonempty_file)
    if path is None:
        path = Path(torch.hub.get_dir()) / "checkpoints" / VGG19_FILENAME
    if nonempty_file(path):
        digest = hashlib.sha256()
        with path.open("rb") as weights:
            for block in iter(lambda: weights.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != VGG19_SHA256:
            raise ValueError(f"Checksum VGG19 tidak cocok untuk {path}. Ganti dengan berkas resmi dari panduan notebook.")
    else:
        if not ALLOW_WEIGHT_DOWNLOADS:
            raise FileNotFoundError(f"Bobot VGG19 belum tersedia. {weight_help(VGG19_FILENAME)}")
        print(f"Mengunduh bobot VGG19: {VGG19_PRETRAINED.url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.hub.download_url_to_file(VGG19_PRETRAINED.url, str(path), hash_prefix=VGG19_SHA256)
        except (URLError, TimeoutError, ConnectionError) as exc:
            raise RuntimeError(f"Unduhan bobot VGG19 gagal ({exc}). {weight_help(VGG19_FILENAME)}") from exc
    print(f"VGG19 | bobot: {path}")
    return path


@lru_cache(maxsize=1)
def vit_weights_folder():
    folder_name = VIT_CKPT.rsplit("/", 1)[-1]
    folder = find_weight_input(folder_name, has_vit_weights)
    if folder is None:
        # fallback: cari folder mana pun di Input yang langsung berisi config.json + model.safetensors
        root = Path(WEIGHTS_INPUT_DIR)
        cands = sorted({p.parent for p in root.rglob("model.safetensors") if has_vit_weights(p.parent)})
        if len(cands) > 1:
            raise ValueError(f"Ada beberapa folder bobot ViT di Input: {cands}. Isi WEIGHTS_INPUT_DIR di B1 dengan dataset yang dipilih.")
        folder = cands[0] if cands else None
    if folder is None:
        from huggingface_hub import snapshot_download, try_to_load_from_cache

        cached_weights = try_to_load_from_cache(VIT_CKPT, "model.safetensors")
        if isinstance(cached_weights, str) and has_vit_weights(Path(cached_weights).parent):
            folder = Path(cached_weights).parent

        if folder is None:
            files = f"folder {folder_name} berisi config.json dan model.safetensors"
            if not ALLOW_WEIGHT_DOWNLOADS:
                raise FileNotFoundError(f"Bobot ViT-B16 belum tersedia. {weight_help(files)}")
            print(f"Mengunduh bobot ViT-B16: {VIT_CKPT}")
            try:
                folder = Path(snapshot_download(repo_id=VIT_CKPT, allow_patterns=list(VIT_FILES)))
            except Exception as exc:
                raise RuntimeError(f"Unduhan bobot ViT-B16 gagal ({exc}). {weight_help(files)}") from exc
            if not has_vit_weights(folder):
                raise FileNotFoundError(f"Berkas ViT-B16 di {folder} belum lengkap. {weight_help(files)}")
    print(f"ViT-B16 | bobot: {folder}")
    return folder


# ------------------- Model -------------------
def _head(in_dim):
    return nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 5))

class VGG19Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        weights_path = vgg19_weights_file()
        base = vgg19(weights=None, init_weights=False)
        # SHA-256 sudah diverifikasi. Gunakan pemuatan checkpoint lama seperti torchvision.
        base.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))
        self.features, self.avgpool = base.features, base.avgpool         # 16 konv 3x3 + 5 max-pool; output 512x7x7
        self.head = nn.Sequential(nn.Flatten(), _head(512 * 7 * 7))
    def forward(self, x): return self.head(self.avgpool(self.features(x)))

class ViTRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import ViTModel
        self.backbone = ViTModel.from_pretrained(
            str(vit_weights_folder()), add_pooling_layer=False, local_files_only=True, use_safetensors=True
        )
        self.head = _head(self.backbone.config.hidden_size)
        self.renorm = VIT_RENORM_TO_HALF
        self.register_buffer("m_in", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("s_in", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
    def forward(self, x):
        if self.renorm:                                    # ImageNet-normalized -> (x - 0.5) / 0.5
            x = (x * self.s_in + self.m_in - 0.5) / 0.5
        return self.head(self.backbone(pixel_values=x).last_hidden_state[:, 0])    # token [CLS]

def build_model(arch):
    return (VGG19Regressor() if arch == "vgg19" else ViTRegressor()).to(device)

def make_optimizer(model, arch, opt_name):
    lr, wd = LR[arch][opt_name], WD[opt_name]
    if opt_name == "adam":  return torch.optim.Adam(model.parameters(), lr=lr, betas=BETAS, eps=1e-8, weight_decay=0.0)
    if opt_name == "adamw": return torch.optim.AdamW(model.parameters(), lr=lr, betas=BETAS, eps=1e-8, weight_decay=wd)
    if opt_name == "sgd":   return torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM, nesterov=True, weight_decay=wd)
    raise ValueError(opt_name)

def make_scaler():
    try:    return torch.amp.GradScaler("cuda", enabled=AMP_ON)
    except (AttributeError, TypeError): return torch.cuda.amp.GradScaler(enabled=AMP_ON)


# ================== ORIGINAL NOTEBOOK CELL 43 ==================
def rmse_loss(pred, y):
    return torch.sqrt(F.mse_loss(pred.float(), y) + EPS)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    se = torch.zeros(5, device=device, dtype=torch.float64); n = 0; preds = []
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=AMP_ON):
            p = model(x)
        p = p.float()
        se += ((p - y) ** 2).sum(0).double(); n += len(y); preds.append(p.cpu())
    mse_k = se / n
    return float(mse_k.mean().sqrt()), [float(v) for v in mse_k.sqrt().cpu()], torch.cat(preds).numpy()

def train_epoch(model, loader, opt, scaler):
    model.train(); tot, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=AMP_ON):
            pred = model(x)
        loss = rmse_loss(pred, y)
        if not torch.isfinite(loss):
            return float("nan")
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot += loss.item() * len(y); n += len(y)
    return tot / n

def _cleanup(*objs):
    for o in objs: del o
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()

def train_unit(arch, opt_name, bs, fold, max_epochs=None, patience=None):
    """Satu unit 5-Fold CV: latih pada 4 fold, validasi pada 1 fold, early stopping. Mengembalikan dict hasil."""
    max_epochs = max_epochs or MAX_EPOCHS; patience = patience or PATIENCE
    set_seed(SEED + fold)                                   # head awal sama untuk semua optimizer pada fold ini
    model = build_model(arch); opt = make_optimizer(model, arch, opt_name); scaler = make_scaler()
    train_dl, val_dl = cv_loaders(fold, bs)
    t0 = time.time(); hist_tr, hist_va = [], []
    best_v, best_k, best_ep, bad, diverged = float("inf"), None, 0, 0, False
    for ep in range(1, max_epochs + 1):
        tl = train_epoch(model, train_dl, opt, scaler)
        if not np.isfinite(tl):
            diverged = True; break
        v, vk, _ = evaluate(model, val_dl)
        hist_tr.append(tl); hist_va.append(v)
        if v < best_v - 1e-6:
            best_v, best_k, best_ep, bad = v, vk, ep, 0
        else:
            bad += 1
            if bad >= patience: break
    if best_ep == 0:                                        # tidak pernah membaik / diverge pada epoch pertama
        best_v, best_k, best_ep = (hist_va[-1] if hist_va else float("nan")), ([float("nan")] * 5), len(hist_va)
    out = dict(best_epoch=best_ep, best_val_rmse=best_v, best_val_rmse_trait=best_k, epochs_run=len(hist_va),
               diverged=diverged, history=dict(train_loss=hist_tr, val_rmse=hist_va), seconds=time.time() - t0,
               n_train=len(train_dl.dataset), n_val=len(val_dl.dataset))
    _cleanup(model, opt, scaler, train_dl, val_dl)
    return out

def train_final(arch, opt_name, bs, epochs, save_path=None):
    """Pelatihan akhir pada SELURUH data latih selama `epochs` (tanpa validasi/early stopping), uji SATU kali di akhir."""
    set_seed(SEED + 99)
    model = build_model(arch); opt = make_optimizer(model, arch, opt_name); scaler = make_scaler()
    train_dl, test_dl = final_loaders(bs)
    t0 = time.time(); hist = []
    for ep in range(1, epochs + 1):
        hist.append(train_epoch(model, train_dl, opt, scaler))
    v, vk, preds = evaluate(model, test_dl)                 # <- satu-satunya evaluasi pada data uji
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    out = dict(test_rmse=v, test_rmse_trait=vk, epochs=epochs, train_loss=hist, seconds=time.time() - t0,
               n_train=len(train_dl.dataset), n_test=len(test_dl.dataset))
    _cleanup(model, opt, scaler, train_dl, test_dl)
    return out, preds


# ================== ORIGINAL NOTEBOOK CELL 45 ==================
def _sig(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]

def base_sig(arch, opt, bs):
    return dict(arch=arch, opt=opt, bs=bs, lr=LR[arch][opt], wd=WD[opt], betas=BETAS, momentum=MOMENTUM, eps=EPS,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED, amp=AMP_ON if "AMP_ON" in globals() else USE_AMP,
                vit_ckpt=VIT_CKPT, vit_renorm=VIT_RENORM_TO_HALF,
                manifest=MANIFEST_MD5)

def _atomic_write(path, obj):
    tmp = Path(str(path) + ".tmp"); tmp.write_text(json.dumps(obj)); os.replace(tmp, path)

def cv_path(arch, opt, bs, fold): return CV_DIR / f"{arch}__{opt}__bs{bs}__fold{fold}.json"

def load_valid(path, sig):
    if not path.exists(): return None
    try: d = json.loads(path.read_text())
    except Exception: return None
    return d if d.get("signature") == sig else None

def cv_unit_done(arch, opt, bs, fold):
    return load_valid(cv_path(arch, opt, bs, fold), _sig(base_sig(arch, opt, bs))) is not None

def run_cv_unit(arch, opt, bs, fold):
    sig = _sig(base_sig(arch, opt, bs)); p = cv_path(arch, opt, bs, fold)
    res = train_unit(arch, opt, bs, fold)
    res.update(arch=arch, opt=opt, bs=bs, fold=fold, signature=sig)
    _atomic_write(p, res)
    return res

def load_cv_results():
    rows = []
    for arch in ARCHS:
        for opt in OPTIMIZERS:
            for bs in sorted({16, 32}):
                for fold in FOLDS:
                    d = load_valid(cv_path(arch, opt, bs, fold), _sig(base_sig(arch, opt, bs)))
                    if d is not None:
                        rows.append(dict(arch=arch, opt=opt, bs=bs, fold=fold, rmse=d["best_val_rmse"], best_epoch=d["best_epoch"],
                                         epochs_run=d["epochs_run"], diverged=d["diverged"], seconds=d["seconds"],
                                         **{f"rmse_{t}": v for t, v in zip(TRAITS, d["best_val_rmse_trait"])}))
    return pd.DataFrame(rows)

def summarize_cv(df):
    """Rata-rata ± SD (ddof=1, Persamaan 5-6) per konfigurasi; hanya konfigurasi dengan semua fold yang dianggap lengkap."""
    if df.empty: return df
    g = df.groupby(["arch", "opt", "bs"])
    s = g.agg(n_folds=("fold", "nunique"), mean_rmse=("rmse", "mean"), sd_rmse=("rmse", lambda v: v.std(ddof=1)),
              mean_best_epoch=("best_epoch", "mean"), diverged=("diverged", "sum"), minutes=("seconds", lambda v: v.sum() / 60),
              **{f"mean_{t}": (f"rmse_{t}", "mean") for t in TRAITS}).reset_index()
    s["complete"] = s.n_folds == len(FOLDS)
    return s

def fmt_ms(m, s): return f"{m:.4f} ± {s:.4f}" if np.isfinite(s) else f"{m:.4f}"


# ================== ORIGINAL NOTEBOOK CELL 47 ==================
T0 = time.time()

def out_of_time(est_unit_seconds):
    return TIME_BUDGET_HOURS is not None and (time.time() - T0 + est_unit_seconds) > TIME_BUDGET_HOURS * 3600

def run_cv(configs):
    est = 0.0
    todo = [(a, o, b, f) for a, o, b in configs for f in FOLDS if not cv_unit_done(a, o, b, f)]
    print(f"Unit tersisa: {len(todo)} dari {len(configs) * len(FOLDS)}")
    for n, (arch, opt, bs, fold) in enumerate(todo, 1):
        if out_of_time(est):
            print(f"⏸️ Batas waktu tercapai; {len(todo) - n + 1} unit tersisa. Lanjutkan di sesi berikutnya (unit selesai otomatis dilewati).")
            return False
        r = run_cv_unit(arch, opt, bs, fold); est = max(est, r["seconds"])
        flag = "  ⚠️ DIVERGE" if r["diverged"] else ""
        print(f"[{n:>2}/{len(todo)}] {arch:7s} {opt:5s} bs{bs:<2d} fold{fold} | val RMSE {r['best_val_rmse']:.4f} "
              f"(epoch {r['best_epoch']}/{r['epochs_run']}) | {r['seconds']/60:.1f} mnt{flag}")
    return True

configs1 = [(a, o, b) for o in OPTIMIZERS for a in ARCHS for b in BATCH_SIZES_STAGE1]
stage1_ok = run_cv(configs1)

stage2_ok = True
if RUN_BS32 and stage1_ok and 32 not in BATCH_SIZES_STAGE1:
    s1 = summarize_cv(load_cv_results()); s1 = s1[(s1.bs == 16) & s1.complete]
    configs2 = [(a, s1[s1.arch == a].sort_values("mean_rmse").iloc[0].opt, 32) for a in ARCHS]
    print("\nTahap 2 (bs=32) untuk optimizer terbaik per arsitektur:", configs2)
    stage2_ok = run_cv(configs2)

CV_COMPLETE = bool(stage1_ok and stage2_ok)
print("\nCV_COMPLETE =", CV_COMPLETE)


# ================== ORIGINAL NOTEBOOK CELL 49 ==================
cv_df = load_cv_results()
cv_sum = summarize_cv(cv_df)
NAMES = {"vgg19": "VGG19", "vit_b16": "ViT-B16"}
OPT_NAMES = {"adam": "Adam (baseline)", "adamw": "AdamW", "sgd": "SGD Nesterov"}

def fold_table(arch, opt, bs):
    return cv_df[(cv_df.arch == arch) & (cv_df.opt == opt) & (cv_df.bs == bs)].set_index("fold").sort_index()

SEL = pd.DataFrame()
if cv_sum.empty:
    print("Belum ada hasil CV.")
else:
    comp = cv_sum[cv_sum.complete]
    # satu baris per (arsitektur, optimizer): batch size terpilih = RMSE validasi rata-rata terendah di antara kandidat yang selesai
    SEL = comp.sort_values("mean_rmse").groupby(["arch", "opt"], as_index=False).head(1)
    SEL = (SEL.assign(_a=SEL.arch.map({a: i for i, a in enumerate(ARCHS)}), _o=SEL.opt.map({o: i for i, o in enumerate(OPTIMIZERS)}))
              .sort_values(["_a", "_o"]).drop(columns=["_a", "_o"]).reset_index(drop=True))
    t9v = pd.DataFrame({"Arsitektur": SEL.arch.map(NAMES), "Optimizer": SEL.opt.map(OPT_NAMES),
                        "RMSE validasi (rata-rata ± SD)": [fmt_ms(m, s) for m, s in zip(SEL.mean_rmse, SEL.sd_rmse)],
                        "RMSE uji": "–"})
    print("Tabel 8 (bagian validasi; kolom RMSE uji diisi di B11). RMSE skala 0-1, SD antar-fold (ddof = 1):"); display(t9v)
    t9v.to_csv(TAB_DIR / "tabel8_validasi.csv", index=False)
    info = pd.DataFrame({"Arsitektur": SEL.arch.map(NAMES), "Optimizer": SEL.opt.map(OPT_NAMES), "BS terpilih": SEL.bs,
                         "RMSE validasi skala 0-40": [fmt_ms(m * LABEL_MAX, s * LABEL_MAX) for m, s in zip(SEL.mean_rmse, SEL.sd_rmse)],
                         "Epoch terbaik (rata2)": SEL.mean_best_epoch.round(1), "Diverge": SEL.diverged.astype(int)})
    print("\nInformasi pendukung (di luar Tabel 8): batch size terpilih, skala kuesioner, epoch terbaik:"); display(info)
    print(f"Referensi tambahan, baseline prediktor rata-rata: {fmt_ms(BASE['val_mean'], BASE['val_sd'])} (0-1) = {BASE['val_mean']*LABEL_MAX:.2f} (0-40)\n")

    detail = cv_sum.assign(Arsitektur=cv_sum.arch.map(NAMES), Optimizer=cv_sum.opt.map(OPT_NAMES),
                           **{"RMSE validasi": [fmt_ms(m, s) for m, s in zip(cv_sum.mean_rmse, cv_sum.sd_rmse)]})
    detail = detail.sort_values(["arch", "opt", "bs"])[["Arsitektur", "Optimizer", "bs", "n_folds", "RMSE validasi", "mean_best_epoch", "diverged"]]
    print("\nDetail seluruh konfigurasi yang dijalankan (termasuk kandidat batch size):"); display(detail.reset_index(drop=True))
    detail.to_csv(TAB_DIR / "cv_detail_konfigurasi.csv", index=False)


# ================== ORIGINAL NOTEBOOK CELL 50 ==================
if not cv_sum.empty:
    # ---- Tabel pendamping (format sama dengan Tabel 8): RMSE validasi per dimensi OCEAN, rata-rata ± SD antar-fold ----
    if len(SEL):
        prow = []
        for _, r in SEL.iterrows():
            d = fold_table(r.arch, r.opt, int(r.bs))
            row = {"Arsitektur": NAMES[r.arch], "Optimizer": OPT_NAMES[r.opt]}
            for t in TRAITS:
                row[TRAIT_NAMES[t]] = fmt_ms(d[f"rmse_{t}"].mean(), d[f"rmse_{t}"].std(ddof=1))
            prow.append(row)
        print("Tabel pendamping (validasi): RMSE per dimensi OCEAN, rata-rata ± SD antar-fold, skala 0-1:"); display(pd.DataFrame(prow))
        print("Referensi baseline rata-rata per dimensi:", {TRAIT_NAMES[t]: round(v, 4) for t, v in zip(TRAITS, BASE["val_trait"])})
        pd.DataFrame(prow).to_csv(TAB_DIR / "tabel8_pendamping_validasi_per_dimensi.csv", index=False)

        # ---- ΔRMSE (Persamaan 7) per fold untuk optimizer yang sama: RMSE(VGG19) - RMSE(ViT-B16); positif = ViT-B16 lebih kecil ----
        drow = []
        for opt in OPTIMIZERS:
            a, b = SEL[(SEL.arch == "vgg19") & (SEL.opt == opt)], SEL[(SEL.arch == "vit_b16") & (SEL.opt == opt)]
            if a.empty or b.empty: continue
            da, db = fold_table("vgg19", opt, int(a.bs.iloc[0])), fold_table("vit_b16", opt, int(b.bs.iloc[0]))
            common = da.index.intersection(db.index)
            row = {"Optimizer": OPT_NAMES[opt], "BS (VGG19/ViT)": f"{int(a.bs.iloc[0])}/{int(b.bs.iloc[0])}"}
            for col, nm in [("rmse", "Keseluruhan")] + [(f"rmse_{t}", TRAIT_NAMES[t]) for t in TRAITS]:
                dl = da.loc[common, col] - db.loc[common, col]
                row[nm] = f"{dl.mean():+.4f} ± {dl.std(ddof=1):.4f}" if len(dl) > 1 else f"{dl.mean():+.4f}"
            row["ViT lebih baik (fold)"] = f"{int(((da.loc[common, 'rmse'] - db.loc[common, 'rmse']) > 0).sum())}/{len(common)}"
            drow.append(row)
        print("ΔRMSE validasi = RMSE(VGG19) − RMSE(ViT-B16), rata-rata ± SD antar-fold (positif = ViT-B16 lebih kecil):"); display(pd.DataFrame(drow))
        pd.DataFrame(drow).to_csv(TAB_DIR / "delta_rmse_validasi.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(16, 4.2))
    s = cv_sum.sort_values(["arch", "opt", "bs"]).reset_index(drop=True)
    lab = [f"{NAMES[a]}\n{o}\nbs{b}" for a, o, b in zip(s.arch, s.opt, s.bs)]
    ax[0].bar(range(len(s)), s.mean_rmse, yerr=s.sd_rmse.fillna(0), capsize=4,
              color=["#4C72B0" if a == "vgg19" else "#DD8452" for a in s.arch])
    ax[0].axhline(BASE["val_mean"], color="r", ls="--", label="baseline rata-rata")
    ax[0].set_xticks(range(len(s))); ax[0].set_xticklabels(lab, fontsize=8); ax[0].set_ylabel("RMSE validasi (0-1)")
    ax[0].set_title("RMSE validasi rata-rata ± SD antar-fold"); ax[0].legend()
    ax[0].set_ylim(0, max(s.mean_rmse.max(), BASE["val_mean"]) * 1.25)

    best_now = s[s.complete].sort_values("mean_rmse").groupby("arch").head(1) if s.complete.any() else s.head(0)
    for _, r in best_now.iterrows():
        for f in FOLDS:
            d = load_valid(cv_path(r.arch, r.opt, int(r.bs), f), _sig(base_sig(r.arch, r.opt, int(r.bs))))
            if d: ax[1].plot(range(1, len(d["history"]["val_rmse"]) + 1), d["history"]["val_rmse"],
                             color="#4C72B0" if r.arch == "vgg19" else "#DD8452", alpha=.6,
                             label=f"{NAMES[r.arch]} · {r.opt}" if f == FOLDS[0] else None)
    ax[1].axhline(BASE["val_mean"], color="r", ls="--", lw=1)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("RMSE validasi (0-1)"); ax[1].set_title("Kurva validasi (konfigurasi terbaik; satu garis per fold)")
    if len(best_now): ax[1].legend()
    plt.tight_layout(); (RUN_DIR / "figures").mkdir(exist_ok=True); plt.savefig(RUN_DIR / "figures" / "cv_overview.png", dpi=130); plt.show()


# ================== ORIGINAL NOTEBOOK CELL 52 ==================
BEST = {}
if not CV_COMPLETE:
    print("⏸️ CV belum lengkap -> pemilihan konfigurasi & uji akhir ditunda. Jalankan ulang notebook (unit selesai akan dilewati).")
else:
    comp = cv_sum[cv_sum.complete]
    for arch in ARCHS:
        r = comp[comp.arch == arch].sort_values("mean_rmse").iloc[0]
        BEST[arch] = dict(opt=r.opt, bs=int(r.bs), mean_rmse=float(r.mean_rmse), sd_rmse=float(r.sd_rmse),
                          final_epochs=max(1, int(round(r.mean_best_epoch))))
        print(f"{NAMES[arch]:8s} -> optimizer {OPT_NAMES[r.opt]:16s} bs={int(r.bs)} | RMSE val {fmt_ms(r.mean_rmse, r.sd_rmse)} "
              f"| epoch pelatihan akhir = {BEST[arch]['final_epochs']}")
    (RUN_DIR / "best_configs.json").write_text(json.dumps(BEST, indent=2))
    for arch in ARCHS:
        if BEST[arch]["mean_rmse"] >= BASE["val_mean"]:
            print(f"⚠️ {NAMES[arch]}: RMSE validasi tidak lebih baik dari baseline rata-rata. Periksa kurva/learning rate sebelum menafsirkan.")


# ================== ORIGINAL NOTEBOOK CELL 54 ==================
FINAL = {}
if CV_COMPLETE:
    for arch in ARCHS:
        opt, bs, ep = BEST[arch]["opt"], BEST[arch]["bs"], BEST[arch]["final_epochs"]
        key = f"{arch}__{opt}__bs{bs}"; jp, pp = FINAL_DIR / f"{key}.json", FINAL_DIR / f"{key}__test_preds.npy"
        sig = _sig(dict(base_sig(arch, opt, bs), epochs=ep, kind="final"))
        d = load_valid(jp, sig)
        if d is not None and pp.exists():
            preds = np.load(pp); print(f"[{NAMES[arch]}] hasil akhir sudah ada -> dilewati")
        else:
            print(f"[{NAMES[arch]}] pelatihan akhir: {OPT_NAMES[opt]}, bs={bs}, {ep} epoch ...")
            d, preds = train_final(arch, opt, bs, ep, save_path=(FINAL_DIR / f"{key}.pt") if SAVE_FINAL_WEIGHTS else None)
            d.update(arch=arch, opt=opt, bs=bs, signature=sig); np.save(pp, preds); _atomic_write(jp, d)
        FINAL[arch] = dict(d, preds=preds)
        print(f"   RMSE uji (0-1) = {d['test_rmse']:.4f}  (= {d['test_rmse']*LABEL_MAX:.2f} pada skala 0-40) | {d['seconds']/60:.1f} mnt")

    # baseline prediktor rata-rata untuk data uji (rata-rata dari SELURUH data latih)
    BASE["test"], bt = rmse_from_err(Y[~is_train] - Y[is_train].mean(0))
    BASE["test_trait"] = dict(zip(TRAITS, bt))
    print(f"Baseline rata-rata, RMSE uji (0-1) = {BASE['test']:.4f}")


# ================== ORIGINAL NOTEBOOK CELL 56 ==================
if CV_COMPLETE:
    def _is_best(r): return BEST[r.arch]["opt"] == r.opt and BEST[r.arch]["bs"] == int(r.bs)
    DIMS = [("Keseluruhan", "rmse", None)] + [(TRAIT_NAMES[t], f"rmse_{t}", i) for i, t in enumerate(TRAITS)]
    def _test(arch, idx): return FINAL[arch]["test_rmse"] if idx is None else FINAL[arch]["test_rmse_trait"][idx]

    # ================= Tabel 8: Hasil Evaluasi (format persis seperti rancangan proposal) =================
    tab8 = pd.DataFrame([{"Arsitektur": NAMES[r.arch], "Optimizer": OPT_NAMES[r.opt],
                          "RMSE validasi (rata-rata ± SD)": fmt_ms(r.mean_rmse, r.sd_rmse),
                          "RMSE uji": f"{FINAL[r.arch]['test_rmse']:.4f}" if _is_best(r) else "–"} for _, r in SEL.iterrows()])
    print("Tabel 8. Hasil Evaluasi (RMSE skala 0-1; RMSE uji hanya untuk konfigurasi terbaik tiap arsitektur):"); display(tab8)
    tab8.to_csv(TAB_DIR / "tabel8_hasil_evaluasi.csv", index=False)
    print("Catatan: kalikan 40 untuk skala kuesioner asli (0-40).")

    # ---- informasi pendukung (di luar Tabel 8) ----
    sup = pd.DataFrame([{"Arsitektur": NAMES[a], "Konfigurasi terbaik": OPT_NAMES[BEST[a]["opt"]], "BS": BEST[a]["bs"],
                         "Epoch pelatihan akhir": BEST[a]["final_epochs"], "RMSE uji (skala 0-40)": round(FINAL[a]["test_rmse"] * LABEL_MAX, 2)} for a in ARCHS])
    sup.loc[len(sup)] = ["Baseline rata-rata (referensi)", "-", "-", "-", round(BASE["test"] * LABEL_MAX, 2)]
    print("\nInformasi pendukung (di luar Tabel 8):"); display(sup); sup.to_csv(TAB_DIR / "informasi_pendukung.csv", index=False)

    # ================= Tabel pendamping 1: nilai per dimensi OCEAN (format sama dengan Tabel 8) =================
    data = []
    for _, r in SEL.iterrows():
        d = fold_table(r.arch, r.opt, int(r.bs)); rec = {}
        for nm, col, idx in DIMS:
            rec[(nm, "RMSE validasi (rata-rata ± SD)")] = fmt_ms(d[col].mean(), d[col].std(ddof=1))
            rec[(nm, "RMSE uji")] = f"{_test(r.arch, idx):.4f}" if _is_best(r) else "–"
        data.append(rec)
    comp_tab = pd.DataFrame(data, index=pd.MultiIndex.from_arrays([SEL.arch.map(NAMES), SEL.opt.map(OPT_NAMES)], names=["Arsitektur", "Optimizer"]))
    comp_tab.columns = pd.MultiIndex.from_tuples(comp_tab.columns)
    print("\nTabel pendamping A. RMSE per dimensi OCEAN (format sama dengan Tabel 8):"); display(comp_tab)
    comp_tab.to_csv(TAB_DIR / "tabel8_pendamping_per_dimensi.csv")

    # ================= Tabel pendamping 2: ΔRMSE = RMSE(VGG19) − RMSE(ViT-B16) (format sama) =================
    def _delta_val(opt_v, bs_v, opt_t, bs_t, col):
        da, db = fold_table("vgg19", opt_v, bs_v), fold_table("vit_b16", opt_t, bs_t); c = da.index.intersection(db.index)
        dl = da.loc[c, col] - db.loc[c, col]
        return f"{dl.mean():+.4f} ± {dl.std(ddof=1):.4f}" if len(dl) > 1 else f"{dl.mean():+.4f}"
    drows, didx = [], []
    for opt in OPTIMIZERS:                                   # pasangan optimizer yang sama, batch size terpilih masing-masing
        a, b = SEL[(SEL.arch == "vgg19") & (SEL.opt == opt)], SEL[(SEL.arch == "vit_b16") & (SEL.opt == opt)]
        if a.empty or b.empty: continue
        rec = {}
        for nm, col, idx in DIMS:
            rec[(nm, "ΔRMSE validasi (rata-rata ± SD)")] = _delta_val(opt, int(a.bs.iloc[0]), opt, int(b.bs.iloc[0]), col); rec[(nm, "ΔRMSE uji")] = "–"
        drows.append(rec); didx.append(OPT_NAMES[opt])
    rec = {}                                                 # konfigurasi terbaik tiap arsitektur: satu-satunya baris dengan nilai uji
    bv, bt = BEST["vgg19"], BEST["vit_b16"]
    for nm, col, idx in DIMS:
        rec[(nm, "ΔRMSE validasi (rata-rata ± SD)")] = _delta_val(bv["opt"], bv["bs"], bt["opt"], bt["bs"], col)
        rec[(nm, "ΔRMSE uji")] = f"{_test('vgg19', idx) - _test('vit_b16', idx):+.4f}"
    drows.append(rec); didx.append("Konfigurasi terbaik tiap arsitektur")
    delta_tab = pd.DataFrame(drows, index=pd.Index(didx, name="Pasangan (VGG19 − ViT-B16)")); delta_tab.columns = pd.MultiIndex.from_tuples(delta_tab.columns)
    print("\nTabel pendamping B. ΔRMSE = RMSE(VGG19) − RMSE(ViT-B16) (positif = ViT-B16 lebih kecil; uji hanya untuk konfigurasi terbaik):"); display(delta_tab)
    delta_tab.to_csv(TAB_DIR / "tabel8_pendamping_delta_rmse.csv")

    # ---- uji per dimensi dalam skala 0-1 dan 0-40, plus baseline referensi ----
    pt = pd.DataFrame({NAMES[a]: dict(zip(TRAITS, FINAL[a]["test_rmse_trait"]), Keseluruhan=FINAL[a]["test_rmse"]) for a in ARCHS}).T
    pt.loc["Baseline rata-rata (referensi)"] = dict(BASE["test_trait"], Keseluruhan=BASE["test"])
    pt = pt.rename(columns=TRAIT_NAMES)
    print("\nRMSE uji per dimensi, skala 0-1 (dan x40):"); display(pt.round(4)); display((pt * LABEL_MAX).round(2)); pt.to_csv(TAB_DIR / "uji_per_dimensi.csv")

    d_test = FINAL["vgg19"]["test_rmse"] - FINAL["vit_b16"]["test_rmse"]
    fv = pd.Series({f: json.loads(cv_path("vgg19", BEST["vgg19"]["opt"], BEST["vgg19"]["bs"], f).read_text())["best_val_rmse"] for f in FOLDS})
    fh = pd.Series({f: json.loads(cv_path("vit_b16", BEST["vit_b16"]["opt"], BEST["vit_b16"]["bs"], f).read_text())["best_val_rmse"] for f in FOLDS})
    d_fold = fv - fh
    print(f"\nΔRMSE uji (keseluruhan) = {d_test:+.4f} -> " + ("ViT-B16" if d_test > 0 else "VGG19") + " memiliki galat uji lebih kecil.")
    if len(d_fold) > 1:
        print(f"ΔRMSE validasi (konfigurasi terbaik) per fold: rata-rata {d_fold.mean():+.4f}, SD {d_fold.std(ddof=1):.4f}, ViT-B16 lebih baik pada {(d_fold > 0).sum()} dari {len(d_fold)} fold.")
        if abs(d_fold.mean()) < d_fold.std(ddof=1):
            print("ℹ️ Selisih rata-rata lebih kecil dari SD antar-fold: perbedaan antar-arsitektur belum dapat dipastikan (uji tunggal pada data uji kecil juga terbatas).")


# ================== ORIGINAL NOTEBOOK CELL 57 ==================
if CV_COMPLETE:
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1]})
    w = 0.38; xs = np.arange(len(TRAITS) + 1)
    for i, a in enumerate(ARCHS):
        vals = FINAL[a]["test_rmse_trait"] + [FINAL[a]["test_rmse"]]
        ax[0].bar(xs + (i - .5) * w, vals, w, label=NAMES[a], color="#4C72B0" if a == "vgg19" else "#DD8452")
    ax[0].plot(xs, list(BASE["test_trait"].values()) + [BASE["test"]], "r--o", ms=4, label="baseline rata-rata")
    ax[0].set_xticks(xs); ax[0].set_xticklabels([TRAIT_NAMES[t][:5] + "." for t in TRAITS] + ["Total"])
    ax[0].set_ylabel("RMSE uji (0-1)"); ax[0].set_title("RMSE uji per dimensi"); ax[0].legend()

    Yte = Y[~is_train]
    for a, c in zip(ARCHS, ["#4C72B0", "#DD8452"]):
        ax[1].scatter(Yte.ravel() * LABEL_MAX, FINAL[a]["preds"].ravel() * LABEL_MAX, s=7, alpha=.35, color=c, label=NAMES[a])
    ax[1].plot([0, LABEL_MAX], [0, LABEL_MAX], "k--", lw=1); ax[1].set_xlabel("skor aktual (0-40)"); ax[1].set_ylabel("prediksi (0-40)")
    ax[1].set_title("Prediksi vs aktual (seluruh dimensi, data uji)"); ax[1].legend()
    plt.tight_layout(); plt.savefig(RUN_DIR / "figures" / "test_overview.png", dpi=130); plt.show()

    manifest.loc[~is_train, ["file", "writer_id", "lang"]].assign(
        **{f"true_{t}": Y[~is_train][:, i] for i, t in enumerate(TRAITS)},
        **{f"{a}_pred_{t}": FINAL[a]["preds"][:, i] for a in ARCHS for i, t in enumerate(TRAITS)}
    ).to_csv(TAB_DIR / "test_predictions.csv", index=False)

meta_run = dict(SMOKE_TEST=SMOKE_TEST, MAX_EPOCHS=MAX_EPOCHS, PATIENCE=PATIENCE, SEED=SEED, FOLDS=FOLDS, USE_AMP=USE_AMP,
                VIT_CKPT=VIT_CKPT, VIT_RENORM_TO_HALF=VIT_RENORM_TO_HALF, LR=LR, WD=WD, MANIFEST_MD5=MANIFEST_MD5,
                CV_COMPLETE=CV_COMPLETE, python=platform.python_version())
(RUN_DIR / "run_meta.json").write_text(json.dumps(meta_run, indent=2))
print("Berkas hasil:", sorted(p.name for p in RUN_DIR.iterdir()))
