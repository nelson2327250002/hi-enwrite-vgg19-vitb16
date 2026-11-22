"""HiEnWrite data audit, cleaning, split, and image preprocessing.

Extracted in notebook cell order from notebooks/00_full_kaggle_experiment.ipynb.
This script intentionally preserves the original notebook workflow and shared variables.
Run in a Kaggle notebook/IPython session with the required dataset and artifacts attached.
"""

from __future__ import annotations
from IPython.display import display

# ================== ORIGINAL NOTEBOOK CELL 2 ==================
import os, glob, json, hashlib, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image, ImageOps
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import KFold

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100
LANCZOS = Image.Resampling.LANCZOS

# ====================== SESUAI PROPOSAL V12 ======================
SEED       = 42        # seed tetap: split & fold
IMG_SIZE   = 224       # Bagian 3: resize seragam 224 x 224
LABEL_MAX  = 40.0      # Bagian 3: skor OCEAN dibagi 40 (batas skala kuesioner, bukan statistik data)
TEST_RATIO = 0.25      # Bagian 4.4: 75 : 25 pada tingkat ID penulis
N_FOLDS    = 5         # Bagian 4.5
BLANK_STD_MIN = 5.0    # Bagian 2: citra kosong bila simpangan baku piksel < 5,0
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # Bagian 3: statistik ImageNet tetap
IMAGENET_STD  = [0.229, 0.224, 0.225]
RESIZE_MODE   = "squash"                # Bagian 3: resize langsung ke 224x224 ('pad' hanya untuk analisis sensitivitas)

# ====================== KEPUTUSAN & PARAMETER TEKNIS ======================
OPENNESS_FIX = "english_row"  # ⚠️ 'english_row' | 'none'  (lihat section A3)
NEAR_DUP_MAX_HAMMING = 8      # ambang dHash 16x16 (256 bit) untuk duplikat "hampir identik"
DHASH_N = 16

# ====================== SELEKSI MANUAL (Bagian 2) ======================
APPLY_MANUAL = True
# Koreksi orientasi: sudut PIL (positif = putar berlawanan arah jarum jam), diterapkan SETELAH koreksi EXIF.
# Hanya mengembalikan citra ke posisi tegak; tidak mengubah kemiringan huruf/garis dasar/spasi.
# Empat entri di bawah terverifikasi visual pada berkas ini (teks vertikal). Proposal menyebut 10 citra:
# sisanya (mis. terbalik 180°) tidak dapat dideteksi otomatis -> tambahkan dari daftar manual Anda.
MANUAL_ROTATE = {"38e.jpeg": 90, "38h.jpeg": 90, "195e.jpeg": -90, "195h.jpeg": -90}
# Penghapusan citra non-tulisan (mis. memuat wajah) atau salah-bahasa: {nama_berkas: alasan}. Berkas ini diperlakukan
# "tidak tersedia" (masuk aturan berkas_hilang seperti pada Gambar 2 proposal) sehingga penulisnya dikeluarkan beserta pasangannya.
MANUAL_REMOVE = {}            # contoh: {"123e.jpeg": "foto wajah"}

# ====================== ANGKA REFERENSI PROPOSAL V12 (Gambar 2 & Tabel 4) ======================
# (penulis dibuang, citra dibuang, citra bermasalah, citra sehat ikut terbuang)
PROPOSAL_REF = {"berkas_hilang": (31, 62, 44, 18), "tidak_terbaca": (1, 2, 1, 1), "citra_kosong": (1, 2, 1, 1),
                "duplikat_EN_HI_penulis_sama": (2, 4, 4, 0), "duplikat_antar_penulis": (46, 92, 78, 14)}
PROPOSAL_FINAL = dict(penulis=264, citra=528, train=198, test=66, fold_penulis=[40, 40, 40, 39, 39])

TRAITS = ["O", "C", "E", "A", "N"]
TRAIT_NAMES = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion", "A": "Agreeableness", "N": "Neuroticism"}

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
seed_everything()


# ================== ORIGINAL NOTEBOOK CELL 3 ==================
DATA_ROOT = None   # contoh manual: "/kaggle/input/nama-dataset-anda/HiEnWrite-Dataset-main"

def find_dataset_root():
    for base in ["/kaggle/input", ".", "/mnt/user-data", "/home/claude/data"]:
        hits = glob.glob(os.path.join(base, "**", "merged_data.csv"), recursive=True)
        if hits:
            return Path(hits[0]).parent.parent
    raise FileNotFoundError("merged_data.csv tidak ditemukan. Atur DATA_ROOT secara manual.")

ROOT     = Path(DATA_ROOT) if DATA_ROOT else find_dataset_root()
META_CSV = ROOT / "data" / "merged_data.csv"
IMG_DIR  = ROOT / "images"
OUT_DIR  = Path("/kaggle/working" if os.path.isdir("/kaggle/working") else ".") / "hienwrite_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Dataset root :", ROOT)
print("Berkas gambar di folder:", len(list(IMG_DIR.glob("*"))), "| folder luaran:", OUT_DIR)


# ================== ORIGINAL NOTEBOOK CELL 5 ==================
raw = pd.read_csv(META_CSV).drop(
    columns=["Unnamed: 0"],
    errors="ignore"
)
raw = raw.rename(columns={"image": "file", "Openness to Experience": "O", "Conscientiousness": "C",
                          "Extroversion": "E", "Agreeableness": "A", "Neuroticism": "N"})
parsed = raw["file"].str.extract(r"^(\d+)([eh])\.jpe?g$")
assert parsed.notna().all().all(), "Ada nama berkas yang tidak mengikuti pola <id><e|h>.jpeg"
raw["writer_id"] = parsed[0].astype(int)
raw["lang"] = parsed[1].map({"e": "EN", "h": "HI"})
raw = raw[["file", "writer_id", "lang"] + TRAITS + ["Age", "Gender"]]

print("Bentuk metadata:", raw.shape, "| nilai kosong:", int(raw.isna().sum().sum()),
      "| baris duplikat:", int(raw.duplicated().sum()))
print("Penulis unik:", raw.writer_id.nunique(), "| citra per penulis:", raw.groupby("writer_id").size().unique().tolist(),
      "| bahasa:", raw.lang.value_counts().to_dict())
assert raw[TRAITS].isna().sum().sum() == 0, "Ada nilai kosong pada skor OCEAN"
print("Rentang skor (skala kuesioner 0-40):"); display(raw[TRAITS].agg(["min", "max", "mean", "std"]).round(2))

meta = raw.drop(columns=["Age", "Gender"]).copy()        # demografi dibuang (Bagian 2)
print("Kolom setelah demografi dihapus:", list(meta.columns))


# ================== ORIGINAL NOTEBOOK CELL 7 ==================
pair = meta.pivot(index="writer_id", columns="lang", values=TRAITS)
print("Jumlah penulis dengan skor EN != HI:", {TRAIT_NAMES[t]: int((pair[(t, "EN")] != pair[(t, "HI")]).sum()) for t in TRAITS})
display(meta.groupby("lang")["O"].agg(["count", "nunique", "min", "max", "mean", "std"]).round(2))
print("Nilai Openness pada baris Hindi:", meta.loc[meta.lang == "HI", "O"].value_counts().to_dict())

fig, axes = plt.subplots(1, 5, figsize=(18, 3.2))
for ax, t in zip(axes, TRAITS):
    sns.histplot(data=meta, x=t, hue="lang", bins=range(0, 42, 2), alpha=.55, ax=ax)
    ax.set_title(TRAIT_NAMES[t] + (" ⚠️ HI konstan" if t == "O" else ""))
plt.tight_layout(); plt.show()


# ================== ORIGINAL NOTEBOOK CELL 9 ==================
meta["O_raw"] = meta["O"]
if OPENNESS_FIX == "english_row":
    en_open = meta.loc[meta.lang == "EN"].set_index("writer_id")["O"]
    meta["O"] = meta["writer_id"].map(en_open)
    print(f"Openness diseragamkan dari baris EN. Baris yang berubah: {int((meta.O != meta.O_raw).sum())}")
else:
    print("OPENNESS_FIX='none' -> skor mentah dipertahankan (Openness baris HI tetap konstan 24).")
n_uniq = meta.groupby("writer_id")[TRAITS].nunique().max()
print("Maks nilai unik per penulis per dimensi:", n_uniq.to_dict())
if OPENNESS_FIX == "english_row":
    assert (n_uniq == 1).all()


# ================== ORIGINAL NOTEBOOK CELL 11 ==================
meta["manual_removed"] = meta["file"].isin(MANUAL_REMOVE) if APPLY_MANUAL else False
meta["file_exists"] = meta["file"].map(lambda f: (IMG_DIR / f).exists()) & ~meta["manual_removed"]
extra = sorted(set(os.listdir(IMG_DIR)) - set(meta.file))
print(f"Record metadata: {len(meta)} | berkas tersedia: {int(meta.file_exists.sum())} | TIDAK tersedia: {int((~meta.file_exists).sum())}"
      f" (di antaranya dihapus manual: {int(meta.manual_removed.sum())}) | berkas tanpa record: {len(extra)}")
miss = meta.loc[~meta.file_exists].groupby("writer_id").lang.apply(lambda s: "+".join(sorted(s)))
print("Penulis dengan berkas tidak tersedia -> kedua citra hilang:", int((miss == "EN+HI").sum()), "| salah satu:", int((miss != "EN+HI").sum()))


# ================== ORIGINAL NOTEBOOK CELL 12 ==================
def load_rgb(path):
    """Muat -> RGB: koreksi EXIF, alfa dikomposit ke latar PUTIH, lalu koreksi orientasi manual (kelipatan 90°, lossless)."""
    path = Path(path)
    with Image.open(path) as im:
        im.load()                                         # dekode penuh; error bila berkas terpotong
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA"); bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
    ang = MANUAL_ROTATE.get(path.name, 0) if APPLY_MANUAL else 0
    return im.rotate(ang, expand=True) if ang else im

def _binarize(gray):
    return cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 10)

def orientation_ratio(gray):
    """<1 berarti garis teks tampak vertikal (kandidat miring 90°); ~2-4 untuk teks horizontal."""
    b = _binarize(gray).astype(np.float32); h, w = b.shape
    b = b[int(h * .05):int(h * .95), int(w * .05):int(w * .95)]
    rp = np.convolve(b.sum(1), np.ones(7) / 7, "same"); cp = np.convolve(b.sum(0), np.ones(7) / 7, "same")
    return float((rp.std() / (rp.mean() + 1e-9)) / (cp.std() / (cp.mean() + 1e-9)))

def script_features(gray):
    """Fitur sederhana (HOG + proporsi garis horizontal panjang/shirorekha) untuk membedakan aksara Latin vs Devanagari."""
    from skimage.feature import hog
    b = _binarize(gray); out = []
    for sz in (128, 256):
        out.append(hog(cv2.resize(b, (sz, sz), interpolation=cv2.INTER_AREA), orientations=9,
                       pixels_per_cell=(sz // 8, sz // 8), cells_per_block=(2, 2), feature_vector=True))
    s = b.sum() + 1
    ex = [cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))).sum() / s for L in (15, 25, 40, 60)]
    return np.r_[np.concatenate(out), ex]

def audit_one(fname):
    path = IMG_DIR / fname
    rec = dict(file=fname, readable=False, error="", fmt=None, mode=None, exif_orient=1, width=np.nan, height=np.nan,
               size_kb=round(path.stat().st_size / 1024, 1), gray_std=np.nan, orient_ratio=np.nan,
               md5=hashlib.md5(path.read_bytes()).hexdigest())
    bits = feat = None
    try:
        with Image.open(path) as im:
            rec["fmt"], rec["mode"] = im.format, im.mode
            rec["exif_orient"] = im.getexif().get(274, 1)
        rgb = load_rgb(path)
        rec["width"], rec["height"] = rgb.size
        g = rgb.convert("L")
        rec["gray_std"] = float(np.asarray(g.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32).std())
        a = np.asarray(g.resize((DHASH_N + 1, DHASH_N), LANCZOS), dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).ravel()                              # dHash 16x16 = 256 bit
        small = rgb.copy(); small.thumbnail((640, 640)); gs = np.asarray(small.convert("L"))
        rec["orient_ratio"] = orientation_ratio(gs); feat = script_features(gs)
        rec["readable"] = True
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:90]
    return rec, bits, feat

present = meta.loc[meta.file_exists, "file"].tolist()
results = [audit_one(f) for f in tqdm(present, desc="Audit citra")]
audit = pd.DataFrame([r for r, _, _ in results])
DH = {r["file"]: b for r, b, _ in results if b is not None}
SF = {r["file"]: f for r, _, f in results if f is not None}
audit["blank"] = audit.readable & (audit.gray_std < BLANK_STD_MIN)
audit.to_csv(OUT_DIR / "audit_images.csv", index=False)

print("Terbaca:", int(audit.readable.sum()), "| tidak terbaca:", int((~audit.readable).sum()))
display(audit.loc[~audit.readable, ["file", "error"]])
print(f"Citra kosong (std < {BLANK_STD_MIN}):"); display(audit.loc[audit.blank, ["file", "width", "height", "gray_std"]])
ok = audit[audit.readable]
print("Format asli:", ok.fmt.value_counts().to_dict(), "| mode:", ok["mode"].value_counts().to_dict())
print("Tag EXIF (1/0 = normal; 6/8 = diputar, dikoreksi):", ok.exif_orient.value_counts().to_dict())
display(ok[["width", "height", "size_kb", "gray_std"]].describe().round(1).T)


# ================== ORIGINAL NOTEBOOK CELL 14 ==================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- (a) kandidat teks vertikal ---
cand_rot = audit[audit.readable & (audit.orient_ratio < 0.7)].sort_values("orient_ratio")[["file", "orient_ratio", "width", "height"]]
print("Kandidat miring 90° yang BELUM dikoreksi (rasio < 0,7):", len(cand_rot)); display(cand_rot.round(2))

# --- (b) kandidat aksara tidak sesuai nama berkas ---
lf = sorted(SF); Xs = np.array([SF[f] for f in lf]); ys = np.array([f[-6] == "h" for f in lf]).astype(int)
clf = make_pipeline(StandardScaler(), PCA(60, random_state=0), LogisticRegression(C=0.01, max_iter=5000))
p_hi = cross_val_predict(clf, Xs, ys, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
lang = pd.DataFrame({"file": lf, "label_berkas": np.where(ys == 1, "HI", "EN"), "p_hindi": p_hi})
lang["curiga"] = ((lang.label_berkas == "EN") & (lang.p_hindi > .95)) | ((lang.label_berkas == "HI") & (lang.p_hindi < .05))
print(f"Akurasi klasifier aksara (validasi silang): {((lang.p_hindi > .5) == (ys == 1)).mean():.3f} -> hanya sebagai penyaring awal")
cand_lang = lang[lang.curiga].sort_values("p_hindi"); print("Kandidat aksara tidak sesuai (berkeyakinan tinggi):", len(cand_lang))
display(cand_lang[["file", "label_berkas", "p_hindi"]].round(2))

show = cand_lang.file.tolist()[:12]
if show:
    cols = 4; rows = (len(show) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * 330, rows * 200), "white")
    for i, f in enumerate(show):
        im = load_rgb(IMG_DIR / f); im.thumbnail((320, 180)); canvas.paste(im, ((i % cols) * 330, (i // cols) * 200 + 14))
    fig, ax = plt.subplots(figsize=(15, 3.6 * rows)); ax.imshow(canvas); ax.axis("off")
    for i, f in enumerate(show):
        ax.text((i % cols) * 330 + 4, (i // cols) * 200 + 11, f, color="red", fontsize=9)
    plt.tight_layout(); plt.show()


# ================== ORIGINAL NOTEBOOK CELL 17 ==================
valid = audit.loc[audit.readable & ~audit.blank, ["file", "md5"]].merge(meta[["file", "writer_id", "lang"]], on="file").reset_index(drop=True)
B = np.stack([DH[f] for f in valid.file]).astype(np.float32)
ham = B.shape[1] - (B @ B.T + (1 - B) @ (1 - B).T)
md5_code = pd.factorize(valid.md5)[0]
adj = (ham <= NEAR_DUP_MAX_HAMMING) | (md5_code[:, None] == md5_code[None, :])
_, valid["dup_cluster"] = connected_components(csr_matrix(adj), directed=False)

g = valid.groupby("dup_cluster")
valid["cluster_n_img"] = g.file.transform("size"); valid["cluster_n_writers"] = g.writer_id.transform("nunique")
valid["dup_same_writer"]  = (valid.cluster_n_img > 1) & (valid.cluster_n_writers == 1)   # EN identik HI pada penulis yang sama
valid["dup_cross_writer"] = valid.cluster_n_writers > 1                                    # citra sama pada beberapa ID penulis

print(f"Citra dalam kelompok duplikat: {int((valid.cluster_n_img > 1).sum())} | antar penulis: {int(valid.dup_cross_writer.sum())} citra / "
      f"{valid.loc[valid.dup_cross_writer, 'writer_id'].nunique()} penulis | sama penulis (EN=HI): "
      f"penulis {sorted(int(w) for w in valid.loc[valid.dup_same_writer, 'writer_id'].unique())}")
display(valid[valid.cluster_n_img > 1].groupby("dup_cluster")
        .agg(n_citra=("file", "size"), n_penulis=("writer_id", "nunique"), contoh=("file", lambda s: ", ".join(list(s)[:6])))
        .sort_values("n_citra", ascending=False).head(15))

h2 = ham.copy(); np.fill_diagonal(h2, 999); nn = h2.min(axis=1)
fig, ax = plt.subplots(figsize=(7, 3.4)); sns.histplot(nn, bins=50, ax=ax); ax.axvline(NEAR_DUP_MAX_HAMMING + .5, color="r", ls="--")
ax.set(xlabel="jarak Hamming ke tetangga terdekat (dari 256 bit)", title="Ambang duplikat (garis merah): puncak tajam di ~0, lalu lembah")
plt.tight_layout(); plt.show()


# ================== ORIGINAL NOTEBOOK CELL 19 ==================
img = (meta.merge(audit[["file", "readable", "blank", "error"]], on="file", how="left")
           .merge(valid[["file", "dup_cluster", "dup_same_writer", "dup_cross_writer"]], on="file", how="left"))
for c in ["readable", "blank", "dup_same_writer", "dup_cross_writer"]:
    img[c] = img[c].fillna(False).astype(bool)
img["missing"] = ~img.file_exists
img["unreadable"] = img.file_exists & ~img.readable

RULES = [("berkas_hilang", "missing", "Berkas tercantum di CSV tetapi tidak ada di folder images/"),
         ("tidak_terbaca", "unreadable", "Berkas ada tetapi gagal didekode (rusak/terpotong)"),
         ("citra_kosong", "blank", f"Citra hampir polos (simpangan baku piksel < {BLANK_STD_MIN})"),
         ("duplikat_EN_HI_penulis_sama", "dup_same_writer", "Citra EN identik/hampir identik dengan citra HI milik penulis yang sama"),
         ("duplikat_antar_penulis", "dup_cross_writer", "Citra sama/hampir sama muncul pada ID penulis berbeda")]

remaining = pd.Series(True, index=sorted(img.writer_id.unique())); reason = pd.Series("", index=remaining.index)
rows = []
for name, col, desc in RULES:
    has = img.groupby("writer_id")[col].any()
    hit = has & remaining
    sub = img[img.writer_id.isin(hit[hit].index)]
    reason[hit] = name; remaining &= ~has
    rows.append(dict(aturan=name, keterangan=desc, penulis_dibuang=int(hit.sum()), citra_dibuang=len(sub),
                     citra_bermasalah=int(sub[col].sum()), citra_sehat_ikut_terbuang=len(sub) - int(sub[col].sum()),
                     penulis_tersisa=int(remaining.sum()), citra_tersisa=int(remaining.sum()) * 2,
                     id_penulis=", ".join(str(w) for w in sorted(hit[hit].index)[:12]) + (" ..." if hit.sum() > 12 else "")))
wf = pd.DataFrame(rows)
tot = dict(aturan="TOTAL", keterangan="", penulis_dibuang=wf.penulis_dibuang.sum(), citra_dibuang=wf.citra_dibuang.sum(),
           citra_bermasalah=wf.citra_bermasalah.sum(), citra_sehat_ikut_terbuang=wf.citra_sehat_ikut_terbuang.sum(),
           penulis_tersisa=wf.penulis_tersisa.iloc[-1], citra_tersisa=wf.citra_tersisa.iloc[-1], id_penulis="")
display(pd.concat([wf, pd.DataFrame([tot])], ignore_index=True).drop(columns=["keterangan"]))
wf.to_csv(OUT_DIR / "waterfall_eksklusi.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(15, 3.8))
vals = [len(img)] + wf.citra_tersisa.tolist(); labs = ["Awal"] + wf.aturan.str.replace("_", "\n").tolist()
b = ax[0].bar(range(len(vals)), vals, color=["#4C72B0"] + ["#DD8452"] * (len(vals) - 2) + ["#55A868"])
ax[0].bar_label(b); ax[0].set_xticks(range(len(vals))); ax[0].set_xticklabels(labs, fontsize=7); ax[0].set_title("Citra tersisa setelah tiap aturan (waterfall)")
ax[1].barh(wf.aturan, wf.citra_bermasalah, color="#C44E52", label="citra bermasalah")
ax[1].barh(wf.aturan, wf.citra_sehat_ikut_terbuang, left=wf.citra_bermasalah, color="#8C8C8C", label="citra sehat (pasangan ikut terbuang)")
ax[1].invert_yaxis(); ax[1].legend(); ax[1].set_title("Citra yang dibuang menurut alasan"); plt.tight_layout(); plt.show()

clean = img[img.writer_id.isin(remaining[remaining].index)].copy()
cnt = clean.groupby("writer_id").agg(n=("file", "size"), n_en=("lang", lambda s: (s == "EN").sum()), n_hi=("lang", lambda s: (s == "HI").sum()))
assert (cnt.n == 2).all() and (cnt.n_en == 1).all() and (cnt.n_hi == 1).all(), "Ada penulis tanpa tepat 1 EN + 1 HI"
print(f"\nHimpunan bersih: {clean.writer_id.nunique()} penulis, {len(clean)} citra ({len(clean)/len(meta)*100:.1f}% dari {len(meta)} citra awal; "
      f"EN:HI = {(clean.lang=='EN').sum()}:{(clean.lang=='HI').sum()})")

excl = img[~img.writer_id.isin(remaining[remaining].index)].copy(); excl["alasan_penulis_dikeluarkan"] = excl.writer_id.map(reason)
excl[["file", "writer_id", "lang", "alasan_penulis_dikeluarkan", "missing", "unreadable", "blank", "dup_same_writer", "dup_cross_writer", "error"]
     ].to_csv(OUT_DIR / "exclusion_log.csv", index=False)


# ================== ORIGINAL NOTEBOOK CELL 20 ==================
# Perbandingan dengan angka pada proposal V12 (Gambar 2)
cmp = []
for _, r in wf.iterrows():
    ref = PROPOSAL_REF[r.aturan]; now = (r.penulis_dibuang, r.citra_dibuang, r.citra_bermasalah, r.citra_sehat_ikut_terbuang)
    cmp.append(dict(aturan=r.aturan, kode=str(now), proposal=str(ref), sama="✓" if tuple(map(int, now)) == ref else "≠"))
cmp.append(dict(aturan="penulis/citra tersisa", kode=f"{int(wf.penulis_tersisa.iloc[-1])} / {int(wf.citra_tersisa.iloc[-1])}",
                proposal=f"{PROPOSAL_FINAL['penulis']} / {PROPOSAL_FINAL['citra']}",
                sama="✓" if (int(wf.penulis_tersisa.iloc[-1]), int(wf.citra_tersisa.iloc[-1])) == (PROPOSAL_FINAL['penulis'], PROPOSAL_FINAL['citra']) else "≠"))
print("(penulis dibuang, citra dibuang, citra bermasalah, citra sehat ikut terbuang)"); display(pd.DataFrame(cmp))
if any(c["sama"] == "≠" for c in cmp):
    print("⚠️ Ada selisih terhadap proposal. Selisih pada 'berkas_hilang' berarti salinan data/daftar manual Anda berbeda dari berkas ini "
          "(mis. berkas yang Anda hapus manual). Isi MANUAL_REMOVE / MANUAL_ROTATE sesuai daftar Anda (section A1), atau perbarui angka pada naskah.")


# ================== ORIGINAL NOTEBOOK CELL 22 ==================
clean["dup_cluster"] = clean["dup_cluster"].astype(int)
clean["group_id"] = clean["writer_id"]                       # unit split = ID penulis

units = np.array(sorted(clean.group_id.unique()))
perm = np.random.RandomState(SEED).permutation(units)
n_train = int(np.floor(len(units) * (1 - TEST_RATIO) + 0.5))   # pembulatan setengah-ke-atas
train_units, test_units = np.sort(perm[:n_train]), np.sort(perm[n_train:])

unit_fold = {}
for k, (_, va) in enumerate(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(train_units)):
    for u in train_units[va]:
        unit_fold[u] = k

clean["split"] = np.where(clean.group_id.isin(test_units), "test", "train")
clean["fold"] = clean.group_id.map(unit_fold).fillna(-1).astype(int)       # -1 = data uji
for t in TRAITS:
    clean[f"{t}_s"] = clean[t] / LABEL_MAX                                  # pembagi tetap 40
assert clean[[f"{t}_s" for t in TRAITS]].min().min() >= 0 and clean[[f"{t}_s" for t in TRAITS]].max().max() <= 1

manifest = clean.sort_values(["writer_id", "lang"]).reset_index(drop=True)
manifest["img_idx"] = np.arange(len(manifest))
manifest = manifest[["img_idx", "file", "writer_id", "group_id", "lang", "split", "fold", "dup_cluster"] + TRAITS + [f"{t}_s" for t in TRAITS]]
manifest.to_csv(OUT_DIR / "split_manifest.csv", index=False)

# berkas tetap (catatan Tabel 5)
pd.DataFrame({"writer_id": test_units}).to_csv(OUT_DIR / "test_writer_ids.csv", index=False)
manifest[manifest.split == "train"].drop_duplicates("writer_id")[["writer_id", "fold"]].sort_values("writer_id").to_csv(OUT_DIR / "fold_indices.csv", index=False)

# ---- Tabel 4 ----
def summarize(df):
    return pd.Series({"Penulis": df.writer_id.nunique(), "Citra Inggris": (df.lang == "EN").sum(), "Citra Hindi": (df.lang == "HI").sum(), "Total citra": len(df)})
tab4 = manifest.groupby("split").apply(summarize).rename(index={"train": "Data latih (5-Fold CV)", "test": "Data uji terisolasi"})
tab4.loc["Total"] = tab4.sum(); tab4["Proporsi"] = (tab4["Total citra"] / tab4.loc["Total", "Total citra"] * 100).round(1).astype(str) + " %"
print("Tabel 4 (hasil kode):"); display(tab4)
fold_w = manifest[manifest.split == "train"].groupby("fold").writer_id.nunique().tolist()
fold_i = manifest[manifest.split == "train"].groupby("fold").size().tolist()
print("Penulis per fold:", fold_w, "| citra per fold:", fold_i)
print("Tabel 4 proposal   : latih", PROPOSAL_FINAL["train"], "penulis, uji", PROPOSAL_FINAL["test"], "penulis; fold", PROPOSAL_FINAL["fold_penulis"])
same = (tab4.loc["Data latih (5-Fold CV)", "Penulis"], tab4.loc["Data uji terisolasi", "Penulis"], sorted(fold_w, reverse=True)) == \
       (PROPOSAL_FINAL["train"], PROPOSAL_FINAL["test"], PROPOSAL_FINAL["fold_penulis"])
print("Kesesuaian dengan Tabel 4 proposal:", "✓ sama" if same else "≠ berbeda (mengikuti jumlah penulis bersih di section A7; perbarui Tabel 4 atau samakan seleksi manual)")


# ================== ORIGINAL NOTEBOOK CELL 23 ==================
w_level = manifest[manifest.lang == "EN"]
print("Rata-rata & simpangan baku skor (0-40) per split (informasi saja):"); display(w_level.groupby("split")[TRAITS].agg(["mean", "std"]).round(2))


# ================== ORIGINAL NOTEBOOK CELL 25 ==================
tr = manifest[manifest.split == "train"]; te = manifest[manifest.split == "test"]
# Pembersihan: tak ada citra duplikat/hampir-duplikat tersisa (tiap kelompok duplikat = 1 citra)
assert manifest.dup_cluster.is_unique, "Masih ada citra duplikat pada data bersih"
# Split latih/uji: tak ada penulis yang sama di kedua subset; dua citra penulis selalu satu split & satu fold
assert set(tr.writer_id).isdisjoint(set(te.writer_id)), "Penulis ada di latih DAN uji"
per_w = manifest.groupby("writer_id").agg(n=("file", "size"), n_split=("split", "nunique"), n_fold=("fold", "nunique"))
assert (per_w.n == 2).all() and (per_w.n_split == 1).all() and (per_w.n_fold == 1).all()
# 5-Fold: fold saling lepas berbasis penulis, menutup seluruh data latih; uji tidak punya fold
assert sorted(tr.fold.unique()) == list(range(N_FOLDS)) and (te.fold == -1).all()
assert tr.groupby("writer_id").fold.nunique().max() == 1
# Berkas tetap konsisten dengan manifest dan tidak beririsan
tw = set(pd.read_csv(OUT_DIR / "test_writer_ids.csv").writer_id); fi = pd.read_csv(OUT_DIR / "fold_indices.csv")
assert tw == set(te.writer_id) and tw.isdisjoint(set(fi.writer_id)) and set(fi.writer_id) == set(tr.writer_id)
# Normalisasi/penskalaan: konstanta tetap (bukan statistik data)
assert IMAGENET_MEAN == [0.485, 0.456, 0.406] and IMAGENET_STD == [0.229, 0.224, 0.225] and LABEL_MAX == 40.0
print("✅ Semua pemeriksaan kebocoran lulus.")
print(f"   latih: {tr.writer_id.nunique()} penulis | uji: {te.writer_id.nunique()} penulis | fold latih (penulis): {fold_w}")


# ================== ORIGINAL NOTEBOOK CELL 27 ==================
def resize_image(im, size=IMG_SIZE, mode=RESIZE_MODE):
    if mode == "squash":
        return im.resize((size, size), LANCZOS)
    if mode == "pad":
        scale = size / max(im.size); new = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
        canvas = Image.new("RGB", (size, size), (255, 255, 255)); canvas.paste(im.resize(new, LANCZOS), ((size - new[0]) // 2, (size - new[1]) // 2))
        return canvas
    raise ValueError(mode)

X = np.stack([np.asarray(resize_image(load_rgb(IMG_DIR / f)), dtype=np.uint8) for f in tqdm(manifest.file, desc="Resize 224x224")])
np.save(OUT_DIR / "images_224.npy", X)
assert X.shape == (len(manifest), IMG_SIZE, IMG_SIZE, 3) and X.dtype == np.uint8
print("images_224.npy:", X.shape, X.dtype, f"({X.nbytes/1e6:.0f} MB) | rentang piksel {X.min()}-{X.max()}")


# ================== ORIGINAL NOTEBOOK CELL 28 ==================
pick = pd.concat([manifest[manifest.lang == "EN"].sample(4, random_state=SEED), manifest[manifest.lang == "HI"].sample(4, random_state=SEED)])
fig, axes = plt.subplots(4, 4, figsize=(14, 10))
for k, (_, r) in enumerate(pick.iterrows()):
    row, col = (k // 4) * 2, k % 4; orig = load_rgb(IMG_DIR / r.file)
    axes[row, col].imshow(orig); axes[row, col].set_title(f"{r.file}  asli {orig.size[0]}x{orig.size[1]}", fontsize=9)
    axes[row + 1, col].imshow(X[r.img_idx]); axes[row + 1, col].set_title("224 x 224", fontsize=9)
for a in axes.ravel(): a.axis("off")
plt.tight_layout(); plt.show()


# ================== ORIGINAL NOTEBOOK CELL 30 ==================
summary = dict(
    config=dict(SEED=SEED, IMG_SIZE=IMG_SIZE, LABEL_MAX=LABEL_MAX, TEST_RATIO=TEST_RATIO, N_FOLDS=N_FOLDS, OPENNESS_FIX=OPENNESS_FIX,
                NEAR_DUP_MAX_HAMMING=NEAR_DUP_MAX_HAMMING, BLANK_STD_MIN=BLANK_STD_MIN, RESIZE_MODE=RESIZE_MODE,
                APPLY_MANUAL=APPLY_MANUAL, MANUAL_ROTATE=MANUAL_ROTATE, MANUAL_REMOVE=MANUAL_REMOVE,
                IMAGENET_MEAN=IMAGENET_MEAN, IMAGENET_STD=IMAGENET_STD, TRAITS=TRAITS, AUGMENTATION=False),
    counts=dict(rows_metadata=int(len(meta)), writers_metadata=int(meta.writer_id.nunique()), writers_clean=int(manifest.writer_id.nunique()),
                images_clean=int(len(manifest)), writers_train=int(tr.writer_id.nunique()), writers_test=int(te.writer_id.nunique()),
                fold_writers=fold_w))
(OUT_DIR / "config.json").write_text(json.dumps(summary, indent=2))
for p in sorted(OUT_DIR.iterdir()):
    print(f"{p.name:24s} {p.stat().st_size/1e6:8.2f} MB")
