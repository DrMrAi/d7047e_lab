"""
Task 1.3 – Model Comparison: Simple ANN vs Bi-LSTM vs Transformer
Runs on BOTH the small (1k) and large (25k) datasets and saves a separate
PNG for each: model_comparison_SMALL.png and model_comparison_LARGE.png.
Run: python model_comparison.py
"""


# pip install torch scikit-learn nltk pandas matplotlib seaborn

import math, copy, time, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – swap to TkAgg / Qt5Agg if you want pop-ups
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import nltk
nltk.download("punkt",      quiet=True)
nltk.download("stopwords",  quiet=True)

from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

# ─── Config ───────────────────────────────────────────────────────────────────
DATASETS = {
    "SMALL": "/home/jovyan/ludvig/lab1/amazon_cells_labelled.txt",
    "LARGE": "/home/jovyan/ludvig/lab1/amazon_cells_labelled_LARGE_25K.txt",
}
TEST_SIZE    = 0.15
RANDOM_STATE = 42
EPOCHS       = 15
BATCH_SIZE   = 32
LR           = 1e-3
MAX_LEN      = 60

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  Device : {DEVICE}")
print(f"  Epochs : {EPOCHS}")
print(f"{'='*60}\n")


# ════════════════════════════════════════════════════════════════
# 1. PREPROCESSING
# ════════════════════════════════════════════════════════════════
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    stop = set(stopwords.words("english"))
    df   = data.copy()
    df["Sentence"] = df["Sentence"].str.lower()
    df["Sentence"] = df["Sentence"].replace(r"[a-zA-Z0-9_.]+@[a-zA-Z0-9_.]+", "", regex=True)
    df["Sentence"] = df["Sentence"].replace(r"(\d{1,3}\.){3}\d{1,3}", "",          regex=True)
    df["Sentence"] = df["Sentence"].str.replace(r"[^\w\s]", "",                     regex=True)
    df["Sentence"] = df["Sentence"].replace(r"\d+", "",                             regex=True)

    cleaned = []
    for sent in df["Sentence"]:
        tokens  = word_tokenize(sent)
        filtered = [w for w in tokens if w not in stop]
        cleaned.append(" ".join(filtered))
    df["Sentence"] = cleaned
    return df


def load_data(path: str):
    raw = pd.read_csv(path, delimiter="\t", header=None, names=["Sentence","Class"])
    raw = preprocess(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        raw["Sentence"].values.astype("U"),
        raw["Class"].values.astype("int32"),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=raw["Class"].values
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"Class balance (train) – 0: {(y_train==0).sum()}  1: {(y_train==1).sum()}\n")
    return X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════════════
# 2. MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════
class SimpleANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=128, dropout=0.3):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden*2, 1)

    def forward(self, x, lengths):
        x      = self.emb(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(self.drop(h))


class TransformerCls(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, max_len=MAX_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab_size+1, d_model, padding_idx=0)

        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i]   = math.sin(pos / (10000 ** (i / d_model)))
                if i+1 < d_model:
                    pe[pos, i+1] = math.cos(pos / (10000 ** (i / d_model)))
        self.register_buffer("pe", pe.unsqueeze(0))

        layer          = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                                    batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.fc          = nn.Linear(d_model, 1)

    def forward(self, x, lengths=None):
        x = self.emb(x) + self.pe[:, :x.size(1)]
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))


# ════════════════════════════════════════════════════════════════
# 3. DATA HELPERS
# ════════════════════════════════════════════════════════════════
def tfidf_loaders(X_train, X_test, y_train, y_test):
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=30000)
    trX  = torch.tensor(vect.fit_transform(X_train).toarray(), dtype=torch.float32)
    teX  = torch.tensor(vect.transform(X_test).toarray(),      dtype=torch.float32)
    trY  = torch.tensor(y_train, dtype=torch.float32)
    teY  = torch.tensor(y_test,  dtype=torch.float32)
    tr   = DataLoader(TensorDataset(trX, trY), batch_size=BATCH_SIZE, shuffle=True)
    te   = DataLoader(TensorDataset(teX, teY), batch_size=BATCH_SIZE)
    return tr, te, trX.shape[1]


def build_vocab(sentences, max_size=8000):
    c = Counter(w for s in sentences for w in s.split())
    vocab = {w: i+2 for i,(w,_) in enumerate(c.most_common(max_size))}
    vocab["<PAD>"] = 0; vocab["<UNK>"] = 1
    return vocab


def encode_seq(sentences, vocab, max_len=MAX_LEN):
    seqs, lens = [], []
    for s in sentences:
        tok = s.split()
        seq = [vocab.get(w,1) for w in tok]
        lens.append(min(len(seq), max_len))
        seq = seq[:max_len] + [0]*(max_len - len(seq))
        seqs.append(seq)
    return np.array(seqs), np.array(lens)


def seq_loaders(X_train, X_test, y_train, y_test):
    vocab = build_vocab(X_train)
    trSeq, trLen = encode_seq(X_train, vocab)
    teSeq, teLen = encode_seq(X_test,  vocab)
    def make(seq, lbl, lng):
        return DataLoader(
            TensorDataset(torch.LongTensor(seq),
                          torch.FloatTensor(lbl),
                          torch.LongTensor(lng)),
            batch_size=BATCH_SIZE, shuffle=(lbl is y_train)
        )
    return make(trSeq, y_train, trLen), make(teSeq, y_test, teLen), len(vocab)


# ════════════════════════════════════════════════════════════════
# 4. TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer):
    model.train(); total = 0
    for batch in loader:
        optimizer.zero_grad()
        if len(batch) == 3:
            X, y, lens = batch
            out = model(X.to(DEVICE), lens.to(DEVICE))
        else:
            X, y = batch
            out = model(X.to(DEVICE))
        y   = y.float().to(DEVICE).unsqueeze(1)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        total += loss.item()
    return total / len(loader)


def eval_epoch(model, loader, criterion):
    model.eval(); total = 0; preds_all = []; labels_all = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                X, y, lens = batch
                out = model(X.to(DEVICE), lens.to(DEVICE))
            else:
                X, y = batch
                out = model(X.to(DEVICE))
            y   = y.float().to(DEVICE).unsqueeze(1)
            total += criterion(out, y).item()
            preds_all.extend((torch.sigmoid(out) > 0.5).int().cpu().numpy())
            labels_all.extend(y.cpu().numpy())
    acc = accuracy_score(labels_all, preds_all)
    return total/len(loader), acc, preds_all, labels_all


def run_training(name, model, tr_loader, te_loader, criterion, optimizer):
    model = model.to(DEVICE)
    best_state = copy.deepcopy(model.state_dict()); best_val = float("inf")
    tr_losses=[]; val_losses=[]; val_accs=[]

    print(f"\n{'─'*50}")
    print(f"  Training: {name}")
    print(f"{'─'*50}")
    t0 = time.time()

    for ep in range(EPOCHS):
        tl = train_epoch(model, tr_loader, criterion, optimizer)
        vl, va, _, _ = eval_epoch(model, te_loader, criterion)
        tr_losses.append(tl); val_losses.append(vl); val_accs.append(va)
        print(f"  Epoch {ep+1:02d}/{EPOCHS} | train={tl:.4f}  val={vl:.4f}  acc={va:.4f}")
        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(model.state_dict())

    elapsed = time.time() - t0
    model.load_state_dict(best_state)

    # final metrics
    _, _, preds, labels = eval_epoch(model, te_loader, criterion)
    metrics = {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "cm":        confusion_matrix(labels, preds),
        "report":    classification_report(labels, preds),
        "train_losses": tr_losses,
        "val_losses":   val_losses,
        "val_accs":     val_accs,
        "train_time_s": elapsed,
        "n_params":     sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    print(f"\n  Test accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1-score      : {metrics['f1']:.4f}")
    print(f"  Train time    : {elapsed:.1f}s")
    print(f"  Parameters    : {metrics['n_params']:,}")
    print(metrics["report"])
    return model, metrics


# ════════════════════════════════════════════════════════════════
# 5. PLOTTING
# ════════════════════════════════════════════════════════════════
PALETTE = {"ANN": "#4C72B0", "Bi-LSTM": "#DD8452", "Transformer": "#55A868"}

def plot_all(results: dict, tag: str = ""):
    names = list(results.keys())
    fig   = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor("#F8F9FA")
    gs    = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── (A) Training loss curves ──────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for name, m in results.items():
        ax.plot(range(1, EPOCHS+1), m["train_losses"], label=name,
                color=PALETTE[name], linewidth=2)
    ax.set_title("Training loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.legend(); ax.set_facecolor("#FFFFFF"); ax.grid(alpha=0.3)

    # ── (B) Validation loss curves ───────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for name, m in results.items():
        ax.plot(range(1, EPOCHS+1), m["val_losses"], label=name,
                color=PALETTE[name], linewidth=2, linestyle="--")
    ax.set_title("Validation loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.legend(); ax.set_facecolor("#FFFFFF"); ax.grid(alpha=0.3)

    # ── (C) Validation accuracy curves ───────────────────────
    ax = fig.add_subplot(gs[0, 2])
    for name, m in results.items():
        ax.plot(range(1, EPOCHS+1), m["val_accs"], label=name,
                color=PALETTE[name], linewidth=2)
    ax.set_title("Validation accuracy", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.0); ax.legend()
    ax.set_facecolor("#FFFFFF"); ax.grid(alpha=0.3)

    # ── (D-F) Confusion matrices ──────────────────────────────
    for col, (name, m) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, col])
        sns.heatmap(m["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Neg","Pos"], yticklabels=["Neg","Pos"],
                    ax=ax, cbar=False, linewidths=0.5)
        ax.set_title(f"{name} – Confusion matrix", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    # ── (G) Bar chart – metric comparison ────────────────────
    ax   = fig.add_subplot(gs[2, 0:2])
    metrics_labels = ["accuracy", "precision", "recall", "f1"]
    x    = np.arange(len(metrics_labels))
    w    = 0.25
    for i, (name, m) in enumerate(results.items()):
        vals = [m[k] for k in metrics_labels]
        bars = ax.bar(x + i*w, vals, width=w, label=name,
                      color=PALETTE[name], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + w)
    ax.set_xticklabels([m.capitalize() for m in metrics_labels])
    ax.set_ylim(0, 1.12); ax.set_title("Test metrics comparison", fontsize=13, fontweight="bold")
    ax.legend(); ax.set_facecolor("#FFFFFF"); ax.grid(axis="y", alpha=0.3)

    # ── (H) Parameters vs Accuracy bubble ────────────────────
    ax = fig.add_subplot(gs[2, 2])
    for name, m in results.items():
        ax.scatter(m["n_params"]/1e3, m["accuracy"]*100,
                   s=m["train_time_s"]*3+80,
                   color=PALETTE[name], alpha=0.85, edgecolors="white", linewidth=1.5, label=name)
        ax.annotate(name, (m["n_params"]/1e3, m["accuracy"]*100),
                    textcoords="offset points", xytext=(8,4), fontsize=9)
    ax.set_xlabel("Parameters (k)"); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs complexity\n(bubble size = train time)", fontsize=12, fontweight="bold")
    ax.set_facecolor("#FFFFFF"); ax.grid(alpha=0.3)

    fig.suptitle(f"Task 1.3 – Model comparison: ANN vs Bi-LSTM vs Transformer  [{tag} dataset]",
                 fontsize=16, fontweight="bold", y=0.98)

    out = f"model_comparison_{tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Plot saved → {out}")
    return out


def print_summary_table(results: dict):
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    header = f"{'Model':<14} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Params':>10} {'Time(s)':>9}"
    print(header)
    print("─"*70)
    for name, m in results.items():
        print(f"{name:<14} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} "
              f"{m['n_params']:>10,} {m['train_time_s']:>9.1f}")
    print("─"*70)


# ════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════
def run_experiment(tag: str, data_path: str):
    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT: {tag} dataset")
    print(f"  Path: {data_path}")
    print(f"{'#'*60}")

    X_train, X_test, y_train, y_test = load_data(data_path)
    results = {}

    # ── ANN (TF-IDF input) ─────────────────────────────────────
    tr_ann, te_ann, input_dim = tfidf_loaders(X_train, X_test, y_train, y_test)
    model_ann  = SimpleANN(input_dim)
    _, res_ann = run_training(
        "ANN", model_ann, tr_ann, te_ann,
        nn.BCEWithLogitsLoss(),
        optim.Adam(model_ann.parameters(), lr=LR)
    )
    results["ANN"] = res_ann

    # ── Bi-LSTM ────────────────────────────────────────────────
    tr_lstm, te_lstm, vocab_size = seq_loaders(X_train, X_test, y_train, y_test)
    model_lstm  = BiLSTM(vocab_size)
    _, res_lstm = run_training(
        "Bi-LSTM", model_lstm, tr_lstm, te_lstm,
        nn.BCEWithLogitsLoss(),
        optim.Adam(model_lstm.parameters(), lr=LR)
    )
    results["Bi-LSTM"] = res_lstm

    # ── Transformer ────────────────────────────────────────────
    model_tfm  = TransformerCls(vocab_size)
    _, res_tfm = run_training(
        "Transformer", model_tfm, tr_lstm, te_lstm,   # same seq loaders
        nn.BCEWithLogitsLoss(),
        optim.Adam(model_tfm.parameters(), lr=5e-4)
    )
    results["Transformer"] = res_tfm

    # ── Report & plots ─────────────────────────────────────────
    print_summary_table(results)
    plot_all(results, tag=tag)
    return results


def main():
    all_results = {}
    for tag, path in DATASETS.items():
        all_results[tag] = run_experiment(tag, path)

    # ── Cross-dataset accuracy summary ─────────────────────────
    print(f"\n{'='*70}")
    print(f"  CROSS-DATASET ACCURACY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<14} {'SMALL acc':>12} {'LARGE acc':>12} {'Delta':>10}")
    print("─"*70)
    for model in ["ANN", "Bi-LSTM", "Transformer"]:
        s = all_results["SMALL"][model]["accuracy"]
        l = all_results["LARGE"][model]["accuracy"]
        print(f"{model:<14} {s:>12.4f} {l:>12.4f} {l-s:>+10.4f}")
    print("─"*70)

    print("\n  Done! Saved model_comparison_SMALL.png and model_comparison_LARGE.png\n")


if __name__ == "__main__":
    main()