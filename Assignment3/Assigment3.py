"""
Distribution and K-means clustering of protein main-chain angles (phi, psi)

Run:    python assignment_3.py          (in PyCharm: right-click the file > Run)
        data_assignment3.csv must be in the same folder as this script.
Needs:  pandas, numpy, matplotlib, scikit-learn
Output: figure1..figure7 PNG files saved next to this script (not shown on screen)
        + a results summary printed at the end.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # save figures to files, no window needed
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


# ============================================================================
# SETTINGS
# ============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))     # the script's own folder
CSV_PATH = os.path.join(HERE, "data_assignment3.csv")

K_RANGE = range(2, 11)          # values of K tried in part 2a and 2c
K_GRID = range(2, 8)            # values of K drawn side by side in figure 4
K_CHOSEN = 3                    # the K we suggest (motivated in the report)
SEED = 0                        # fixed random seed, so every run gives the same result
N_INIT = 10                     # K-means is restarted 10 times from different centres,
                                # the best run is kept (protects against bad local minima)
SIL_SAMPLE = 8000               # silhouette is computed on a random sample of 8000 points
                                # (all 29,369 points would need ~860 million distances)

# Colours: one fixed colour per cluster (largest cluster first), never reused
CLUSTER_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
# Single-hue blue scale for the 2D histogram (light = few points, dark = many)
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
GREY = "#b8b7b2"

# The three well-known regions of the Ramachandran plot (used for labels only)
REGION_COLOUR = {"beta-sheet": CLUSTER_COLOURS[0],
                 "alpha-helix": CLUSTER_COLOURS[1],
                 "left-handed helix": CLUSTER_COLOURS[2]}
REGION_LABEL = {"beta-sheet": "β-sheet",
                "alpha-helix": "α-helix",
                "left-handed helix": "left-handed helix"}


def savefig(fig, name):
    """Save a figure as PNG in the script's folder and close it."""
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name), dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def style_rama(ax, title):
    """Common look for every phi/psi plot: full -180..180 range, square, same ticks."""
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal")
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-180, 181, 60))
    ax.axhline(0, color="#52514e", lw=0.6, zorder=0)
    ax.axvline(0, color="#52514e", lw=0.6, zorder=0)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel("φ (phi), degrees")
    ax.set_ylabel("ψ (psi), degrees")
    ax.set_title(title)


def label_regions(ax):
    """Write the names of the three known regions next to them (with a pointer)."""
    for text, xy, xytext in [("β-sheet", (-115, 140), (-40, 150)),
                             ("α-helix", (-65, -40), (5, -95)),
                             ("left-handed\nhelix", (60, 40), (110, 105))]:
        ax.annotate(text, xy=xy, xytext=xytext, fontsize=9, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color="#0b0b0b", lw=0.8))


# ============================================================================
# 1. Load the data                                                [part 1]
#    Each row is one amino acid residue: its type, position, chain and the
#    two angles phi and psi (degrees, from -180 to 180).
# ============================================================================
df = pd.read_csv(CSV_PATH)
phi = df["phi"].to_numpy(float)
psi = df["psi"].to_numpy(float)
X = np.column_stack([phi, psi])                 # the data that is clustered in part 2a
n_missing = int(df[["phi", "psi"]].isna().sum().sum())


# ============================================================================
# 2. Figure 1: scatter plot, default vs improved                  [part 1a]
#    Left: matplotlib defaults. With ~29,000 points the big solid dots cover
#    each other and the dense areas become one blob.
#    Right: small, semi-transparent points (dense areas look darker), the full
#    -180..180 range on both axes, equal scales, and the known regions named.
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(phi, psi)                       # default settings
axes[0].set(xlabel="phi", ylabel="psi", title="(a) Default scatter plot")

axes[1].scatter(phi, psi, s=2, alpha=0.15, color=CLUSTER_COLOURS[0], lw=0, rasterized=True)
style_rama(axes[1], "(b) Improved scatter plot")
label_regions(axes[1])

fig.suptitle(f"Figure 1: Scatter plot of phi and psi ({len(df):,} residues)")
savefig(fig, "figure1_scatter_plot.png")


# ============================================================================
# 3. Figure 2: 2D histogram, linear vs logarithmic colour scale   [part 1b]
#    The plane is cut into 5 x 5 degree squares (72 x 72 bins); the colour
#    shows how many residues fall in each square. Empty squares stay white.
#    Linear scale: only the two densest peaks are visible.
#    Log scale: also shows the rarer, but real, regions.
# ============================================================================
BINS = np.arange(-180, 181, 5)
counts, _, _ = np.histogram2d(phi, psi, bins=[BINS, BINS])

fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
_, _, _, im = axes[0].hist2d(phi, psi, bins=[BINS, BINS], cmap=BLUES, cmin=1)
fig.colorbar(im, ax=axes[0], shrink=0.8, label="Residues per 5° × 5° bin")
style_rama(axes[0], "(a) Linear colour scale")

_, _, _, im = axes[1].hist2d(phi, psi, bins=[BINS, BINS], cmap=BLUES, cmin=1,
                             norm=LogNorm(vmin=1, vmax=counts.max()))
fig.colorbar(im, ax=axes[1], shrink=0.8, label="Residues per 5° × 5° bin (log scale)")
style_rama(axes[1], "(b) Logarithmic colour scale")
label_regions(axes[1])

fig.suptitle("Figure 2: 2D histogram of phi and psi")
savefig(fig, "figure2_2d_histogram.png")


# ============================================================================
# 4. K-means helpers                                              [parts 2a-2c]
# ============================================================================
def run_kmeans(features, k, seed=SEED):
    """K-means with N_INIT restarts. Labels are renumbered so that 0 is the
    largest cluster, 1 the next, and so on (keeps colours consistent)."""
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(features)
    order = np.argsort(-np.bincount(km.labels_))           # biggest cluster first
    new_label = np.empty(k, int)
    new_label[order] = np.arange(k)
    return new_label[km.labels_], km.inertia_


def centres_in_degrees(labels, circular=False):
    """Centre of each cluster as (phi, psi) in degrees.
    circular=False: ordinary mean (what plain K-means uses as its centre).
    circular=True:  circular mean (average of cos and sin, then atan2), correct
                    for clusters that cross the +-180 edge (part 2c)."""
    out = []
    for c in range(labels.max() + 1):
        pts = X[labels == c]
        if circular:
            a = np.radians(pts)
            out.append(np.degrees(np.arctan2(np.sin(a).mean(0), np.cos(a).mean(0))))
        else:
            out.append(pts.mean(0))
    return np.array(out)


def region_name(centre):
    """Name a cluster by where its centre lies (standard Ramachandran regions):
    phi > 0 -> left-handed helix; phi < 0 and psi > 50 -> beta-sheet; else alpha-helix."""
    c_phi, c_psi = centre
    if c_phi > 0:
        return "left-handed helix"
    return "beta-sheet" if c_psi > 50 else "alpha-helix"


def try_all_k(features):
    """Run K-means for every K in K_RANGE; return a table with SSE and silhouette."""
    rows = []
    for k in K_RANGE:
        labels, sse = run_kmeans(features, k)
        sil = silhouette_score(features, labels, sample_size=SIL_SAMPLE, random_state=SEED)
        rows.append({"K": k, "SSE": sse, "silhouette": sil, "labels": labels})
    return pd.DataFrame(rows)


def plot_choose_k(table, fig_title, sse_unit, filename, sse_scale=1.0):
    """Elbow plot (SSE vs K) and silhouette vs K side by side; chosen K marked.
    sse_scale divides the SSE for readable axis numbers (e.g. 1e6 = millions)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, values, ylabel in [
            (axes[0], table["SSE"] / sse_scale, f"Sum of squared distances ({sse_unit})"),
            (axes[1], table["silhouette"], "Mean silhouette coefficient")]:
        ax.plot(table["K"], values, "-o", color=CLUSTER_COLOURS[0], lw=2, ms=6)
        chosen = values[table["K"] == K_CHOSEN].iloc[0]
        ax.plot(K_CHOSEN, chosen, "o", ms=13, mfc="none", mec="#0b0b0b", mew=1.5)
        ax.annotate(f"K = {K_CHOSEN}", (K_CHOSEN, chosen),
                    textcoords="offset points", xytext=(12, 8))
        ax.set(xlabel="Number of clusters K", ylabel=ylabel, xticks=list(K_RANGE))
        ax.margins(y=0.15)                          # room for the label under the title
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_title("(a) Elbow method: lower is tighter")
    axes[1].set_title("(b) Silhouette: higher is better separated")
    fig.suptitle(fig_title)
    savefig(fig, filename)


def plot_clusters(ax, labels, title, names=None, circular=False):
    """Scatter plot coloured by cluster, with each cluster centre marked."""
    centres = centres_in_degrees(labels, circular)
    for c in range(labels.max() + 1):
        m = labels == c
        colour = REGION_COLOUR[names[c]] if names else CLUSTER_COLOURS[c]
        lab = f"{REGION_LABEL[names[c]]} ({m.sum():,})" if names else None
        ax.scatter(phi[m], psi[m], s=2, alpha=0.3, color=colour, lw=0, rasterized=True)
        if lab:                                             # bigger marker for the legend
            ax.scatter([], [], s=40, color=colour, label=lab)
    ax.scatter(centres[:, 0], centres[:, 1], marker="X", s=110, color="#0b0b0b",
               edgecolor="white", lw=1.2, zorder=5, label="Cluster centre" if names else None)
    style_rama(ax, title)
    if names:                                        # legend under the plot, not on the data
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9,
                  frameon=False)


# ============================================================================
# 5. K-means on the angles as they are, K = 2..10                 [part 2a]
#    phi and psi have the same unit (degrees) and the same range, so no
#    scaling is needed. Two scores help choose K:
#    - SSE (elbow method): sum of squared distances from each point to its
#      cluster centre; look for the K after which it stops falling quickly.
#    - Silhouette: for each point, (b - a) / max(a, b), where a = mean distance
#      to its own cluster and b = mean distance to the nearest other cluster.
#      Close to 1 = clearly in the right cluster; close to 0 = on a border.
# ============================================================================
plain = try_all_k(X)
plot_choose_k(plain, "Figure 3: Choosing K for K-means on the raw angles",
              "million degrees²", "figure3_choose_k.png", sse_scale=1e6)

# Figure 4: what the clusters look like for K = 2..7
fig, axes = plt.subplots(2, 3, figsize=(13, 9))
for ax, k in zip(axes.ravel(), K_GRID):
    labels_k = plain.loc[plain["K"] == k, "labels"].iloc[0]
    plot_clusters(ax, labels_k, f"K = {k}" + ("  (suggested)" if k == K_CHOSEN else ""))
fig.suptitle("Figure 4: K-means clusters for different values of K (black X = cluster centre)")
savefig(fig, "figure4_kmeans_k2_to_k7.png")


# ============================================================================
# 6. The chosen K: are the clusters reasonable?                   [part 2b]
#    Checks: (1) centres vs the known regions, (2) share of glycine (GLY) in
#    each cluster - glycine has no side chain, so it is the residue that most
#    often has phi > 0, (3) stability: same clusters from other random seeds?
# ============================================================================
labels_plain = plain.loc[plain["K"] == K_CHOSEN, "labels"].iloc[0]
centres_plain = centres_in_degrees(labels_plain)
names_plain = [region_name(c) for c in centres_plain]

fig, ax = plt.subplots(figsize=(7, 7.6))
plot_clusters(ax, labels_plain, f"Figure 5: K-means clusters, K = {K_CHOSEN} (raw angles)",
              names_plain)
savefig(fig, "figure5_kmeans_k3.png")

is_gly = (df["residue name"] == "GLY").to_numpy()


def cluster_table(labels, names, circular=False):
    """Size, centre and glycine share of each cluster."""
    centres = centres_in_degrees(labels, circular)
    rows = []
    for c, name in enumerate(names):
        m = labels == c
        rows.append({"cluster": name, "residues": int(m.sum()),
                     "share_%": round(100 * m.mean(), 1),
                     "centre_phi": round(centres[c, 0]), "centre_psi": round(centres[c, 1]),
                     "GLY_share_%": round(100 * is_gly[m].mean(), 1)})
    return pd.DataFrame(rows)


def stability(features):
    """Run K-means again with 5 other seeds; adjusted Rand index (ARI) with the
    first run: 1.0 = exactly the same grouping."""
    ref, _ = run_kmeans(features, K_CHOSEN, seed=SEED)
    return min(adjusted_rand_score(ref, run_kmeans(features, K_CHOSEN, seed=s)[0])
               for s in range(SEED + 1, SEED + 6))


table_plain = cluster_table(labels_plain, names_plain)
ari_plain = stability(X)


# ============================================================================
# 7. Wrap-around (torus) version                                  [part 2c]
#    phi = 180 and phi = -180 are the same angle (same for psi), but plain
#    K-means sees them as 360 degrees apart. Fix: replace each angle by the
#    point (cos, sin) on a circle, so every residue becomes
#        [cos(phi), sin(phi), cos(psi), sin(psi)].
#    Two angles that are close on the circle are now close in this space, also
#    across the +-180 edge. K-means is then run on these 4 numbers, and the
#    centres are converted back to degrees with the circular mean.
# ============================================================================
rad = np.radians(X)
X_torus = np.column_stack([np.cos(rad[:, 0]), np.sin(rad[:, 0]),
                           np.cos(rad[:, 1]), np.sin(rad[:, 1])])

torus = try_all_k(X_torus)
plot_choose_k(torus, "Figure 6: Choosing K for K-means with wrap-around (cos/sin features)",
              "unitless", "figure6_choose_k_wraparound.png")

labels_torus = torus.loc[torus["K"] == K_CHOSEN, "labels"].iloc[0]
names_torus = [region_name(c) for c in centres_in_degrees(labels_torus, circular=True)]
table_torus = cluster_table(labels_torus, names_torus, circular=True)
ari_torus = stability(X_torus)

# Which residues change cluster when the wrap-around is taken into account?
region_plain = np.array(names_plain)[labels_plain]
region_torus = np.array(names_torus)[labels_torus]
changed = region_plain != region_torus
moves = pd.crosstab(pd.Series(region_plain, name="raw angles"),
                    pd.Series(region_torus, name="wrap-around"))
near_bottom = psi < -120                       # beta region continues here from the top
near_right = phi > 150                         # beta region continues here from the left

# Figure 7: raw vs wrap-around, and the residues that moved
fig, axes = plt.subplots(1, 3, figsize=(17, 6.8))
plot_clusters(axes[0], labels_plain, "(a) Raw angles", names_plain)
plot_clusters(axes[1], labels_torus, "(b) With wrap-around", names_torus, circular=True)
axes[2].scatter(phi, psi, s=2, alpha=0.15, color=GREY, lw=0, rasterized=True)
for name in REGION_COLOUR:
    m = changed & (region_torus == name)
    if m.any():
        axes[2].scatter(phi[m], psi[m], s=6, color=REGION_COLOUR[name], lw=0, rasterized=True)
        axes[2].scatter([], [], s=40, color=REGION_COLOUR[name],
                        label=f"now {REGION_LABEL[name]} ({m.sum():,})")
style_rama(axes[2], f"(c) Residues that changed cluster ({changed.sum():,})")
axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9,
               frameon=False)
fig.suptitle(f"Figure 7: K-means with K = {K_CHOSEN}, without and with wrap-around")
savefig(fig, "figure7_raw_vs_wraparound.png")


# ============================================================================
# 8. Results summary                                    [parts 1a, 1b, 2a, 2b, 2c]
# ============================================================================
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
line = "=" * 74

print("\n" + line)
print("RESULTS SUMMARY")
print(line)

print("\n[1  DATA]")
print(f"Residues: {len(df):,} | residue types: {df['residue name'].nunique()} "
      f"| missing phi/psi values: {n_missing}")
print(f"phi range: {phi.min():.0f} to {phi.max():.0f} | psi range: {psi.min():.0f} to {psi.max():.0f}")
print(f"Residues with phi > 0: {np.sum(phi > 0):,} ({100 * np.mean(phi > 0):.1f}%)")

print("\n[1b  2D HISTOGRAM, 5 x 5 degree bins]")
print(f"Non-empty bins: {int((counts > 0).sum()):,} of {counts.size:,} | "
      f"max residues in one bin: {int(counts.max())}")
i, j = np.unravel_index(counts.argmax(), counts.shape)
print(f"Densest bin: phi {BINS[i]:.0f}..{BINS[i + 1]:.0f}, psi {BINS[j]:.0f}..{BINS[j + 1]:.0f}")

print("\n[2a  K-MEANS ON RAW ANGLES: SSE AND SILHOUETTE]")
print(plain[["K", "SSE", "silhouette"]].round({"SSE": 0, "silhouette": 3}).to_string(index=False))
print("Drop in SSE when K grows by 1 (%):",
      {int(k): round(100 * (1 - b / a), 1)
       for k, a, b in zip(plain["K"][1:], plain["SSE"][:-1], plain["SSE"][1:])})

print(f"\n[2b  CHOSEN K = {K_CHOSEN}, RAW ANGLES]")
print(table_plain.to_string(index=False))
print(f"GLY share in the whole data set: {100 * is_gly.mean():.1f}%")
print(f"Share of all GLY that is in the left-handed cluster: "
      f"{100 * is_gly[region_plain == 'left-handed helix'].sum() / is_gly.sum():.1f}%")
print(f"Stability, lowest ARI over 5 other seeds: {ari_plain:.3f}")

print("\n[2c  WRAP-AROUND: SSE AND SILHOUETTE]")
print(torus[["K", "SSE", "silhouette"]].round({"SSE": 1, "silhouette": 3}).to_string(index=False))
print(f"\n[2c  CHOSEN K = {K_CHOSEN}, WRAP-AROUND]")
print(table_torus.to_string(index=False))
print(f"Stability, lowest ARI over 5 other seeds: {ari_torus:.3f}")
print(f"\nResidues that change cluster: {changed.sum():,} ({100 * changed.mean():.1f}%)")
print("From (rows, raw angles) to (columns, wrap-around):")
print(moves.to_string())
print(f"Bottom edge (psi < -120): {near_bottom.sum():,} residues | raw angles: "
      f"{pd.Series(region_plain[near_bottom]).value_counts().to_dict()} | wrap-around: "
      f"{pd.Series(region_torus[near_bottom]).value_counts().to_dict()}")
print(f"Right edge (phi > 150):   {near_right.sum():,} residues | raw angles: "
      f"{pd.Series(region_plain[near_right]).value_counts().to_dict()} | wrap-around: "
      f"{pd.Series(region_torus[near_right]).value_counts().to_dict()}")

print("\nFigures saved in", HERE)
for f in ["figure1_scatter_plot.png", "figure2_2d_histogram.png", "figure3_choose_k.png",
          "figure4_kmeans_k2_to_k7.png", "figure5_kmeans_k3.png",
          "figure6_choose_k_wraparound.png", "figure7_raw_vs_wraparound.png"]:
    print("  ", f)
print(line)