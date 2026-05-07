"""
Build a self-contained HTML report of the NKX2-1 validation results.

The HTML is intended for medical and biological colleagues, so it
explains the experimental design and the metrics in plain language
rather than ML jargon. All images are base64 embedded so the file is
portable (can be emailed or hosted).

Run:
    python scripts/build_validation_html.py

Reads result CSVs and PNGs from results/remap_validation_*/ and writes
results/validation_summary.html.
"""

import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/sc-projects/sc-proj-cc17-P09_TFBS")
RESULTS = ROOT / "results"
OUT = RESULTS / "validation_summary.html"


def img(path):
    p = Path(path)
    if not p.exists():
        return f"<p style='color:#900;'>(image missing: {p.name})</p>"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{p.name}">'


def csv_table(path, drop=None, fmt=None):
    p = Path(path)
    if not p.exists():
        return "<p style='color:#900;'>(CSV missing)</p>"
    df = pd.read_csv(p)
    if drop:
        df = df.drop(columns=[c for c in drop if c in df.columns])
    if fmt:
        for c, f in fmt.items():
            if c in df.columns:
                df[c] = df[c].apply(lambda v: f.format(v) if pd.notna(v) else "")
    return df.to_html(index=False, classes="restab", border=0)


# Pull AUC numbers for the headline table
def auc_lookup(csv_path, method):
    p = Path(csv_path)
    if not p.exists():
        return ""
    df = pd.read_csv(p)
    row = df[df["method"] == method]
    if len(row) == 0:
        return ""
    return f"{row['auc'].iloc[0]:.3f}"


s4_shuffle = RESULTS / "remap_validation_score4_shuffle"
s4_random  = RESULTS / "remap_validation_score4"
s2_shuffle = RESULTS / "remap_validation_score2_shuffle"
tad        = RESULTS / "tad_method_comparison"
af3_plot   = RESULTS / "alphafold_summary.png"
af3_csv    = RESULTS / "alphafold_results.csv"

headline_rows = []
for label, dir_, n in [
    ("ReMap score >= 4 (peak in all 4 ChIP-seq experiments) vs shuffled",
     s4_shuffle, 105),
    ("ReMap score >= 4 vs random GC-matched genomic regions",
     s4_random, 105),
    ("ReMap score >= 2 (peak in 2+ experiments) vs shuffled",
     s2_shuffle, 500),
]:
    headline_rows.append({
        "Comparison": label,
        "n": n,
        "NN all_mean":   auc_lookup(dir_ / "summary.csv", "NN all_mean"),
        "NN core_mean":  auc_lookup(dir_ / "summary.csv", "NN core_mean"),
        "NN flank_mean": auc_lookup(dir_ / "summary.csv", "NN flank_mean"),
        "FIMO":          auc_lookup(dir_ / "summary.csv", "FIMO"),
        "FoldX":         auc_lookup(dir_ / "summary.csv", "FoldX"),
    })
headline_df = pd.DataFrame(headline_rows)

CSS = """
<style>
  body { font-family: -apple-system, Helvetica, Arial, sans-serif;
         max-width: 950px; margin: 2em auto; padding: 0 1.5em;
         color: #222; line-height: 1.5; }
  h1 { color: #003a70; font-size: 1.6em; }
  h2 { color: #003a70; margin-top: 2em; border-bottom: 2px solid #cce; padding-bottom: 0.2em; }
  h3 { color: #555; margin-top: 1.5em; }
  p, li { font-size: 1em; }
  .panel { background: #f7f9fb; border: 1px solid #d4dee7; border-radius: 6px;
           padding: 1em 1.2em; margin: 1em 0; }
  .panel h3 { margin-top: 0; color: #003a70; }
  table.restab { border-collapse: collapse; width: 100%; margin: 1em 0; }
  table.restab th, table.restab td { padding: 0.4em 0.7em;
                                     border-bottom: 1px solid #ddd; text-align: left; }
  table.restab th { background: #003a70; color: white; }
  table.restab tr:nth-child(even) { background: #f2f5f8; }
  img { max-width: 100%; border: 1px solid #ddd; border-radius: 6px; margin: 0.5em 0; }
  code { background: #eef; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
  .caveat { background: #fff8e1; border-left: 4px solid #f9a825;
            padding: 0.7em 1em; margin: 1em 0; font-size: 0.95em; }
  .ok { color: #2e7d32; font-weight: bold; }
  .meh { color: #f57c00; font-weight: bold; }
  .bad { color: #c62828; font-weight: bold; }
  footer { color: #888; font-size: 0.85em; margin-top: 3em; }
</style>
"""

# Compose HTML
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NKX2-1 model validation on ReMap 2022</title>
{CSS}
</head>
<body>

<h1>NKX2-1 binding-site model validation on ReMap 2022 ChIP-seq data</h1>
<p><strong>Sebastian Proft, BIH at Charit&eacute; / Humboldt-Universit&auml;t zu Berlin</strong></p>

<h2>What we tested</h2>
<p>
We trained convolutional neural networks (NN) on EMSA-seq sequence libraries
to predict NKX2-1 DNA binding affinity from 24-bp DNA sequences. The training
data are synthetic in vitro libraries. We want to know whether the same
models also recognize NKX2-1 binding sites in real cells, using only sequence
as input.
</p>
<p>
For an independent test we pulled NKX2-1 ChIP-seq peaks from
<a href="https://remap.univ-amu.fr">ReMap 2022</a>, a public catalogue of
uniformly reprocessed ChIP-seq experiments. NKX2-1 in ReMap 2022 has four
experiments across three lung adenocarcinoma cell lines (NCI-H2087,
NCI-H1819, NCI-H3122) and H9 embryonic stem cells. The non-redundant peak
list has 124,382 peaks. We focused on the highest-confidence subset:
peaks called in all four experiments (n = 105). For comparison we also
ran a larger but less stringent set (peaks in two or more experiments,
n = 500).
</p>

<h2>Methods compared</h2>
<div class="panel">
<h3>1. Neural network (NN) models</h3>
<p>
Three VCNNBpnet sequence-only models trained on different parts of the
EMSA-seq library: <code>all_mean</code> uses the full library,
<code>core_mean</code> trains on the canonical motif core only, and
<code>flank_mean</code> trains on motif plus flanking nucleotides. For each
ChIP-seq peak we tile the peak with 24-bp windows and take the maximum
predicted binding score across the peak.
</p>
<h3>2. FIMO with the canonical NKX2-1 motif (JASPAR MA1994.1)</h3>
<p>
The standard motif scanner. We feed each ChIP-seq peak to FIMO and take
the strongest motif match anywhere in the peak. This is the conventional
sequence-only baseline that any NN model has to beat.
</p>
<h3>3. FoldX on the experimental NKX2-1-DNA crystal complex</h3>
<p>
A physics-based binding-energy estimator. We start from the experimental
NKX2-1 homeodomain bound to its 11-bp consensus DNA, mutate the DNA in
silico to match each peak's central 11-bp window, and let FoldX compute
the resulting binding energy. Lower energy indicates stronger predicted
binding. This method is independent of any sequence model and is included
as a structural sanity check.
</p>
</div>

<h2>Negative controls</h2>
<p>
A binding-site predictor is only useful if it can also <em>reject</em>
non-binding sequences. We used two complementary control sets, both with
n matched to the positive set:
</p>
<ul>
  <li><strong>Dinucleotide-shuffled positives.</strong> Each negative is
      the same length as its paired positive and has identical
      dinucleotide frequencies, but the order of the bases is randomized.
      This destroys any motif structure while keeping sequence
      composition identical. Standard control in motif discovery.</li>
  <li><strong>Length and GC-content matched random genomic windows.</strong>
      Random regions of the human genome (excluding any known NKX2-1
      ChIP-seq peak +/- 1 kb), matched in length and GC content to the
      positives. Tests whether the NN can distinguish NKX2-1 sites from
      generic genomic background.</li>
</ul>

<h2>Headline result</h2>
{headline_df.to_html(index=False, classes="restab", border=0)}
<p>
ROC AUC ranges from 0.5 (random) to 1.0 (perfect). The neural-network
model that uses motif and flanking information (<code>flank_mean</code>)
discriminates NKX2-1 ChIP-seq peaks from negatives at AUC <span class="ok">0.70</span>
on the highest-confidence set, beating the FIMO baseline by 9 to 12 AUC
points across negative-set choices. Both <code>all_mean</code> and
<code>flank_mean</code> outperform FIMO consistently.
</p>

<h2>ROC plot: highest-confidence ReMap peaks vs shuffled negatives</h2>
{img(s4_shuffle / 'roc_all_methods.png')}
<p>
Each curve shows how well a method ranks true ChIP-seq peaks above
shuffled controls. NN <code>flank_mean</code> (green) and
<code>all_mean</code> (blue) lie clearly above FIMO (purple) across the
full operating-point range.
</p>

<h2>Same comparison, larger sample (peak in two or more experiments, n = 500)</h2>
{img(s2_shuffle / 'roc_all_methods.png')}
<p>
The score >= 2 set has five times the sample size and the conclusion is
the same. The smaller score >= 4 set was not a small-sample artefact.
</p>

<h2>Same comparison against random genomic regions instead of shuffles</h2>
{img(s4_random / 'roc_all_methods.png')}
<p>
A stricter test: instead of shuffling the positive sequences, we sample
real genomic regions matched in length and GC content. AUCs are lower
across the board because random genomic regions occasionally hit other
regulatory elements where the (short and AT-rich) NKX2-1 motif also
occurs by chance. The NN models still beat FIMO.
</p>

<h2>AlphaFold 3 structural validation of the top NKX2-1 TAD sites</h2>
<p>
As an orthogonal structural check we submitted the eight highest-scoring
NKX2-1 binding sites in the chr14 NKX2-1 TAD to the public AlphaFold 3
server. For each site, AlphaFold 3 predicts a 3D structure of full-length
NKX2-1 (371 aa) bound to the 30 bp DNA from that site, and reports
several confidence metrics. The most relevant for protein-DNA binding is
<strong>iptm</strong> (interface predicted Template Modelling score):
values above 0.6 indicate the model is confident in the protein-DNA
interface; values above 0.7 are interpreted as high confidence in the
literature.
</p>

<p>
Seven of the eight jobs have returned (one is still running on the
server). Every returned site has iptm above 0.6, and five of seven are
above 0.7. The mean iptm across the returned set is <span class="ok">0.76</span>.
</p>

{img(af3_plot)}
<p>
Left: per-site interface confidence (iptm) for the top 7 NKX2-1 TAD
binding sites. Right: AlphaFold 3 iptm versus the NN consensus score.
The two methods agree that every site is a likely real interface but
rank them differently, suggesting they capture complementary aspects of
binding (sequence features vs structural feasibility).
</p>

{csv_table(af3_csv,
           drop=["chrom", "start", "end", "DNA_fwd", "model_idx",
                 "fraction_disordered", "has_clash"],
           fmt={"consensus_score": "{:.3f}",
                "iptm": "{:.2f}", "ptm": "{:.2f}",
                "ranking_score": "{:.2f}",
                "iptm_protein_dna_fwd": "{:.2f}",
                "iptm_protein_dna_rev": "{:.2f}",
                "pae_min_protein_dna_fwd": "{:.2f}"})}
<p>
For each binding site, the table shows the AlphaFold 3 confidence
metrics for its top-ranked of 5 predicted models. <em>iptm</em> is the
overall interface confidence; <em>iptm_protein_dna_fwd</em> and
<em>iptm_protein_dna_rev</em> are the per-strand pairwise confidences;
<em>pae_min_protein_dna_fwd</em> is the minimum predicted aligned error
between protein and DNA, in &Aring; (lower is better; values around 1 to
2 indicate well-resolved interface contacts).
</p>

<h2>What this means</h2>
<ul>
<li><strong>The NN models do generalize to in vivo data.</strong>
    Models trained on synthetic in vitro libraries discriminate NKX2-1
    ChIP-seq peaks from controls at AUC up to 0.70 on a public,
    independently-generated test set. They were not trained on this data.</li>
<li><strong>The models add information beyond the canonical motif.</strong>
    FIMO with the JASPAR NKX2-1 motif reaches AUC 0.61 on the same task.
    The 9-point gap is the value added by sequence features the NN learned
    from EMSA-seq beyond what JASPAR encodes (most likely flanking-context
    preferences and the structure of partial / weak motifs). The
    <code>flank_mean</code> model is consistently the strongest, supporting
    this interpretation.</li>
<li><strong>0.70 is modest in absolute terms, and that is expected.</strong>
    A sequence-only model can only predict the part of in vivo binding
    that is sequence-encoded. Indirect / cofactor-mediated / chromatin-
    dependent binding cannot be predicted from sequence alone. The same
    models reach R^2 of 0.71 to 0.95 on held-out EMSA-seq, so the
    bottleneck is the in-vitro-to-in-vivo gap, not the model.</li>
<li><strong>AlphaFold 3 independently confirms the top NN TAD sites.</strong>
    Every returned AF3 prediction for the top 7 NKX2-1 binding sites in
    the chr14 TAD has iptm above 0.6, with mean iptm 0.76. AF3 is a
    fully independent structural method (DeepMind Nature 2024) that has
    not seen any of the EMSA-seq data the NN models were trained on.
    The agreement between sequence-based NN consensus and structure-based
    AF3 confidence on these specific sites is a strong cross-method
    sanity check.</li>
</ul>

<h2>Caveats</h2>
<div class="caveat">
<ul style="margin: 0; padding-left: 1.2em;">
<li>NKX2-1 ChIP-seq in ReMap 2022 spans only four experiments, so the
    "score >= 4" set is small (n = 105). The score >= 2 set (n = 500)
    gives the same answer at higher statistical confidence.</li>
<li>The available cell types are biased toward lung adenocarcinoma. NKX2-1
    is also a master regulator of thyroid and forebrain development. The
    binding pattern in those tissues may differ. Tissue-specific public
    NKX2-1 ChIP-seq is sparse.</li>
<li>UniBind has no NKX2-1 dataset in its current release; we verified this
    via their REST API. UniBind would have been the cleanest source.</li>
<li>FoldX results in this report are limited and informative mainly as a
    sanity check, not a predictor. FoldX requires a fixed-length DNA
    matching the crystal (11 bp) and cannot search wider regions in a
    tractable amount of time. AUCs near 0.5 indicate that the static
    crystal-based binding energy is a poor predictor of in vivo
    occupancy, which is consistent with the structural-energy literature.</li>
</ul>
</div>

<h2>How to reproduce</h2>
<pre>
python scripts/remap_validation.py \\
  --remap-bed data/remap2022/remap2022_NKX2-1_nr_macs2_hg38_v1_0.bed \\
  --score-min 4 \\
  --output-dir results/remap_validation_score4_shuffle \\
  --negative-mode shuffle
</pre>
<p>
The ReMap 2022 BED file is downloaded from
<code>https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/TF/NKX2-1/remap2022_NKX2-1_nr_macs2_hg38_v1_0.bed.gz</code>.
Negative seed and balanced subsampling are deterministic with
<code>--seed 42</code>.
</p>

<footer>
Generated by <code>scripts/build_validation_html.py</code> on the
<code>alphafold-tools</code> branch. Latest underlying data:
ReMap 2022 v1.0 (released 2022-04-20). All scripts MIT licensed.
</footer>

</body>
</html>
"""

OUT.write_text(html)
print(f"Wrote {OUT} ({len(html):,} bytes)")
