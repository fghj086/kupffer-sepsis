#!/usr/bin/env bash
# Public human files read by stage 07_human (human relevance of the mouse findings).
#  GSE141864  post-mortem organs, meningococcal septic shock (n=5) vs non-infectious death (n=2);
#             processed FFPE log2 table (heart, lung, kidney, liver, spleen), Affymetrix HTA 2.0
#  GSE22248   human PBMC, untreated / LPS / LPS then LPS (endotoxin tolerance), 4 donors, paired
#  GSE15219   human monocytes, in vitro endotoxin tolerance
#  GSE254271  human CD14+ monocytes, GM-CSF vs GM-CSF + HMGB1 (72 h), 4 donors (CEL files)
#  GSE162834  human monocytes, control vs HMGB1 (+/- DMOG), one donor, triplicate cultures
#  GSE46955  blood monocytes, Gram-negative sepsis / recovery / healthy, basal and ex vivo LPS 3 h
#  GSE78162  CSF1-cultured human CD14+ cells, control / LPS / HMGB1 / nab-paclitaxel 96 h (RT-qPCR panel)
#  GSE233483 iPSC liver organoids with Kupffer-like cells: control / LPS+IFN-g / recovery days 1-3 (n=2)
#  MGI HMD_HumanPhenotype.rpt  mouse-human orthologues (MGI homology report)
# Human sepsis blood cohorts (GAinS, GSE95233, GSE57065, GSE222393, MARS) and GSE252331 single-cell
# libraries are downloaded by scripts/stages/00_download_human.sh (see config/07_human.yaml).
# GEO HTTPS can refuse automated clients, so everything comes from the NCBI FTP mirror.
# Existing files are kept. Writes results/07_human/raw_checksums.tsv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; RAW="$ROOT/data/raw"
OUT="$ROOT/results/07_human"; mkdir -p "$OUT"
FTP="ftp://ftp.ncbi.nlm.nih.gov/geo/series"

fetch () { local url="$1" out="$2"; mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then
    case "$out" in
      *.gz) if ! gzip -t "$out" 2>/dev/null; then echo "[bad ] $(basename "$out")"; rm -f "$out"; fi;;
      *.xlsx|*.zip) if ! unzip -tq "$out" >/dev/null 2>&1; then echo "[bad ] $(basename "$out")"; rm -f "$out"; fi;;
    esac
  fi
  if [ -s "$out" ]; then echo "[skip] $(basename "$out")"; return; fi
  echo "[get ] $(basename "$out")"
  local k
  for k in 1 2 3 4 5 6; do   # the FTP mirror drops connections; retry the whole transfer
    local u="$url"   # even attempts use the HTTPS view of the same NCBI mirror
    if [ $((k % 2)) -eq 0 ]; then u="${url/ftp:\/\/ftp.ncbi.nlm.nih.gov/https://ftp.ncbi.nlm.nih.gov}"; fi
    if curl -s --fail --retry 3 --retry-delay 5 --max-time 1800 -o "$out.part" "$u"; then
      case "$out" in
        *.gz)  gzip -t "$out.part" 2>/dev/null && { mv "$out.part" "$out"; return; } ;;
        *.tar) tar -tf "$out.part" >/dev/null 2>&1 && { mv "$out.part" "$out"; return; } ;;
        *.xlsx|*.zip) unzip -tq "$out.part" >/dev/null 2>&1 && { mv "$out.part" "$out"; return; } ;;
        *)     mv "$out.part" "$out"; return ;;
      esac
      echo "[bad ] $(basename "$out"), retrying" >&2
    fi
    sleep $((k * 5))
  done
  echo "failed: $url" >&2; return 1; }

geo () { # geo GSE relative_path_under_series
  local g="$1" rel="$2"; local stub="${g:0:${#g}-3}nnn"
  fetch "$FTP/$stub/$g/$rel" "$RAW/07_$g/$(basename "$rel")"; }

geo GSE141864 suppl/GSE141864_Processed_datafile_Heart_Lungs_Kidneys_Liver_Spleen_FFPE_filt_5_log2.xlsx
geo GSE141864 matrix/GSE141864_series_matrix.txt.gz
geo GSE22248  matrix/GSE22248_series_matrix.txt.gz
geo GSE15219  matrix/GSE15219_series_matrix.txt.gz
geo GSE15219  suppl/GSE15219_raw_data.txt.gz
geo GSE254271 matrix/GSE254271_series_matrix.txt.gz
geo GSE162834 suppl/GSE162834_rawCounts.tsv.gz
geo GSE162834 suppl/GSE162834_sample_metadata.tsv.gz
geo GSE162834 matrix/GSE162834_series_matrix.txt.gz
geo GSE46955  matrix/GSE46955_series_matrix.txt.gz
geo GSE78162  suppl/GSE78162_non-normalized.txt.gz
geo GSE78162  suppl/GSE78162_Fold-change.txt.gz
geo GSE78162  matrix/GSE78162_series_matrix.txt.gz
geo GSE233483 suppl/GSE233483_Combined_processed_data_KuLOs.txt.gz
geo GSE233483 matrix/GSE233483_series_matrix.txt.gz
# MGI vertebrate homology classes (mouse-human), used for one-to-one orthologues; the
# HMD_HumanPhenotype report is kept as a cross-check of the same MGI homology data
fetch "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt" "$RAW/07_MGI/HOM_MouseHumanSequence.rpt"
fetch "https://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt" "$RAW/07_MGI/HMD_HumanPhenotype.rpt"
# platform annotation tables (GEO full data tables; GPL17586 = Affymetrix HTA 2.0)
for gpl in GPL17586 GPL6883 GPL6255 GPL6104; do
  o="$RAW/07_platforms/${gpl}_data.txt.gz"
  if [ ! -s "$o" ] || ! gzip -t "$o" 2>/dev/null; then
    mkdir -p "$RAW/07_platforms"; echo "[get ] $gpl"
    curl -s --fail --retry 5 --max-time 1800 "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=$gpl&targ=self&form=text&view=data" \
      | gzip -c > "$o.part" && mv "$o.part" "$o"
  else echo "[skip] $gpl"; fi
done
# HGNC complete set: previous / alias symbols, to update symbols in older gene lists and arrays
fetch "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt" "$RAW/07_HGNC/hgnc_complete_set.txt"
# Pena et al. 2014 EBioMedicine (PMC4326653) supplementary material: Supplementary Table 1
# (endotoxin-tolerance signature) and Table 2 (inflammatory signature); mmc1.pdf md5 declared by
# the article XML is 19f27445e15769e52377d5bf09cb59d0
fetch "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4326653/supplementaryFiles" "$RAW/07_Pena2014/supp.zip"
( cd "$RAW/07_Pena2014" && unzip -oq supp.zip mmc1.pdf && \
  [ "$(md5sum < mmc1.pdf | cut -d' ' -f1)" = "19f27445e15769e52377d5bf09cb59d0" ] && \
  pdftotext -layout mmc1.pdf mmc1.txt ) || { echo "Pena 2014 supplement failed" >&2; exit 1; }

printf "dataset\tfile\tbytes\tmd5\tsha256\n" > "$OUT/raw_checksums.tsv"
for D in "$RAW"/07_*; do
  for f in "$D"/*; do
    [ -f "$f" ] || continue
    case "$f" in *.part) continue;; esac
    printf "%s\t%s\t%s\t%s\t%s\n" "$(basename "$D" | sed 's/^07_//')" "$(basename "$f")" "$(stat -L -c %s "$f")" \
      "$(md5sum < "$f" | cut -d' ' -f1)" "$(sha256sum < "$f" | cut -d' ' -f1)" >> "$OUT/raw_checksums.tsv"
  done
done
echo "checksums -> $OUT/raw_checksums.tsv"
