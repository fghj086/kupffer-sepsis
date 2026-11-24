#!/usr/bin/env bash
# Stage 04 downloads: GSE224127 count files, metadata and cytokine-blockade counts; GSE248904 spatial h5ad.
# Idempotent; records MD5 checksums next to the files.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXTRA="$ROOT/data/processed/04_gse224127_extra"
SPAT="$ROOT/data/raw/GSE248904"
mkdir -p "$EXTRA" "$SPAT"
B1="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE224nnn/GSE224127/suppl"
B2="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE248nnn/GSE248904/suppl"
fetch() {  # url dest
  if [ ! -s "$2" ]; then
    curl -fsSL --retry 5 --retry-all-errors --retry-delay 10 -o "$2.part" "$1" \
      || curl -fsS --retry 5 -o "$2.part" "${1/https:/ftp:}"
    mv "$2.part" "$2"
  fi
  gzip -t "$2"
}
# count files 1-8 and 11 and the two series matrices (data/raw/GSE224127)
RAW4="$ROOT/data/raw/GSE224127"; mkdir -p "$RAW4"
B0="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE224nnn/GSE224127"
for f in GSE224127-GPL21626_series_matrix.txt.gz GSE224127-GPL30172_series_matrix.txt.gz; do
  fetch "$B0/matrix/$f" "$RAW4/$f"
done
for f in 1_counts_LPS_timecourse 2_counts_CLP_timecourse 3_counts_Cytokine_combo 4_counts_Cytokine_combo2          5_counts_LPS_Plasma 6_counts_LPS_KO 7_counts_LPS_KO2 8_counts_CLP_timecourse2 11_counts_CLP_antiTNF; do
  fetch "$B0/suppl/GSE224127_${f}.txt.gz" "$RAW4/GSE224127_${f}.txt.gz"
done
for f in GSE224127_metadata.txt.gz GSE224127_9_counts_CLP_CytoBlock.txt.gz GSE224127_10_counts_CLP_CytoBlock2.txt.gz; do
  fetch "$B1/$f" "$EXTRA/$f"
done
fetch "$B2/GSE248904_All_Samples.h5ad.gz" "$SPAT/GSE248904_All_Samples.h5ad.gz"
if [ ! -s "$SPAT/GSE248904_All_Samples.h5ad" ]; then
  gunzip -c "$SPAT/GSE248904_All_Samples.h5ad.gz" > "$SPAT/GSE248904_All_Samples.h5ad"
fi
(cd "$EXTRA" && md5sum GSE224127_*.txt.gz > MD5SUMS.txt)
(cd "$RAW4" && md5sum GSE224127*.txt.gz > MD5SUMS.txt)
(cd "$SPAT" && md5sum GSE248904_All_Samples.h5ad.gz GSE248904_All_Samples.h5ad > MD5SUMS.txt)
echo "stage 04 downloads complete"
