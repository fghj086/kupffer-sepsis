#!/usr/bin/env bash
# Stage 03 inputs: GSE290684 (liver immune-cell scRNA-seq, CLP + vehicle vs CLP + TPPU).
# and GSE279167 (whole-liver scRNA-seq, sham vs CLP, one pooled library each);
# both datasets are checksummed into results/03_single_cell/raw_checksums.tsv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RAW="$ROOT/data/raw"
OUT="$ROOT/results/03_single_cell"
mkdir -p "$RAW/GSE290684" "$OUT"

GEO=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE290nnn/GSE290684
cd "$RAW/GSE290684"
[ -s GSE290684_series_matrix.txt.gz ] || curl -sS -L -o GSE290684_series_matrix.txt.gz "$GEO/matrix/GSE290684_series_matrix.txt.gz"
[ -s GSE290684_RAW.tar ] || curl -sS -L -o GSE290684_RAW.tar "$GEO/suppl/GSE290684_RAW.tar"
tar -tf GSE290684_RAW.tar > /dev/null          # fails if the archive is truncated
tar -xf GSE290684_RAW.tar --skip-old-files

if [ ! -d "$RAW/GSE279167" ]; then
  mkdir -p "$RAW/GSE279167"; cd "$RAW/GSE279167"
  G2=https://ftp.ncbi.nlm.nih.gov/geo
  curl -sS -L -o GSE279167_series_matrix.txt.gz "$G2/series/GSE279nnn/GSE279167/matrix/GSE279167_series_matrix.txt.gz"
  curl -sS -L -O "$G2/samples/GSM8563nnn/GSM8563084/suppl/GSM8563084_S2_filtered_feature_bc_matrix.h5"
  curl -sS -L -O "$G2/samples/GSM8563nnn/GSM8563085/suppl/GSM8563085_CLP_filtered_feature_bc_matrix.h5"
fi

{
  printf "dataset\tfile\tbytes\tmd5\n"
  for d in GSE279167 GSE290684; do
    for f in "$RAW/$d"/*; do
      [ -f "$f" ] || continue
      printf "%s\t%s\t%s\t%s\n" "$d" "$(basename "$f")" "$(stat -L -c %s "$f")" "$(md5sum "$f" | cut -d' ' -f1)"
    done
  done
} > "$OUT/raw_checksums.tsv"
echo "wrote $OUT/raw_checksums.tsv"
