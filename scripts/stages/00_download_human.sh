#!/usr/bin/env bash
# Public human blood files read by stages 00a_blood_cohorts, 00b_blood_deconvolution and 07_human.
#  E-MTAB-5273 / E-MTAB-4421  GAinS sepsis leukocyte arrays with 28-day outcome (ArrayExpress)
#  GSE65682   MARS ICU sepsis whole blood, 28-day outcome
#  GSE95233   septic shock whole blood, day 1 and day 2-3, healthy controls
#  GSE57065   septic shock whole blood, 0 / 24 / 48 h, healthy controls
#  GSE222393  sepsis whole-blood RNA-seq, admission and later days; earlier admission samples and
#             healthy donors of the same study in GSE185263
#  GSE252331  PBMC single-cell RNA-seq, sepsis and healthy donors (gene expression matrices)
#  GSE60424   RNA-seq of immune cells sorted from whole blood (deconvolution reference)
#  GSE279452  one 10x feature table (Ensembl gene -> symbol, same GENCODE annotation as GSE222393)
#  GPL570 / GPL10558 / GPL13667 platform annotations
# Existing files are kept. Writes results/00a_blood_cohorts/raw_checksums.tsv (sha256 of every file).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; RAW="$ROOT/data/human/raw"; mkdir -p "$RAW"
OUT="$ROOT/results/00a_blood_cohorts"; mkdir -p "$OUT"
AE="https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-"
GEO="https://ftp.ncbi.nlm.nih.gov/geo"
fetch () { local url="$1" out="$2"; mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then echo "[skip] $(basename "$out")"; return; fi
  echo "[get ] $(basename "$out")"; wget -q --tries=3 --timeout=120 -O "$out.part" "$url"; mv "$out.part" "$out"; }

# GAinS (ArrayExpress)
for a in 5273:Burnham_sepsis_discovery_normalised_231.txt 4421:Davenport_sepsis_Jan2016_normalised_265.txt; do
  n=${a%%:*}; f=${a#*:}
  fetch "$AE/${n: -3}/E-MTAB-$n/Files/E-MTAB-$n.sdrf.txt" "$RAW/E-MTAB-$n/E-MTAB-$n.sdrf.txt"
  fetch "$AE/${n: -3}/E-MTAB-$n/Files/$f" "$RAW/E-MTAB-$n/$f"
done

# GEO series matrices
for G in GSE57065 GSE95233; do
  fetch "$GEO/series/${G:0:5}nnn/$G/matrix/${G}_series_matrix.txt.gz" "$RAW/$G/${G}_series_matrix.txt.gz"
done
fetch "$GEO/series/GSE65nnn/GSE65682/matrix/GSE65682_series_matrix.txt.gz" "$RAW/GSE65682/GSE65682_series_matrix.txt.gz"
# GSE222393 counts, and earlier admission samples of the same patients (and healthy donors) in GSE185263
fetch "$GEO/series/GSE222nnn/GSE222393/matrix/GSE222393_series_matrix.txt.gz" "$RAW/GSE222393/GSE222393_series_matrix.txt.gz"
fetch "$GEO/series/GSE222nnn/GSE222393/suppl/GSE222393_RAW.tar" "$RAW/GSE222393/GSE222393_RAW.tar"
fetch "$GEO/series/GSE185nnn/GSE185263/matrix/GSE185263_series_matrix.txt.gz" "$RAW/GSE222393/GSE185263_series_matrix.txt.gz"
fetch "$GEO/series/GSE185nnn/GSE185263/suppl/GSE185263_raw_counts.csv.gz" "$RAW/GSE222393/GSE185263_raw_counts.csv.gz"
# GSE279452: only the 10x feature table of its first sample is read (Ensembl -> symbol)
fetch "$GEO/samples/GSM8571nnn/GSM8571082/suppl/GSM8571082_Abd-PS04.features.tsv.gz" \
  "$RAW/GSE279452/GSE279452_features.tsv.gz"

# GSE252331: single-cell data of circulating immune cells after sepsis (gene expression matrices)
mkdir -p "$RAW/GSE252331"
fetch "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE252331&targ=gsm&form=text&view=brief" "$RAW/GSE252331/gsm_brief.txt"
grep -o "ftp://[^ ]*quants_mat[^ ]*" "$RAW/GSE252331/gsm_brief.txt" | tr -d '\r' | sed 's#^ftp:#https:#' | grep -v "_FB_" \
  | while read -r u; do fetch "$u" "$RAW/GSE252331/$(basename "$u")"; sleep 1; done

# GSE60424: RNA-seq of immune cells sorted from whole blood (healthy donors, sepsis and others);
# reference profiles for deconvolution (stage 00b)
fetch "$GEO/series/GSE60nnn/GSE60424/suppl/GSE60424_GEOSubmit_FC1to11_normalized_counts.txt.gz" \
  "$RAW/reference_sorted/GSE60424_GEOSubmit_FC1to11_normalized_counts.txt.gz"
fetch "$GEO/series/GSE60nnn/GSE60424/matrix/GSE60424_series_matrix.txt.gz" "$RAW/reference_sorted/GSE60424_series_matrix.txt.gz"

# platform annotations
fetch "$GEO/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz" "$RAW/platforms/GPL570.annot.gz"
fetch "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL10558&targ=self&form=text&view=data" "$RAW/platforms/GPL10558_full.txt"
fetch "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL13667&targ=self&form=text&view=data" "$RAW/platforms/GPL13667.txt"

( cd "$RAW" && find -L . -type f ! -name "*.part" | sort | while read -r f; do
    printf '%s\t%s\n' "${f#./}" "$(sha256sum "$f" | cut -d' ' -f1)"; done ) > "$OUT/raw_checksums.tsv"
echo "[done] $(wc -l < "$OUT/raw_checksums.tsv") files"
