#!/usr/bin/env bash
# Public files read by stage 02_sorted_bulk (sorted liver cell bulk RNA-seq after polymicrobial sepsis).
# Existing files are kept. Writes results/02_sorted_bulk/raw_checksums.tsv (md5 and sha256 of every file).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; RAW="$ROOT/data/raw"
OUT="$ROOT/results/02_sorted_bulk"; mkdir -p "$OUT"
GEO="ftp://ftp.ncbi.nlm.nih.gov/geo/series"   # HTTPS mirror answers 403 to scripted clients on this host
fetch () { local url="$1" out="$2"; mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then echo "[skip] $(basename "$out")"; return; fi
  echo "[get ] $(basename "$out")"
  for i in 1 2 3 4 5; do   # the NCBI FTP server intermittently refuses directory changes
    curl -sS --max-time 1200 -o "$out.part" "$url" && { mv "$out.part" "$out"; return; }; sleep 20; done
  echo "failed: $url" >&2; return 1; }

# GSE328774: sorted Kupffer cells, CLP 2 h vs sham
# GSE311736: sorted hepatocytes, KC, HSC, LSEC and CD45+ non-KC cells, CLP 8 h vs sham
# GSE303231: isolated hepatocytes, faecal-slurry peritonitis vs control, both sexes
declare -A COUNTS=( [GSE328774]=GSE328774_counts.txt.gz
                    [GSE311736]=GSE311736_counts.txt.gz
                    [GSE303231]=GSE303231_all_samples_counts.tsv.gz )
for G in GSE328774 GSE311736 GSE303231; do
  d="$GEO/${G:0:6}nnn/$G"
  fetch "$d/matrix/${G}_series_matrix.txt.gz" "$RAW/$G/${G}_series_matrix.txt.gz"
  fetch "$d/suppl/${COUNTS[$G]}" "$RAW/$G/${COUNTS[$G]}"
done

# GENCODE vM30 (Ensembl 107, GRCm39): the annotation both sorted-cell series were counted against;
# used for gene symbols and, for GSE303231, transcript -> gene aggregation
fetch "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/gencode.vM30.annotation.gtf.gz" \
  "$RAW/GSE311736/gencode.vM30.annotation.gtf.gz"

printf "dataset\tfile\tbytes\tmd5\tsha256\n" > "$OUT/raw_checksums.tsv"
for G in GSE328774 GSE311736 GSE303231; do
  for f in "$RAW/$G"/*; do
    printf "%s\t%s\t%s\t%s\t%s\n" "$G" "$(basename "$f")" "$(stat -L -c %s "$f")" \
      "$(md5sum < "$f" | cut -d' ' -f1)" "$(sha256sum < "$f" | cut -d' ' -f1)" >> "$OUT/raw_checksums.tsv"
  done
done
echo "checksums -> $OUT/raw_checksums.tsv"
