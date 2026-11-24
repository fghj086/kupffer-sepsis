#!/usr/bin/env bash
# Public files read by stage 05_proteomics (protein layer).
#  PXD062961  whole liver, CLP 24 h vs sham, 4 + 4 mice, DIA-NN 1.9.2 output (PRIDE)
#  PXD021657  Qie et al. 2022 Nat Commun 13:7389 — processed copy-number tables published as
#             Supplementary Data 9 (in vivo LPS: Kupffer cells vs liver-recruited macrophages, 3 + 3)
#             and Supplementary Data 11 (cultured Kupffer cells, LPS vs PBS 6 h, 3 + 3 per genotype)
#  MGI        marker report (symbol history), for harmonising withdrawn gene symbols
# Raw MS files are not downloaded. Existing files are kept.
# Writes results/05_proteomics/raw_checksums.tsv (bytes, md5 and sha256 of every file).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; RAW="$ROOT/data/raw"
OUT="$ROOT/results/05_proteomics"; mkdir -p "$OUT"
fetch () { local url="$1" out="$2"; mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then echo "[skip] $(basename "$out")"; return; fi
  echo "[get ] $(basename "$out")"; wget -q --tries=5 --timeout=300 -O "$out.part" "$url"; mv "$out.part" "$out"; }

PRIDE="https://ftp.pride.ebi.ac.uk/pride/data/archive"
D1="$RAW/05_PXD062961"
fetch "$PRIDE/2025/09/PXD062961/Experimental_design_forPRIDE.xlsx" "$D1/Experimental_design_forPRIDE.xlsx"
fetch "$PRIDE/2025/09/PXD062961/output.zip" "$D1/output.zip"
# PRIDE lists output.zip at 367699698 bytes; a shorter file is an interrupted transfer
if [ "$(stat -L -c %s "$D1/output.zip")" != "367699698" ]; then
  echo "output.zip incomplete; removing, rerun this script" >&2; rm -f "$D1/output.zip"; exit 1; fi

D2="$RAW/05_PXD021657"
ESM="https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-35095-7/MediaObjects"
fetch "$ESM/41467_2022_35095_MOESM3_ESM.pdf"   "$D2/Qie2022_description_of_supplementary_files.pdf"
fetch "$ESM/41467_2022_35095_MOESM12_ESM.xlsx" "$D2/Qie2022_SupplementaryData9_invivoLPS_resident_vs_recruited.xlsx"
fetch "$ESM/41467_2022_35095_MOESM14_ESM.xlsx" "$D2/Qie2022_SupplementaryData11_KC_LPS_vs_PBS.xlsx"
fetch "$PRIDE/2022/10/PXD021657/PXD021657_community_annotated.sdrf.tsv" "$D2/PXD021657_community_annotated.sdrf.tsv"

# MGI marker list: maps withdrawn mouse symbols (used in the 2022 tables, e.g. Irg1) to current ones
D3="$RAW/05_MGI"
fetch "https://www.informatics.jax.org/downloads/reports/MRK_List1.rpt" "$D3/MRK_List1.rpt"

printf "dataset\tfile\tbytes\tmd5\tsha256\n" > "$OUT/raw_checksums.tsv"
for D in "$D1" "$D2" "$D3"; do
  for f in "$D"/*; do
    [ -f "$f" ] || continue
    case "$f" in *.part) continue;; esac
    printf "%s\t%s\t%s\t%s\t%s\n" "$(basename "$D" | sed 's/^05_//')" "$(basename "$f")" "$(stat -L -c %s "$f")" \
      "$(md5sum < "$f" | cut -d' ' -f1)" "$(sha256sum < "$f" | cut -d' ' -f1)" >> "$OUT/raw_checksums.tsv"
  done
done
echo "checksums -> $OUT/raw_checksums.tsv"
