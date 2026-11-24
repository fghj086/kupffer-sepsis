#!/usr/bin/env bash
# Public files read by stage 01_perturbation (and the gene annotation shared with stages 02 and 06).
#  GSE200210  BMDM: PBS / fully reduced HMGB1 / disulfide HMGB1 / LPS / LPS + IFN-g, 24 h, 3 each (counts)
#  Ensembl 102 mouse gene table (gene id, symbol, biotype), from the GRCm38.102 GTF
#  MSigDB 2024.1.Mm mouse collections: Hallmark, GO biological process, Reactome
# Existing files are kept. Writes results/01_perturbation/raw_checksums.tsv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; RAW="$ROOT/data/raw/GSE200210"; ANN="$RAW/annotation"
OUT="$ROOT/results/01_perturbation"; mkdir -p "$OUT" "$ANN"
fetch () { local url="$1" out="$2"; mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then echo "[skip] $(basename "$out")"; return; fi
  echo "[get ] $(basename "$out")"
  for i in 1 2 3 4 5; do
    curl -sS --fail --max-time 1800 -o "$out.part" "$url" && { mv "$out.part" "$out"; return; }; sleep 20; done
  echo "failed: $url" >&2; return 1; }

GEO="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE200nnn/GSE200210"
fetch "$GEO/suppl/GSE200210_Raw_count_BMDM.txt.gz" "$RAW/GSE200210_Raw_count_BMDM.txt.gz"
fetch "$GEO/matrix/GSE200210_series_matrix.txt.gz" "$RAW/GSE200210_series_matrix.txt.gz"

MSIG="https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Mm"
for f in mh.all m5.go.bp m2.cp.reactome; do
  fetch "$MSIG/$f.v2024.1.Mm.symbols.gmt" "$ANN/$f.v2024.1.Mm.symbols.gmt"
done

GTF="$ANN/Mus_musculus.GRCm38.102.gtf.gz"
fetch "https://ftp.ensembl.org/pub/release-102/gtf/mus_musculus/Mus_musculus.GRCm38.102.gtf.gz" "$GTF"
if [ ! -s "$ANN/ens102_mouse_genes.tsv" ]; then
  zcat "$GTF" | awk -F'\t' 'BEGIN{OFS="\t"; print "ensembl_gene_id","gene_name","gene_biotype"}
    $3=="gene"{id="";nm="";bt=""
      if(match($9,/gene_id "[^"]*"/)) id=substr($9,RSTART+9,RLENGTH-10)
      if(match($9,/gene_name "[^"]*"/)) nm=substr($9,RSTART+11,RLENGTH-12)
      if(match($9,/gene_biotype "[^"]*"/)) bt=substr($9,RSTART+14,RLENGTH-15)
      print id,nm,bt}' > "$ANN/ens102_mouse_genes.tsv"
fi

printf "file\tbytes\tmd5\n" > "$OUT/raw_checksums.tsv"
( cd "$RAW" && find . -type f ! -name "*.part" | sort | while read -r f; do
    printf "%s\t%s\t%s\n" "${f#./}" "$(stat -L -c %s "$f")" "$(md5sum < "$f" | cut -d' ' -f1)"; done ) >> "$OUT/raw_checksums.tsv"
echo "checksums -> $OUT/raw_checksums.tsv"
