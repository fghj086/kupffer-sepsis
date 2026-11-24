#!/usr/bin/env bash
# Public files read by stage 06_attribution (stimulus reference panel).
#  GSE232827  BMDM, untreated / IFN-b / IFN-g / TNF / TGF-b, 6 h, 3 each (raw counts)
#  GSE315641  BMDM, 24 h cytokine panel (IFN-g, IL-6, IL-10, IL-1b ...), TMM-normalised counts
#  GSE72964   BMDM, Pam3CSK4 / poly(I:C) / KLA 6 h, KLA+IFN-g 24 h, tissue homogenate 6 h & 24 h (FPKM)
#  GSE221734  BMDM, hemin 12 h vs vehicle, 3 each (per-sample files in RAW.tar)
#  GSE193577  BMDM, apoptotic hepatocyte conditioned medium 24 h vs untreated, 4 each (FPKM)
#  GSE240633  RAW264.7 (Lamtor1-KO reconstituted with WT), CpG-B / poly(I:C) / R848 / cGAMP / LPS 14 h
#  GSE300953  primary Kupffer cells: control / LPS 6 h / LPS-tolerised then LPS 6 h, 2 each (endotoxin tolerance)
#  GSE157518  BMDM: control-Ig then LPS 4 h vs LPS-pre-exposed then LPS 4 h, 3 each (tolerance shift; IgG arms only)
#  GSE306502  BMDM M0 / M1 (LPS 24 h) / M2, vehicle vs dexamethasone 4 h, 4 each per genotype (featureCounts)
#  GSE224127  author metadata (liver cytokine and LPS designs); counts are read from data/raw/GSE224127
# GEO HTTPS can refuse automated clients, so everything comes from the NCBI FTP mirror.
# Existing files are kept. Writes results/06_attribution/raw_checksums.tsv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; RAW="$ROOT/data/raw"
OUT="$ROOT/results/06_attribution"; mkdir -p "$OUT"
FTP="ftp://ftp.ncbi.nlm.nih.gov/geo/series"

fetch () { local url="$1" out="$2"; mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then
    case "$out" in *.gz) if ! gzip -t "$out" 2>/dev/null; then echo "[bad ] $(basename "$out")"; rm -f "$out"; fi;; esac
  fi
  if [ -s "$out" ]; then echo "[skip] $(basename "$out")"; return; fi
  echo "[get ] $(basename "$out")"
  local k
  for k in 1 2 3 4 5 6; do   # the FTP mirror drops connections; retry the whole transfer
    if curl -s --fail --retry 3 --retry-delay 5 --max-time 1800 -o "$out.part" "$url"; then
      # transfers from the mirror occasionally arrive full-length but corrupted: test archives
      case "$out" in
        *.gz)  gzip -t "$out.part" 2>/dev/null && { mv "$out.part" "$out"; return; } ;;
        *.tar) tar -tf "$out.part" >/dev/null 2>&1 && { mv "$out.part" "$out"; return; } ;;
        *)     mv "$out.part" "$out"; return ;;
      esac
      echo "[bad ] $(basename "$out"), retrying" >&2
    fi
    sleep $((k * 5))
  done
  echo "failed: $url" >&2; return 1; }

geo () { # geo GSE relative_path_under_series
  local g="$1" rel="$2"; local stub="${g:0:${#g}-3}nnn"
  fetch "$FTP/$stub/$g/$rel" "$RAW/06_$g/$(basename "$rel")"; }

geo GSE232827 suppl/GSE232827_raw_counts.csv.gz
geo GSE232827 matrix/GSE232827_series_matrix.txt.gz
geo GSE315641 suppl/GSE315641_TMMnorm-allbatches.txt.gz
geo GSE315641 matrix/GSE315641_series_matrix.txt.gz
geo GSE72964  suppl/GSE72964_GeneExpression.txt.gz
for f in GSE72964-GPL13112_series_matrix.txt.gz GSE72964-GPL19057_series_matrix.txt.gz \
         GSE72964-GPL21103_series_matrix.txt.gz; do geo GSE72964 "matrix/$f"; done
geo GSE221734 suppl/GSE221734_RAW.tar
geo GSE221734 matrix/GSE221734_series_matrix.txt.gz
geo GSE193577 suppl/GSE193577_fpkm_sample.txt.gz
geo GSE193577 matrix/GSE193577_series_matrix.txt.gz
geo GSE240633 suppl/GSE240633_R_RNA-seq_GEO_processed.txt.gz
geo GSE240633 matrix/GSE240633_series_matrix.txt.gz
geo GSE300953 suppl/GSE300953_RAW.tar
geo GSE300953 matrix/GSE300953_series_matrix.txt.gz
geo GSE157518 suppl/GSE157518_Tol_Imm_counts.txt.gz
for f in GSE157518-GPL19057_series_matrix.txt.gz GSE157518-GPL18573_series_matrix.txt.gz; do
  geo GSE157518 "matrix/$f" || true; done
# GSE306502: the series RAW.tar on the mirror is truncated after its first member, so the
# wild-type (GR-flox) M0 and M1 libraries are fetched one by one from the sample directories
n=9202272
for rep in 1 2 3 4; do
  for arm in M0Veh M0Dex M1Veh M1Dex M2Veh M2Dex; do
    gsm="GSM$n"; f="${gsm}_dFlox${rep}${arm}_featurecounts_fragments.txt.gz"
    case "$arm" in M0*|M1*) fetch "ftp://ftp.ncbi.nlm.nih.gov/geo/samples/${gsm:0:7}nnn/$gsm/suppl/$f" \
      "$RAW/06_GSE306502/samples/$f";; esac
    n=$((n+1))
  done
done
geo GSE306502 matrix/GSE306502_series_matrix.txt.gz

# GSE224127 author metadata: use the copy placed by stage 04 when present
EXTRA="$ROOT/data/processed/04_gse224127_extra/GSE224127_metadata.txt.gz"
if [ ! -s "$EXTRA" ]; then geo GSE224127 suppl/GSE224127_metadata.txt.gz; fi

# per-sample tables: unpack each RAW.tar into its own samples/ directory
for g in GSE221734 GSE300953; do
  D="$RAW/06_$g"
  if [ -s "$D/${g}_RAW.tar" ] && [ ! -d "$D/samples" ]; then
    mkdir -p "$D/samples"; tar -xf "$D/${g}_RAW.tar" -C "$D/samples"; fi
done

printf "dataset\tfile\tbytes\tmd5\tsha256\n" > "$OUT/raw_checksums.tsv"
for D in "$RAW"/06_*; do
  for f in "$D"/* "$D"/samples/*; do
    [ -f "$f" ] || continue
    case "$f" in *.part) continue;; esac
    printf "%s\t%s\t%s\t%s\t%s\n" "$(basename "$D" | sed 's/^06_//')" "$(basename "$f")" "$(stat -L -c %s "$f")" \
      "$(md5sum < "$f" | cut -d' ' -f1)" "$(sha256sum < "$f" | cut -d' ' -f1)" >> "$OUT/raw_checksums.tsv"
  done
done
echo "checksums -> $OUT/raw_checksums.tsv"
