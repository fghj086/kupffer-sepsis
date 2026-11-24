# Septic liver macrophages: stimulus-response attribution

Code for a study of which stimuli the response of septic liver macrophages resembles and how resident
Kupffer cells differ from recruited monocyte-derived macrophages. Responses of sorted liver cells, single
cells, liver time courses, spatial sections, proteomes and human monocytes and blood after sepsis are
compared with 41 reference responses of macrophages and liver to defined stimuli (microbial ligands,
disulfide and fully reduced HMGB1, cytokines, tolerance, glucocorticoid, haem and apoptotic-cell
stimuli) by non-negative regression. The resolution of HMGB1-like versus LPS-like components is measured
with held-out replicates, references from other laboratories and simulated mixtures at in vivo
mouse-to-mouse noise. Within septic livers, Kupffer cells and recruited macrophages are compared on
identity and scavenger programs beyond the shared inflammatory response. All data are public; every
dataset is processed and scored on its own, and only effect directions and sizes are compared across
studies.

## Public datasets

| Accession | Data type | Model / design | Used for |
|---|---|---|---|
| GSE200210 | Bulk RNA-seq | Mouse BMDM: PBS, fully reduced HMGB1, disulfide HMGB1, LPS, LPS + IFN-γ, 24 h, 3 per arm | HMGB1 versus LPS response; stimulus references |
| GSE232827 | Bulk RNA-seq | BMDM: TNF, IFN-β, IFN-γ, 6 h | Stimulus references |
| GSE315641 | Bulk RNA-seq | BMDM: IFN-γ, IL-1β, IL-6, IL-10, 24 h | Stimulus references |
| GSE72964 | Bulk RNA-seq | BMDM: Kdo2-lipid A ± IFN-γ, Pam3CSK4, poly(I:C), tissue homogenate | Stimulus references |
| GSE221734 | Bulk RNA-seq | BMDM: hemin, 12 h | Stimulus references |
| GSE193577 | Bulk RNA-seq | BMDM: apoptotic hepatocyte medium, 24 h | Stimulus references |
| GSE240633 | Bulk RNA-seq | RAW264.7: CpG, LPS, poly(I:C), 14 h | Stimulus references |
| GSE300953 | Bulk RNA-seq | Cultured primary Kupffer cells: LPS, LPS after tolerance | Stimulus references |
| GSE157518 | Bulk RNA-seq | BMDM: tolerised vs naive LPS | Stimulus references |
| GSE306502 | Bulk RNA-seq | BMDM: LPS + IFN-γ, dexamethasone | Stimulus references |
| GSE328774 | Bulk RNA-seq | Sorted Kupffer cells, CLP vs sham, 2 h (4 vs 4) | Early Kupffer cell response |
| GSE311736 | Bulk RNA-seq | Sorted hepatocytes, Kupffer, stellate, endothelial and CD45+ non-Kupffer cells, CLP vs sham, 8 h | Cell-type responses |
| GSE303231 | Bulk RNA-seq | Isolated hepatocytes, faecal-slurry peritonitis, both sexes | Hepatocyte response |
| GSE279167 | scRNA-seq | Whole liver, sham vs CLP 24 h, one pooled library each | Liver cell states (effect sizes only) |
| GSE290684 | scRNA-seq | Whole liver, CLP 24 h with vehicle or TPPU, and sham (12 libraries) | Kupffer cells vs recruited macrophages |
| GSE224127 | Bulk RNA-seq | Whole liver: LPS and CLP time courses, cytokine injections, blockade and knockouts | Time courses, cytokine references |
| GSE248904 | Spatial transcriptomics (Array-seq) | Whole-mouse sections, LPS 24 h, 2 vs 2 | Zonal and myeloid neighbourhoods (effect sizes only) |
| PXD062961 | Proteomics (DIA) | Whole liver, CLP vs sham, 24 h | Protein-level response |
| PXD021657 | Proteomics | Kupffer cells (LPS 6 h); Kupffer vs recruited macrophages after LPS | Macrophage protein responses |
| GSE141864 | Microarray | Post-mortem organs, meningococcal septic shock vs non-infectious death | Human septic liver and spleen |
| GSE233483 | AmpliSeq | iPSC liver organoids with Kupffer-like cells: LPS + IFN-γ and recovery | Human liver model |
| GSE22248, GSE15219 | Microarray | Human PBMC and monocytes, endotoxin tolerance | Human tolerance references |
| GSE254271, GSE162834 | Microarray / RNA-seq | Human monocytes stimulated with HMGB1 | Human HMGB1 references |
| GSE78162 | RT-qPCR panel | Human CSF1 macrophages: LPS, HMGB1 | Human LPS vs HMGB1 genes |
| GSE46955 | Microarray | Blood monocytes, Gram-negative sepsis, recovery and healthy donors, ± ex vivo LPS | Human sepsis monocytes |
| GSE252331 | scRNA-seq | PBMC of sepsis patients and healthy donors | Human blood monocytes, per donor |
| E-MTAB-5273, E-MTAB-4421 | Microarray (ArrayExpress) | GAinS sepsis leukocytes, 28-day outcome | Human blood time course and mortality |
| GSE65682 | Microarray | MARS ICU sepsis whole blood, 28-day outcome | Human blood mortality |
| GSE95233, GSE57065 | Microarray | Septic shock whole blood, repeated samples, healthy controls | Human blood time course |
| GSE222393 (with GSE185263) | Bulk RNA-seq | Sepsis whole blood, admission and later days; healthy donors | Human blood time course and mortality |
| GSE60424 | Bulk RNA-seq | Immune cells sorted from whole blood | Deconvolution reference |

Annotation resources downloaded by the same scripts: Ensembl 102 mouse genes, MSigDB 2024.1.Mm
collections, GENCODE vM30, MGI mouse–human homology, HGNC complete set, the endotoxin-tolerance and
inflammatory gene lists of Pena et al. 2014 (EBioMedicine, Supplementary Tables 1–2), and GEO platform
annotations (GPL570, GPL10558, GPL13667, GPL17586, GPL6883, GPL6255, GPL6104). Only the gene feature
table of one GSE279452 sample is read, for Ensembl-to-symbol conversion of GSE222393 and GSE60424.

## Environment

```bash
conda env create -f environment.yml
conda activate septic-liver-macrophage
```

## Running

All commands are run from the repository root. Each stage reads its own `config/<stage>.yaml`.

1. Download the public data (into `data/raw/` and `data/human/raw/`):

   ```bash
   bash scripts/stages/00_download_01.sh
   bash scripts/stages/00_download_02.sh
   bash scripts/stages/00_download_03.sh
   bash scripts/stages/00_download_04.sh
   bash scripts/stages/00_download_05.sh
   bash scripts/stages/00_download_06.sh
   bash scripts/stages/00_download_07.sh
   bash scripts/stages/00_download_human.sh
   ```

2. Run the analysis stages in numeric order:

   ```bash
   python scripts/stages/00a_blood_cohorts.py
   python scripts/stages/00b_blood_deconvolution.py
   python scripts/stages/01_perturbation.py
   python scripts/stages/02_sorted_bulk.py
   python scripts/stages/03_single_cell_process.py
   python scripts/stages/03_single_cell_analyse.py
   python scripts/stages/03_single_cell_independent.py
   python scripts/stages/04_timecourse.py
   python scripts/stages/04_spatial.py
   python scripts/stages/05_proteomics.py
   python scripts/stages/06_attribution.py
   python scripts/stages/07_human.py
   python scripts/stages/08_injury_link.py
   python scripts/stages/09_identifiability.py
   python scripts/stages/10_sepsis_specificity.py
   ```

   Stages 00a and 00b prepare the human blood cohorts and their deconvolved leukocyte proportions,
   which stage 07 reads.

3. Draw the figures (they read only the tables under `results/`):

   ```bash
   for f in scripts/figures/Fig*.py scripts/figures/SuppFig*.py; do python "$f"; done
   ```

## Output layout

- `data/raw/`, `data/human/raw/` — downloaded public files; `data/processed/`, `data/human/processed/` —
  intermediate matrices and caches.
- `results/<stage>/` — tab-separated result tables of each stage, with `PROVENANCE.json` (input and output
  checksums, seed, configuration hash and software versions) and, for download scripts,
  `raw_checksums.tsv`.
- `figures/` — figure panels (PDF and PNG); `figures/source_data/` — the table behind each panel.

Seeds are fixed in each configuration file (`seed: 20261124`).
