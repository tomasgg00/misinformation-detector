# Data

> **Note:** The raw data files are not committed to this repository due to size and sensitivity. See below for how to obtain and place them.

---

## Datasets

### Labeled Misinformation Dataset (LMD)

A compiled dataset combining multiple publicly available misinformation datasets:

- FakeNewsNet (PolitiFact + GossipCop)
- LIAR dataset
- MultiFC
- Additional social media sources

The combined dataset covers news articles and social media posts across different topics. Labels are binary: `0 = misinformation`, `1 = factual`.

**Note:** Some content in this dataset contains derogatory language and misleading claims about refugees and migrants. This is included solely for research purposes.

### UNHCR Refugee X Dataset (URXD)

250 X (Twitter) posts about refugees and migrants, collected in 2024 and manually labeled by the thesis authors in consultation with UNHCR staff.

- Labels: `0 = misinformation`, `1 = factual`
- Domain: refugee/migrant-specific social media content
- Purpose: out-of-domain validation set for model evaluation

---

## Dataset Configurations

The LMD is split into 4 configurations for the benchmarking experiments:

| Config | Description |
|--------|-------------|
| `twitter_filtered` | Twitter/X posts only, with quality filtering |
| `twitter_unfiltered` | Twitter/X posts only, no filtering |
| `complete_filtered` | All media types (news + social media), filtered |
| `complete_unfiltered` | All media types, no filtering |

"Filtering" means: removing posts under 10 words, removing near-duplicates, normalizing encoding issues.

---

## File Placement

Place CSV files in `data/raw/` with these exact names:

```
data/raw/
├── twitter_filtered.csv
├── twitter_unfiltered.csv
├── complete_filtered.csv
├── complete_unfiltered.csv
└── urxd.csv               # UNHCR dataset
```

Each CSV should have at minimum a `content` (or `text`) column and a `label` column. The preprocessing script handles column name variations.

---

## Preprocessing

After placing the raw files, run:

```bash
python scripts/run_preprocessing.py --all
```

This generates train/val/test splits in `data/processed/`.
