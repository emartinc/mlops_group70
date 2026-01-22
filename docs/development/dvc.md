# Data Version Control (DVC)

This guide covers using DVC for versioning datasets and models with Google Cloud Storage.

## Overview

DVC (Data Version Control) is Git for data. It:

- **Tracks large files** without bloating Git repos
- **Versions datasets and models** with Git commits
- **Enables collaboration** via cloud storage
- **Ensures reproducibility** with checksums

## Architecture

```
Local Repository                  Google Cloud Storage
┌──────────────────┐              ┌─────────────────────┐
│ Git              │              │ mlops70_bucket      │
│ ├── data/raw.dvc │──────────────│ └── dvcstore/       │
│ ├── models.dvc   │   metadata   │     └── files/md5/  │
│ └── .dvc/config  │              │         ├── 34/...  │
└──────────────────┘              │         ├── f3/...  │
                                  │         └── ...     │
                                  └─────────────────────┘

Local Cache                        
┌──────────────────┐
│ .dvc/cache/      │ (linked to GCS)
│ ├── files/       │
│ └── tmp/         │
└──────────────────┘
```

## Setup

### Initialize DVC

Already done in this project:

```bash
# Check DVC status
uv run dvc status

# View configuration
cat .dvc/config
```

Configuration:
```ini
[core]
    remote = gcs
[remote "gcs"]
    url = gs://mlops70_bucket/dvcstore
```

### Authenticate

#### Option A: gcloud CLI (Recommended for local)

```bash
# First time only
gcloud auth application-default login

# Test access
gsutil ls gs://mlops70_bucket/
```

#### Option B: Service Account (CI/CD)

```bash
# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# Or in .env
echo "GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json" >> .env
```

!!! warning
    Never commit service account keys to Git. They're in `.gitignore`.

## Basic Workflow

### Pull Data

Download datasets and models:

```bash
# Pull everything
uv run dvc pull

# Or via invoke
uv run invoke dvc-pull

# Pull specific file
uv run dvc pull data/raw.dvc
```

**What happens:**
1. Reads `data/raw.dvc` metadata
2. Downloads files from GCS
3. Reconstructs `data/raw/` directory
4. Verifies checksums

### Check Status

```bash
# See what's changed
uv run dvc status

# Or via invoke
uv run invoke dvc-status
```

Output:
```
Data and pipelines are up to date.
```

Or if changes:
```
data/raw.dvc:
    modified:           data/raw/mbti_1.csv
```

### Push Changes

After training a new model:

```bash
# 1. Add to DVC
uv run dvc add models

# 2. Commit metadata to Git
git add models.dvc .gitignore
git commit -m "Train model v2"

# 3. Push data to GCS
uv run dvc push

# Or via invoke
uv run invoke dvc-push

# 4. Push Git changes
git push
```

## File Structure

### DVC Metadata Files

```yaml
# models.dvc
outs:
- md5: f3dc2086c25e37c552d4000a242aee83.dir
  size: 803623819
  nfiles: 2
  hash: md5
  path: models
```

**Fields:**
- `md5`: Content hash (identifies version)
- `size`: Total size in bytes
- `nfiles`: Number of files in directory
- `path`: Local path

### Storage Layout

```
gs://mlops70_bucket/dvcstore/
└── files/
    └── md5/
        ├── 34/           # Hash prefix for organization
        │   └── 6acadd... # Actual file content
        ├── f3/
        │   └── dc2086... # Model checkpoint
        └── ...
```

**Why this structure?**
- **Deduplication**: Identical files share one copy
- **Scalability**: Distributes files across directories
- **Integrity**: Filename IS the hash

## Advanced Usage

### Track New Data

```bash
# Add new dataset
uv run dvc add data/external/new_dataset.csv

# Commit
git add data/external/new_dataset.csv.dvc data/external/.gitignore
git commit -m "Add external dataset"

# Push
uv run dvc push
```

### Checkout Specific Version

```bash
# Switch to older commit
git checkout abc123

# Get data from that version
uv run dvc checkout

# Return to latest
git checkout main
uv run dvc checkout
```

### Compare Versions

```bash
# Show DVC diff between commits
uv run dvc diff HEAD~1

# Output:
# Modified:
#     data/processed.dvc
```

### Remove Tracking

```bash
# Stop tracking with DVC
uv run dvc remove models.dvc

# Start tracking with Git instead
git rm models.dvc
git add models/
```

## Collaboration Workflow

### Team Member Cloning

```bash
# 1. Clone repo
git clone https://github.com/mlopsgroup70/mlops_group70.git
cd mlops_group70

# 2. Install dependencies
uv sync --dev

# 3. Authenticate
gcloud auth application-default login

# 4. Pull data
uv run dvc pull
```

### Updating Data

**Person A:**
```bash
# 1. Modify data
python preprocess.py --new-method

# 2. Track changes
uv run dvc add data/processed

# 3. Commit and push
git add data/processed.dvc
git commit -m "Improve preprocessing"
uv run dvc push
git push
```

**Person B:**
```bash
# 1. Pull Git changes
git pull

# 2. Pull new data
uv run dvc pull

# 3. Verify
ls -lh data/processed/
```

## Integration with Training

### Before Training

```bash
# Ensure latest data
uv run dvc pull
```

### After Training

```bash
# Track new model
uv run dvc add models

# Commit experiment
git add models.dvc
git commit -m "Experiment: lr=5e-5, batch_size=32

Metrics:
- Val F1: 0.85
- Test F1: 0.83"

# Share with team
uv run dvc push
git push
```

### In Docker

```bash
# Pull before building
uv run dvc pull

# Build with latest data
docker compose build train

# Train
docker compose run --rm train

# Track new model
uv run dvc add models
```

## Verification

### Check Data Integrity

```bash
# Verify local files match DVC
uv run dvc status

# Recalculate checksums
uv run dvc status --cloud
```

### Inspect Remote

```bash
# List files in GCS
gsutil ls -r gs://mlops70_bucket/dvcstore/

# Check specific file
gsutil ls gs://mlops70_bucket/dvcstore/files/md5/f3/
```

### Dry Run

```bash
# See what would be pushed
uv run dvc push --dry-run

# See what would be pulled
uv run dvc pull --dry-run
```

## Storage Management

### Cache Management

```bash
# Show cache usage
uv run dvc cache dir

# Clean unused cache
uv run dvc gc --workspace

# Clean all except current
uv run dvc gc --cloud
```

### Fetch vs Pull

```bash
# Fetch: Download to cache only
uv run dvc fetch

# Pull: Fetch + checkout to workspace
uv run dvc pull
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Train Model

on:
  push:
    paths:
      - 'data/**'

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Setup DVC
        run: pip install dvc[gs]
      
      - name: Configure GCS credentials
        env:
          GOOGLE_APPLICATION_CREDENTIALS: ${{ secrets.GCS_KEY }}
        run: echo "$GOOGLE_APPLICATION_CREDENTIALS" > key.json
      
      - name: Pull data
        run: dvc pull
      
      - name: Train
        run: uv run invoke train
      
      - name: Push model
        run: |
          dvc add models
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add models.dvc
          git commit -m "Auto: Update model"
          dvc push
          git push
```

## Troubleshooting

### Authentication fails

```bash
# Re-authenticate
gcloud auth application-default login

# Check credentials
gcloud auth list

# Test bucket access
gsutil ls gs://mlops70_bucket/
```

### Files not found in remote

```bash
# Check what's in remote
uv run dvc status --cloud

# Force push
uv run dvc push --force
```

### Checksum mismatch

```bash
# Recalculate local checksums
uv run dvc status --recalculate

# If corrupted, re-pull
uv run dvc pull --force
```

### Large cache size

```bash
# Show cache size
du -sh .dvc/cache/

# Remove old versions
uv run dvc gc --workspace --cloud
```

## Best Practices

### ✅ Do

- **Pull before training**: `dvc pull` → `train` → `dvc push`
- **Commit meaningful messages**: Include metrics in commit
- **Use .dvcignore**: Ignore temporary files
- **Regular garbage collection**: Clean old versions

### ❌ Don't

- **Don't commit large files to Git**: Use DVC instead
- **Don't share credentials**: Use gcloud CLI or CI secrets
- **Don't skip `dvc push`**: Team won't have your data
- **Don't modify .dvc files manually**: Use DVC commands

## Monitoring

### Track Usage

```bash
# See what's tracked
uv run dvc list . -R --dvc-only

# Show file sizes
uv run dvc status --verbose
```

### GCS Costs

Monitor bucket usage in Google Cloud Console:
- Storage: ~$0.02/GB/month
- Download: ~$0.12/GB
- Operations: Minimal

## Next Steps

- Explore [Docker](docker.md) integration
- Learn about [Testing](testing.md) with DVC
- Read [Configuration](../configuration/hydra.md)
