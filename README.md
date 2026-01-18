## Quick Start (Minimal Reproducible)

### 1) Install dependencies
```bash
python -m pip install -r requirements.txt
2) Prepare data (not included in repo)
Place XMIDI-format MIDI files under:

data/Data/ (full dataset) or
data/Data_100/ (small subset)
Expected structure: *.mid (or .midi).

3) Install midi2vec node deps (for edgelist generation)
cd external/midi2vec/midi2edgelist
npm install
cd ../../../
4) Run the unified pipeline
python src/runners/run_experiment.py \
  --base configs/base.yaml \
  --dataset configs/datasets/data.yaml \
  --experiment configs/experiments/exp_default.yaml \
  --resume
Outputs are written to artifacts/ and results/.
