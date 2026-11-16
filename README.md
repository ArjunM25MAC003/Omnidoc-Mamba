# OmniDoc-Mamba v3.1

**A research prototype for visually rich document understanding**  
**Author:** Arjun | IIT Jodhpur, M.Tech DCS  
**Primary environment:** Remote Linux cluster, VS Code Remote SSH, NVIDIA A30

OmniDoc-Mamba explores image-based document reading with multi-view visual features, a bidirectional state-space encoder, and a pretrained causal language model. Version 3.1 updates the v2 prototype with corrected masked-patch training, visual-prefix conditioning, real LoRA adapters, improved evaluation, and a synthetic reading pretraining stage.

> **Research status:** This is an experimental prototype, not a production OCR or information-extraction system. The recorded run used a gated-EMA fallback rather than Mamba, and its image-ablation losses were nearly identical for real, blank, and shuffled images. The results therefore do not demonstrate that the model learned to read document images.

## Contents

- [Notebook](#notebook)
- [Highlights](#highlights)
- [Architecture and training](#architecture-and-training)
- [Environment and setup](#environment-and-setup)
- [Run the notebook](#run-the-notebook)
- [Recorded run](#recorded-run)
- [Outputs](#outputs)
- [Known limitations](#known-limitations)

## Notebook

The implementation and runnable walkthrough are in [`omnidoc_mamba_v3_1_college_A30.ipynb`](omnidoc_mamba_v3_1_college_A30.ipynb). It is designed to run from top to bottom in the cluster's `omnidoc` Python environment. An architecture overview is available at [`output/OmniDoc-Mamba_Architecture_and_Flow.png`](../output/OmniDoc-Mamba_Architecture_and_Flow.png).

## Highlights

- Processes each page as four local crops plus one global view. The default crop size is 336 px, producing a 672 px square canvas.
- Projects 16 × 16 image patches into 128-dimensional visual tokens and applies spatial embedding and local graph-style message passing.
- Uses a four-layer bidirectional sequence core. When `mamba_ssm` is available and passes the CUDA preflight, the notebook uses Mamba blocks; otherwise, it explicitly falls back to a gated-EMA implementation that is **not Mamba**.
- Pools visual tokens and prepends them to the decoder through `inputs_embeds`; the decoder is `distilgpt2` by default, with LoRA adapters.
- Includes masked-patch reconstruction, synthetic reading pretraining, FUNSD fine-tuning, free-running generation evaluation, image-use ablations, efficiency measurements, and a local Gradio demo for image/PDF inputs.

## Architecture and training

| Component | Notebook configuration |
|---|---|
| Image views | 4 local views + 1 global view |
| Crop size | 336 px by default (`OMNIDOC_CROP`) |
| Patch size / embedding | 16 px / 128 dimensions |
| Sequence encoder | 4 bidirectional layers; real `mamba_ssm` if available, otherwise gated-EMA fallback |
| Decoder | `distilgpt2` with PEFT LoRA; optional 4-bit mode |
| Dataset | FUNSD (`nielsr/funsd`); optional CORD experiment is disabled by default |
| Training stages | Masked-patch reconstruction (Stage A), synthetic reading (Stage A2), FUNSD supervised fine-tuning (Stage B) |
| Training defaults | 5 Stage A epochs, 15 Stage B epochs, AdamW, learning rate `2e-4`, weight decay `0.01` |

The notebook creates a deterministic development split from FUNSD's 149 training documents (134 train / 15 dev) and reserves the official 50-document test split for final reporting. The task currently trains reading-order text extraction. Its diagnostic bag-of-words token F1 is **not** FUNSD entity-level F1.

## Environment and setup

The documented cluster configuration is:

- Python 3.10.12
- PyTorch 2.1.2 + CUDA 11.8 (`2.1.2+cu118`)
- CUDA-enabled NVIDIA A30 GPU
- A preconfigured `omnidoc` environment with notebook dependencies installed

Connect to the Linux server using VS Code Remote SSH, open this notebook remotely, select the `Python (omnidoc)` kernel, and confirm the notebook's environment check reports CUDA and the A30. Dependencies should be installed in the cluster environment before the run; the notebook intentionally does not install or replace PyTorch/CUDA packages. Core imports include `transformers`, `datasets`, `accelerate`, `peft`, `scikit-learn`, `gradio`, `pymupdf`, `jiwer`, `ipywidgets`, `Pillow`, `matplotlib`, and `seaborn`. The notebook contains a commented optional package-install cell for administrators maintaining that environment.

The compute node must be able to access Hugging Face or have the required dataset and model cached. The default model is `distilgpt2`; keep `OMNIDOC_4BIT=0` for the standard configuration.

## Run the notebook

1. Open the notebook on the remote Linux cluster and select the `omnidoc` kernel.
2. Run cells in order. The notebook checks the environment, loads FUNSD, preprocesses pages, builds the model, trains the stages, and evaluates on the held-out test split.
3. Review the printed SSM implementation label, synthetic CER trend, image ablation, and evaluation protocol before interpreting metrics.
4. The final cell starts a local Gradio app on port 7860. Use VS Code's **Ports** panel to forward port 7860 if you want to open the demo from your workstation.

The notebook stores its runtime data outside this repository under `/scratch/m25mac003/omnidoc_outputs`:

```text
/scratch/m25mac003/omnidoc_outputs/
├── data/       # FUNSD and Hugging Face caches
├── models/     # checkpoints
├── results/    # metrics, CSV files, plots, and execution summary
└── notebooks/  # notebook workspace
```

Optional settings include `OMNIDOC_CROP` for crop size, `OMNIDOC_LLM` for the decoder model name, `OMNIDOC_4BIT=1` to request 4-bit QLoRA when CUDA is available, and `GRADIO_SERVER_PORT` for the demo port.

## Recorded run

The notebook's saved execution summary reports the following run. These are prototype diagnostics, not a claim of competitive OCR or KIE performance.

| Measure | Recorded value |
|---|---:|
| FUNSD documents | 149 train, split into 134 train / 15 dev; 50 held-out test |
| SSM implementation | Gated-EMA fallback (**not Mamba**) |
| Decoder | `distilgpt2` with LoRA adapters |
| Batch size | 2 |
| Best development loss | 4.6946 (epoch 5) |
| Held-out test CER / WER | 0.809 / 0.989 |
| Diagnostic bag-of-words token F1 | 8.93% (not entity-level F1) |
| Visual ablation loss (real / blank / shuffled) | 4.6946 / 4.6945 / 4.6945 |

The near-identical real, blank, and shuffled image losses are a warning that this run did not establish meaningful visual grounding. The benchmark CSV also labels the sequence core as a Python-loop EMA fallback; those timings are not Mamba performance and should not be presented as a Mamba-versus-Transformer result.

## Outputs

After a run, the notebook writes artifacts under `/scratch/m25mac003/omnidoc_outputs`:

- `models/omnidoc_mamba_v3_checkpoint.pt` — model checkpoint
- `results/execution_summary.txt` — configuration and summary metrics
- `results/efficiency_benchmark.csv` and `.png` — sequence benchmark data and plot
- `results/five_test_documents_process.csv` and `.png` — sample outputs and visualizations
- `results/training_curves.png`, `dataset_visualizations.png`, and `model_visual_diagnostics.png`
- `results/test_document_*_index_*.png` — individual test document visualizations

## Known limitations

The notebook's final notes identify these open research issues:

1. Box-derived row/column topology is prepared but is not consumed by the visual forward path; patch-grid positions are used instead.
2. The current 672 px canvas may still be too small for fine text in scanned forms.
3. Synthetic rendered pages have a domain gap from real scanned documents; broader synthetic data or a pretrained vision encoder should be explored.
4. Current fine-tuning targets reading-order text, not structured FUNSD entity JSON. Entity-level F1, CORD, and RVL-CDIP results are not reported by this run.
5. Prompt gating is not meaningfully tested with a single prompt, and only one random seed is reported.
6. The default decoder is `distilgpt2`; larger thesis-design decoders are not part of this recorded configuration.
7. Real Mamba results require `mamba_ssm` to install and pass the notebook's CUDA preflight. Otherwise, clearly report the fallback implementation.

Treat the notebook's evaluation and benchmark outputs as experimental diagnostics until these limitations are addressed and the image-use sanity check passes.
