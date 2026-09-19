# OmniDoc-Mamba v2: Visually-Rich Document Understanding

**Author**: Arjun (IIT Jodhpur, M.Tech AI)
**Domain**: Visually-Rich Document Understanding (VRDU) / State-Space Models (SSMs)

OmniDoc-Mamba v2 is an end-to-end, OCR-free deep learning architecture designed for Visually-Rich Document Understanding (VRDU). By replacing traditional quadratic-scaling transformers with a linear-time bidirectional Mamba-style state-space model, this pipeline achieves state-of-the-art memory efficiency. The system processes document images through self-supervised patch modeling and performs downstream Key Information Extraction (KIE) via a multimodal 4-bit QLoRA fine-tuned Large Language Model.

---

## 🚀 Key Features & Advantages

* **Linear-Time Visual Processing:** Eliminates the quadratic memory bottleneck of standard patch attention by leveraging a Segment-First Bidirectional Scan (SFBS) Mamba SSM.


* **Extreme Memory Efficiency:** Operates with a measured peak GPU memory footprint of just 1.61 GB, dramatically outperforming LayoutLMv3 (42.8 GB) and UDoc (40.0 GB).


* **Scale-Invariant Topology:** Maps document features onto a 100x100 topological mesh and utilizes 224x224 dynamic multi-view cropping (4 local crops + 1 global context view).


* **Interactive Inference:** Includes a built-in interactive upload widget to dynamically process and visualize document crops, topological meshes, and predicted text.



---

## 🛠️ Setup & Installation

This project is configured as a standalone Jupyter Notebook optimized for Google Colab environments with GPU access (e.g., Tesla T4).

### Prerequisites

The environment requires **PyTorch 2.11.0+cu128** with CUDA support. The following core dependencies must be installed:

* **Deep Learning & PEFT**: `transformers`, `accelerate`, `peft`, `bitsandbytes`

* **Data Processing & Evaluation**: `datasets`, `scikit-learn`

* **Document Parsing & UI**: `pymupdf` (fitz), `gradio`, `ipywidgets`

* **Visualization**: `matplotlib`, `seaborn`, `Pillow` (PIL)



### Project Structure

Upon execution, the notebook automatically provisions the following local directory structure to manage pipeline assets:

* `/content/data` — For raw and cached datasets (FUNSD/CORD).


* `/content/models` — For saving pretraining and fine-tuning model checkpoints (`.pt`).


* `/content/results` — For outputting evaluation metrics and visualization montages.


* `/content/notebooks` — For Jupyter workspace files.



---

## 🧠 Neural Network Architecture (The 5 Phases)

The `OmniDocMambaV2` model is constructed using a five-phase modular architecture:

1. **Phase 1: PatchProjector**
* Projects raw document images into a compact visual sequence using a 16x16 patch size and a 128-dimensional embedding space.




2. **Phase 2: Spatial & Local Graph Attention**
* Applies a scale-invariant 2D spatial embedding (row/column indexing up to 100x100).


* Utilizes a Graph Attention Network (GAT)-style local message passing mechanism across 4 neighboring patch nodes.




3. **Phase 3: Bidirectional Mamba Core**
* Processes the visual sequence using a 4-layer `SimplifiedMambaBlock` with a segment size of 16. This ensures $O(N)$ linear-time state updates in both forward and backward directions.




4. **Phase 4: Dimension Alignment**
* Uses an MLP (Multi-Layer Perceptron) with GELU activations to map the 128-D Mamba embeddings into a 768-D latent space compatible with the LLM.




5. **Phase 5: Causal LLM Decoder**
* Integrates a causal language model (default: `distilgpt2`) equipped with 4-bit Quantized Low-Rank Adapters (QLoRA). The aligned visual features act as a prompt prefix to guide auto-regressive text generation.





---

## 📊 Training Protocol

The pipeline utilizes a two-stage training protocol optimized with AdamW and a Cosine Annealing Learning Rate scheduler (`lr=2e-4`, `weight_decay=0.01`).

* **Stage A: Self-Supervised Masked Patch Reconstruction (Pretraining)**
* Randomly masks 15% of the projected visual patches.


* Trains the Mamba core to reconstruct the masked pixels via Mean Squared Error (MSE) loss.


* Duration: 5 Epochs.




* **Stage B: Multimodal QLoRA Fine-tuning**
* Supervised fine-tuning using teacher-forced target-token predictions on the FUNSD dataset.


* Task: Key Information Extraction (KIE).


* Duration: 15 Epochs.





---

## 📈 Evaluation & Benchmarks

The model is evaluated on the held-out test split of the FUNSD benchmark (149 train documents / 50 test documents). The current pipeline tracks Entity-Level F1 Score (Precision, Recall, F1) for KIE tasks and Accuracy for classification (e.g., RVL-CDIP).

| Model | FUNSD F1 (%) | CORD F1 (%) | RVL-CDIP Acc (%) | Peak GPU Memory (GB) |
| --- | --- | --- | --- | --- |
| UDoc | 87.93 | 96.86 | 95.05 | 40.00 |
| LayoutLMv3 | 90.29 | 96.56 | 95.44 | 42.80 |
| DocMamba | 91.70 | 97.00 | 96.00 | 5.00 |
| **OmniDoc-Mamba v2 (Ours)** | 12.72* | N/A | N/A | **1.61** |

(Note: The current F1 score of 12.72% reflects a rapid 15-epoch prototype training run on a single Colab GPU; extended epoch counts and larger visual encoders are recommended for production-level accuracy.)

Are there any specific installation scripts (e.g., a `requirements.txt` generation) or deployment pipelines you would like me to draft to accompany this repository?
