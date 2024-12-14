# Iterative Structured Knowledge Distillation: Optimizing Language Models Through Layer-by-Layer Distillation
by Malthe Have Musaeus and Rob van der Goot

This repository contains the code and implementation details for the paper *"Iterative Structured Knowledge Distillation: Optimizing Language Models Through Layer-by-Layer Distillation"*. This work introduces a novel approach to compressing transformer-based language models by combining structured pruning and knowledge distillation. By iteratively substituting transformer blocks with smaller, efficient versions during training, the method aims to reduce computational demands while preserving model performance.

## 🔍 Overview

Transformer-based models achieve state-of-the-art performance in NLP tasks but are often computationally expensive. *Iterative Structured Knowledge Distillation (ISKD)* bridges the gap between pruning and knowledge distillation, allowing gradual compression of model architectures. From our experiements we see that this method maintains over 80% of the original performance on language modeling and commonsense reasoning tasks while offering greater flexibility in model design.

## 🛠️ Methodology

The ISKD approach involves:
1. **Compact Block Design**: Creating smaller versions of transformer blocks by reducing the hidden dimension and number of attention heads.
2. **Iterative Substitution**: Gradually replacing full transformer blocks with compact blocks.
3. **Fine-Tuning**: Using a dual learning rate strategy to fine-tune the compact blocks alongside the remaining model, ensuring compatibility and performance retention.

## 📊 Key Results

- **GPT-2**: Reduced parameters by **30.68%**, maintaining **79.82%** performance on CBT and **94.25%** on HellaSwag.
- **Phi-1**: Reduced parameters by **30.16%**, increasing performance by **4.93%** on CBT and maintaining **92.43%** on HellaSwag.
- ISKD outperforms structured L1 pruning and achieves similar performance to knowledge distillation while offering iterative and adaptable compression.


## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Malthehave/ISKD-COLING-2024.git
cd ISKD
pip install -r requirements.txt
```

## 📖 Paper Abstract
Traditional language model compression techniques, like knowledge distillation, require a fixed architecture, limiting flexibility, while structured pruning methods often fail to preserve performance. This paper introduces Iterative Structured Knowledge Distillation (ISKD), which integrates knowledge distillation and structured pruning by progressively replacing transformer blocks with smaller, efficient versions during training. This study validates ISKD on two transformer-based language models: GPT-2 and Phi-1. ISKD outperforms L1 pruning and achieves similar performance to knowledge distillation while offering greater flexibility. ISKD reduces model parameters - $30.68\%$ for GPT-2 and $30.16\%$ for Phi-1 - while maintaining at least four-fifths of performance on both language modeling and commonsense reasoning tasks. These findings suggest that this method offers a promising balance between model efficiency and accuracy.