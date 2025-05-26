# Leveraging AM and FM Rhythm Spectrograms for Dementia Classification and Assessment

This repository contains the implementation and results for the paper:

**"Leveraging AM and FM Rhythm Spectrograms for Dementia Classification and Assessment"**

---

## Overview

This study explores **Rhythm Formant Analysis (RFA)** to capture **long-term temporal modulations** in dementia speech. We introduce **AM and FM rhythm spectrograms** as novel features for both dementia **classification** and **regression** (MMSE score prediction) tasks.

We present two complementary methodologies:
1. **Handcrafted Features:** 
   - Extracted from RFA-derived rhythm spectrograms.
   - Includes statistical descriptors and 2D DCT-based representations.
   - Input to traditional ML models.
2. **Data-driven Fusion Approach:**
   - Uses **Vision Transformer (ViT)** for acoustic modeling of rhythm spectrograms.
   - Fused with **BERT-based linguistic embeddings**.

---

##  Key Contributions

- **Novel Rhythm Spectrograms**: First use of RFA-based rhythm spectrograms in dementia assessment.
- **Performance Gains**:
  - Handcrafted features outperform eGeMAPs with **14.2% relative improvement** in classification accuracy.
  - RFA + ViT surpasses Mel spectrograms with **13.1% relative improvement**.
- **Multimodal Fusion**: Integrates acoustic (ViT) and linguistic (BERT) features for robust prediction.

---


