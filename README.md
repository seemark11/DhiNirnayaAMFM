# Leveraging AM and FM Rhythm Spectrograms for Dementia Classification and Assessment

This repository contains the implementation and results for the paper:

**"Leveraging AM and FM Rhythm Spectrograms for Dementia Classification and Assessment"**

This work hypothesizes that **long-term rhythmic deviations** in speech can be captured more effectively using **RFA-derived rhythm spectrograms**, and demonstrates their utility in both handcrafted and deep learning pipelines.

---

## Overview

This study explores the potential of **Rhythm Formant Analysis (RFA)** to capture long-term temporal modulations in dementia speech. Specifically, we introduce **RFA-derived rhythm spectrograms** as novel features for dementia classification and regression (MMSE score prediction) tasks.

We present two complementary methodologies:
1. **Handcrafted Feature Approach**:
   - Extracts rhythm-based features from AM and FM spectrograms.
   - Uses **Support Vector Machine (SVM)** for classification and **SVM/Decision Tree (DT)** for regression.
2. **Data-driven ViT-BERT Fusion**:
   - Extracts embeddings from a **Vision Transformer (ViT)** for acoustic data and **BERT** for linguistic features.
   - Embeddings are fused for **end-to-end (E2E) dementia classification**, and also reused in ML models for MMSE score regression.

---

##  Key Contributions

- **Novel Rhythm Spectrograms**: First use of RFA-based rhythm spectrograms in dementia assessment.
- **Performance Gains**:
  - Handcrafted features outperform eGeMAPs with **14.2% relative improvement** in classification accuracy.
  - RFA + ViT surpasses Mel spectrograms with **13.1% relative improvement**.
- **Multimodal Fusion**: Integrates acoustic (ViT) and linguistic (BERT) features for robust prediction.

---


