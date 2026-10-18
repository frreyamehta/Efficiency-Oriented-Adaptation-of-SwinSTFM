# Efficiency Oriented Adaptation of SwinSTFM for Spatiotemporal Satellite Image Fusion

# Efficiency-Oriented Adaptation of SwinSTFM for Spatiotemporal Satellite Image Fusion

An efficiency-focused adaptation of **SwinSTFM (Remote Sensing Spatiotemporal Fusion Using Swin Transformer)** that replaces the original Swin-Tiny backbone with a lightweight Swin-Nano variant to reduce training cost while retaining reconstruction quality.

## Overview

Satellite image fusion combines observations from multiple sensors to produce images with improved spatial and temporal resolution. High-spatial-resolution imagery often has a lower revisit frequency, while frequently acquired imagery may lack fine spatial details.

SwinSTFM addresses this challenge using a Swin Transformer-based architecture for spatiotemporal fusion. However, training transformer-based models can be computationally expensive.

This project explores a lightweight adaptation of SwinSTFM by replacing its Swin-Tiny backbone with Swin-Nano and evaluating the trade-off between computational efficiency and fusion quality.

## Objectives

- Reduce training time through a lightweight transformer backbone.
- Preserve spatial reconstruction quality while minimizing computational overhead.
- Evaluate the effect of backbone scaling on spectral consistency.
- Benchmark the adapted model against baseline configurations using standard image-fusion metrics.

## Methodology

1. **Baseline:** Establish a reference configuration using the original SwinSTFM architecture.
2. **Backbone adaptation:** Replace Swin-Tiny with Swin-Nano to reduce model complexity.
3. **Training:** Train the adapted model using PyTorch and the selected satellite image dataset.
4. **Evaluation:** Compare model configurations using reconstruction, structural similarity, and spectral consistency metrics.
5. **Analysis:** Examine the relationship between training efficiency and fusion quality.

## Experimental Results

### Quantitative Comparison on the LGC Dataset

The original SwinSTFM model was compared against the modified SwinSTFM configuration using six evaluation metrics to assess reconstruction quality, structural similarity, and spectral consistency.

| Metric | SwinSTFM (Original) | SwinSTFM (Modified) | Better Result |
|---|---:|---:|---|
| RMSE ↓ | 0.0172 ± 0.0026 | **0.0121 ± 0.0018** | Modified |
| SSIM ↑ | **0.6761 ± 0.0605** | 0.6304 ± 0.0296 | Original |
| PSNR ↑ | 26.0239 ± 1.6695 | **26.4519 ± 1.0525** | Modified |
| ERGAS ↓ | 15.8297 ± 6.0970 | **7.8620 ± 1.1589** | Modified |
| CC ↑ | **0.8986 ± 0.0358** | 0.8643 ± 0.0315 | Original |
| SAM ↓ | **3.5535 ± 0.7841** | 3.7805 ± 0.8533 | Original |

*Arrows indicate the preferred direction of each metric. Values are reported as mean ± standard deviation.*

### Key Findings

- **Improved reconstruction accuracy:** RMSE decreased from 0.0172 to 0.0121, approximately a 29.7% reduction.
- **Higher signal-to-noise ratio:** PSNR increased by approximately 0.43 dB, indicating improved pixel-level reconstruction quality.
- **Lower global error:** ERGAS decreased from 15.8297 to 7.8620, approximately a 50.3% reduction.
- **Structural similarity trade-off:** SSIM decreased from 0.6761 to 0.6304, while the correlation coefficient (CC) also declined.
- **Spectral consistency trade-off:** SAM increased from 3.5535 to 3.7805, indicating a modest deterioration in spectral-angle fidelity.

### Discussion

The modified SwinSTFM configuration achieved better RMSE, PSNR, and ERGAS scores, suggesting improved pixel-level reconstruction and lower global error. However, the original model retained higher SSIM, CC, and spectral consistency as measured by SAM.

These results illustrate that improvements in reconstruction accuracy do not necessarily translate into better structural or spectral fidelity. Model selection should therefore consider the intended downstream application and the relative importance of each evaluation metric.

The modified configuration also achieved a reported **25% improvement in training efficiency**, making it a promising option when computational cost is an important constraint.

## Repository Structure

```text
SwinSTFM-Efficient-Adaptation/
├── datasets/           # Dataset preparation and loading
├── models/             # Model architecture and backbone adaptation
├── train.py            # Training pipeline
├── test.py             # Evaluation and inference
├── utils.py            # Utility functions
├── requirements.txt    # Python dependencies
└── README.md
```

## Limitations

- Performance depends on the dataset, sensor characteristics, and training configuration.
- A lightweight backbone may affect spectral fidelity or reconstruction quality differently across metrics.
- Results obtained on a particular dataset should not automatically be generalized to other satellite sensors or geographic regions.

## Acknowledgements

This project builds on the publicly available [SwinSTFM implementation](https://github.com/LouisChen0104/swinstfm).

Original work: *SwinSTFM: Remote Sensing Spatiotemporal Fusion Using Swin Transformer.*

This repository documents an efficiency-oriented adaptation and its empirical evaluation.
