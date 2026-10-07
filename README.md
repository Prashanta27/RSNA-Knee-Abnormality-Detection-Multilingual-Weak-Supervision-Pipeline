# RSNA Knee Abnormality Detection — Multilingual Weak-Supervision Pipeline

**Prashanta Das · DUET**

A Kaggle competition pipeline for detecting 12 clinically important knee MRI abnormalities, built around a core challenge: only **58 of 4,407 training studies** have expert labels. The remaining studies carry only free-text radiology reports in **9 different languages**.

## Problem

- **Task:** Multi-label classification of 12 knee abnormalities from MRI
- **Data:** 4,407 training studies and 819K DICOM slices (~570GB)
- **Expert labels:** Only 58 studies have radiologist-verified labels
- **Weakly labeled studies:** 4,349 studies contain only free-text radiology reports
- **Languages:** English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, and French
- **Constraint:** Code competition with reports available only during training; test-time inference uses images only, with internet disabled during submission

### 12 Target Abnormalities

1. ACL tear
2. MCL tear
3. Meniscus tear
4. Medial compartment osteoarthritis
5. Lateral compartment osteoarthritis
6. Patellofemoral osteoarthritis
7. Effusion
8. Synovitis
9. Baker's cyst
10. Contusion
11. Fracture
12. Other clinically relevant knee abnormalities

## Approach: Report → Weak Labels → Image Model

### 1. Weak Supervision from Text — NLP

A rule-based labeler extracts soft labels (`0.05–0.95`) from radiology reports using:

- Anatomy and finding regex
- Negation detection
- Severity modifiers such as small/moderate vs. large/severe
- Section-aware parsing
- Anatomical compartment detection

For example, a finding under a **"Lateral compartment"** section can be correctly attributed to the lateral compartment even when the sentence itself does not explicitly contain the word "lateral".

The labeler was built and validated separately for each language against the 58 gold-labeled studies.

### 2. Image Classification — Computer Vision

A **2.5D CNN** pipeline was developed using ResNet18 and later EfficientNet-B0.

Each study contributes:

- 3 anatomical planes
- 12 selected slices
- Shared CNN backbone for slice-level feature extraction
- Feature pooling across slices
- Final linear layer for 12-target multi-label prediction

The image model is trained using weak labels generated from the radiology reports.

Validation is performed against the **58 expert-labeled studies**, never against the model's own weak labels, avoiding circular validation.

### 3. Multilingual Weak-Supervision Expansion

Initially, the rule-based labeler understood only English. Non-English reports therefore produced nearly random label quality:

**AUC ≈ 0.49–0.50 against gold labels**

Per-language rule sets were then added for:

- Spanish
- Turkish
- Croatian
- Greek
- German
- Bulgarian
- Dutch
- French

Each language was independently validated against available gold-labeled reports before being used for image-model training.

This significantly improved the quality of the training signal.

## Key Engineering Details

### Laterality Normalization

Right-knee studies are mirrored so that medial/lateral anatomy is consistently aligned across the dataset.

Laterality is recovered using a three-tier fallback:

1. DICOM `Laterality` tag
2. Series description
3. Radiology report text

### DICOM Slice-Ordering Bug

DICOM filenames are random UIDs and do not represent anatomical slice order.

Sorting slices by filename instead of the DICOM `InstanceNumber` header can silently scramble the anatomical sequence.

This was caught by visually rendering slice grids before building the full image cache, preventing a wasted **1.6-hour cache build** on corrupted slice ordering.

### Offline Inference

Kaggle code competitions run submission notebooks with internet disabled.

Therefore:

- Pretrained ImageNet weights are downloaded only during training
- Trained checkpoints are saved locally
- Inference loads local checkpoints with `pretrained=False`
- No Hugging Face Hub or external internet dependency exists during submission

### Validation Discipline

The 58-study gold-labeled dataset was treated as the trustworthy validation set.

It was used to evaluate:

- Labeler regex changes
- Language-specific label quality
- Model architecture changes
- Training decisions

Leaderboard submissions were used only after local validation showed an improvement.

## Results

| Stage | Local AUC (Gold Set) | Leaderboard AUC |
|---|---:|---:|
| Dummy baseline (all 0.5) | — | 0.500 |
| ResNet18, English-only weak labels | 0.623 | 0.634 |
| EfficientNet-B0 backbone | 0.688 | 0.730 |
| + Spanish labeling rules | 0.713 | 0.762 |
| + 8 additional languages | **0.776** | **0.789** |

### Key Finding

The largest performance gains came from improving **training-label quality across languages**, rather than simply changing the model architecture.

This highlights an important principle in weak-supervision systems:

> **When labels are weak, improving the label-generation process can matter more than making the model more complex.**

## Tech Stack

- **PyTorch**
- **timm**
- **ResNet18**
- **EfficientNet-B0**
- **pydicom**
- **langdetect**
- **OpenCV**
- **Kaggle Notebooks**
- **NVIDIA T4 GPU**

## What This Project Demonstrates

### Weak Supervision

Designing and iteratively validating a useful training signal when expert labels are extremely scarce — only **58 of 4,407 studies** have verified labels.

### Multilingual Clinical NLP

Building rule-based information extraction across **9 languages**, with independent quality measurement instead of assuming translations or rules are correct.

### Medical Imaging Pipeline Engineering

Handling:

- DICOM data
- Multi-planar MRI
- Slice ordering
- Laterality normalization
- 2.5D volume construction
- Large-scale image caching

### Rigorous ML Practice

- Held-out validation against trustworthy ground truth
- No circular validation
- AUC-based evaluation
- Language-specific validation
- Careful debugging of silent data corruption
- Offline-safe inference
- Efficient use of limited GPU resources

## Competition

**RSNA Knee Abnormality Detection**

Hosted by the **Radiological Society of North America (RSNA)**.

[Kaggle Competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

**Final Submission Deadline:** October 22, 2026

---

## Project Summary

This project explores how **multilingual clinical text can be converted into weak supervision for medical image classification when expert annotations are extremely limited**.

The central idea is:

**Radiology Reports → Multilingual Weak Labels → MRI Training → Knee Abnormality Detection**

The results show that improving the quality and coverage of weak labels across languages can produce larger gains than simply increasing model complexity.
