# RRCLIP: Semantic-Guided Residual Representation Learning for Generalizable Deep Video Frame Interpolation Detection

## 📖 Abstract

Deep Video Frame Interpolation (DVFI) can generate temporally consistent intermediate frames and improve video smoothness. However, when maliciously used, DVFI may insert fabricated frames into videos and alter the temporal semantics of real events, posing serious threats to video authenticity verification.

Existing DVFI detection methods mainly rely on visual artifacts. Although these methods can capture interpolation traces under known settings, they often suffer from poor generalization when facing unseen or advanced DVFI algorithms, because modern DVFI techniques produce increasingly subtle forgery cues.

To address this limitation, we propose **RRCLIP**, a semantic-guided residual representation learning framework for generalizable DVFI detection. RRCLIP adapts the powerful cross-modal representation capability of CLIP to the DVFI forensic task by jointly exploiting visual evidence and textual semantic guidance.

RRCLIP consists of three key components:

- **CLIP-based visual and text encoders**: Extract high-level visual representations from video frames and provide semantic textual guidance for cross-modal alignment.
- **Frame Representation Encoder**: Enhances inter-frame discriminability and amplifies subtle differences between authentic and interpolated frames.
- **DVFI Detection Adapter**: Aggregates motion residual representations and produces compact video-level features for detection.

In addition, RRCLIP introduces a **learnable text prompt strategy** to improve the semantic alignment between visual and textual modalities, and employs **supervised contrastive learning** to enlarge the feature distance between real and interpolated samples.

Extensive experiments demonstrate that RRCLIP achieves strong performance across multiple DVFI algorithms, target frame rates, and datasets, showing superior generalization under cross-frame-rate, cross-method, and cross-dataset evaluation settings.

## 🏗️ Architecture Overview

![RRCLIP Architecture](model.pdf)

*Architecture of the RRCLIP framework for DVFI detection. RRCLIP integrates CLIP visual and text encoders, a frame-level feature encoder, and a DVFI detection adapter to learn semantic-guided residual representations.*

The overall RRCLIP framework contains the following modules:

### 1. CLIP Visual Encoder

RRCLIP first extracts frame-level visual representations using a pre-trained CLIP visual encoder. Each frame is divided into non-overlapping patches and encoded by the CLIP ViT backbone. The output `[frame-class]` token is used as the semantic representation of each frame.

### 2. Frame Representation Encoder

The frame-level CLIP features are fed into a temporal feature encoder to enhance inter-frame differences. The encoder consists of temporal convolutional blocks that model short-range temporal dynamics and produce discriminative frame-level representations.

This module is trained with supervised contrastive learning, encouraging authentic and interpolated frames to be better separated in the feature space.

### 3. Motion Residual Representation

To capture DVFI-specific temporal artifacts, RRCLIP computes frame-wise residual features between consecutive frame representations. These residuals highlight subtle temporal inconsistencies caused by interpolation.

### 4. DVFI Detection Adapter

The residual sequence is combined with a learnable `[video-class]` token and inter-frame positional embeddings. A lightweight Tiny ViT adapter then aggregates the residual sequence and extracts compact video-level representations for classification and video-text similarity learning.

### 5. Learnable Text Prompt Learning

Instead of using hand-crafted text prompts, RRCLIP adopts learnable context tokens to adapt the CLIP text encoder to DVFI detection. The text prompt is formulated as:

```text
[tokenize(label), p1, p2, ..., pK]
