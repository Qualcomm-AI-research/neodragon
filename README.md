This repository contains the code for a GitHub pages site for the publication [Neodragon](). The project page is hosted at https://qualcomm-ai-research.github.io/neodragon.

We introduce **Neodragon**, a text-to-video system capable of generating 2s (49 frames @24 fps) videos at a resolution of **[640×1024]** directly on a **Qualcomm Hexagon NPU** in a record **~6.7s** (7 FPS). Differing from existing transformer-based offline text-to-video generation models, **Neodragon** is the first to have been specifically optimised for mobile hardware to achieve efficient, low-cost, and high-fidelity video synthesis.

- **Replacing the original large 4.762B T5<sub>XXL</sub> Text-Encoder** with a much smaller 0.2B DistilT5 (DT5) with minimal quality loss, enabling the entire model to run without CPU offloading. This is enabled through a novel Text-Encoder Distillation procedure which uses only generative text-prompt data and *does not* require any image or video data.
- **Proposing an Asymmetric Decoder Distillation approach** which allows us to replace the native codec-latent-VAE decoder with a more efficient one, without disturbing the generative latent-space of the video generation pipeline.
- **Pruning of MMDiT blocks** within the denoiser backbone based on their relative importance, with recovery of original performance through a two-stage distillation process.
- **Reducing the NFE (Neural Functional Evaluation) requirement** of the denoiser by performing step distillation using a technique adapted from DMD for *pyramidal* flow-matching, thereby significantly accelerating video generation.

When paired with an optimised SSD1B first-frame image generator and QuickSRNet for **2×** super-resolution, our end-to-end **Neodragon** system becomes a highly parameter (**4.945B** full model), memory (**3.5GB** peak RAM usage), and runtime (**6.7s** E2E latency) efficient mobile-friendly model, while achieving a *VBench* total score of **81.61**, yielding high-fidelity generated videos.

By enabling low-cost, private, and on-device text-to-video synthesis, **Neodragon** democratizes AI-based video content creation, empowering creators to generate high-quality videos without reliance on cloud services. Code and model will be made publicly available at our website.

