# A Wavelet Diffusion Framework for Accelerated Generative Modeling with Lightweight Denoisers

[![Huggingface](https://img.shields.io/badge/Huggingface-Model%20Hub-blue.svg)](https://huggingface.co/markos-aivazoglou/wavelet-diffusion)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![FAIEMA 2025](https://img.shields.io/badge/FAIEMA-2025-blue)](https://www.faiema.org/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)

This paper has been accepted at the [FAIEMA 2025](https://www.faiema.org/) conference.

## Abstract

Denoising diffusion models have emerged as a powerful class of deep generative models, yet they remain computationally demanding due to their iterative nature and high-dimensional input space. In this work, we propose a novel framework that integrates wavelet decomposition into diffusion-based generative models to reduce spatial redundancy and improve training and sampling efficiency. By operating in the wavelet domain, our approach enables a compact multiresolution representation of images, facilitating faster convergence and more efficient inference with minimal architectural modifications. We assess this framework using UNets and UKANs as denoising backbones across multiple diffusion models and benchmark datasets. Our experiments show that a 1-level wavelet decomposition achieves a speedup of up to three times in training, with competitive Fréchet Inception Distance (FID) scores. We further demonstrate that KAN-based architectures offer lightweight alternatives to convolutional backbones, enabling parameter-efficient generation. In-depth analysis of sampling dynamics, including the impact of implicit configurations and wavelet depth, reveals trade-offs between speed, quality, and resolution-specific sensitivity. These findings offer practical insights into the design of efficient generative models and highlight the potential of frequency-domain learning for future generative modeling research.

## Architecture Overview
<img src="figures/wddpm-diffusion-new.png" alt="Architecture Overview" width="100%">

*Figure 1: Overview of the Wavelet Diffusion Model (WDDM) architecture. The model operates in the wavelet domain, leveraging wavelet decomposition to reduce spatial redundancy and improve training efficiency. The denoising backbone can be a UNet or a KAN-based architecture, allowing for flexible and efficient generative modeling.*

## Samples

<div align="center">
<img src="figures/grid_cifar10_unet_32x32.png" alt="Generated Samples C10 UNET" width="70%">

*Figure 2: Uncurated list of samples from the `unet-cifar10-lvl1` model.*

</div>


<div align="center">
<img src="figures/grid_stl10_ukan_64x64.png" alt="Generated Samples STL10 UKAN" width="70%">

*Figure 3: Uncurated list of samples from the `ukan-stl10-lvl1` model*
</div>


## 🚀 Key Features

- **Efficient Training**: Up to 3x faster training compared to standard diffusion models
- **Wavelet-Based Compression**: Operates in wavelet domain for reduced spatial redundancy
- **Multiple Architectures**: Supports multiple denoising backbones such as UNet and U-KAN
- **Flexible Framework**: Compatible with DDPM, DDIM and other standard diffusion solvers
- **Multi-Dataset Support**: Evaluated on CIFAR-10, CELEBA-HQ, and STL-10
- **Parameter Efficiency**: Significant reduction in model parameters while maintaining quality

## 🔧 Installation

```bash
pip install -r requirements.txt
# or
pip install -r requirements-cpu.txt  # For CPU-only installations
```

## 📊 Datasets

The framework supports three main datasets:

1. **CIFAR-10**: 32×32 Natural images (60,000 samples)
2. **CelebA-HQ**: 256×256 facial images (30,000 samples) 
3. **STL-10**: 64×64 natural images (100,000 samples)

Datasets will be automatically downloaded when first used.

## 🏃 Quick Start

### Training a Model

```bash
TODO
```

### Sampling with our pretrained models

```bash
# Generate samples using DDPM
python wavelet_sampling.py \
    --model-dir markos-aivazoglou/wddm-ukan-cifar10-lvl1 \
    --output-dir ./generated-images \
    --model-type UKAN \
    --num-samples 2 \
    --scheduler ddpm \
    --sampling-steps 1000 \
    --prediction-type epsilon

# Generate samples using DDIM (faster)
python wavelet_sampling.py \
    --model-dir markos-aivazoglou/wddm-ukan-cifar10-lvl1 \
    --output-dir ./generated-images \
    --model-type UKAN \
    --num-samples 2 \
    --scheduler ddim \
    --sampling-steps 50 \
    --prediction-type epsilon
```

## 📝 Configuration

The framework can run on Huggingface Accelerate for distributed training and inference.
Training configurations are stored in `config/` directory:

- `single-gpu-config.yaml`: Single GPU  setup
- `multi-gpu-config.yaml`: Multi-GPU distributed training


## 📄 License

This project is licensed under the Creative Commons License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation



## 👥 Authors

- **Markos Aivazoglou-Vounatsos** - Pioneer Centre for AI, University of Copenhagen
- **Mostafa Mehdipour Ghazi** - Pioneer Centre for AI, University of Copenhagen

## 📞 Contact

For questions feel free to contact:
- Contact the authors at `mav@di.ku.dk` or `ghazi@di.ku.dk`