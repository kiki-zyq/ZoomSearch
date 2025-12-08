
<div align="center">

# <img src="images/logo.png" alt="ZoomSearch Logo" width="45"> Look Where It Matters

### Training-Free Ultra-HR Remote Sensing VQA via Adaptive Zoom Search

<br>

**Yunqi Zhou**<sup>1*</sup> &nbsp;·&nbsp; **Chengjie Jiang**<sup>2*</sup> &nbsp;·&nbsp; **Chun Yuan**<sup>2</sup> &nbsp;·&nbsp; **Jing Li**<sup>3†</sup>

<sup>1</sup> Central University of Finance and Economics &nbsp;&nbsp; <sup>2</sup> Tsinghua University &nbsp;&nbsp; <sup>3</sup> East China Normal University

<sub>* Equal Contribution &nbsp;&nbsp; † Corresponding Author</sub>

<br>

[![Project Page](https://img.shields.io/badge/🌐_Project-Page-4A90E2?style=for-the-badge)](https://kiki-zyq.github.io/Zoom-Search/)
&nbsp;&nbsp;
[![arXiv](https://img.shields.io/badge/📄_arXiv-2511.20460-B31B1B?style=for-the-badge)](https://arxiv.org/abs/2511.20460)

<img src="images/teaser.png" alt="ZoomSearch">
</div>

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**

</div>

## 🔥 News
- **[2025.12.07]** We released our code!
- **[2025.11.26]** We released our [arXiv paper](https://arxiv.org/abs/2511.20460) and [Project Page](https://kiki-zyq.github.io/Zoom-Search/)!

## 📖 Introduction

With advances in satellite constellations, sensor technologies, and imaging pipelines, ultra-high-resolution (Ultra-HR) remote sensing imagery is becoming increasingly widespread. However, current remote sensing foundation models are ill-suited to such inputs: full-image encoding exhausts token and memory budgets, while resize-based preprocessing loses fine-grained and answer-critical details. In this context, guiding the model look where it matters before prediction becomes crucial. Therefore, we present ZoomSearch, a training-free, plug-and-play pipeline that decouples 'where to look' from 'how to answer' for Ultra-HR Remote Sensing Visual Question Answering (RS-VQA). ZoomSearch combines Adaptive Multi-Branch Zoom Search, which performs a hierarchical search over image patches to localize query-relevant regions, with Layout-Aware Patch Reassembly, which reorganizes the selected patches into a compact, layout-faithful canvas. We conduct comprehensive experiments on Ultra-HR RS-VQA benchmarks MME-RealWorld-RS and LRS-VQA, comparing against (i) strong general foundation models, (ii) remote sensing foundation models, (iii) Ultra-HR RS-VQA methods, and (iv) plug-and-play search-based VQA methods. When integrated with LLaVA-ov, ZoomSearch attains state-of-the-art accuracy across diverse tasks, improving the LLaVA-ov baseline by 26.3% on LRS-VQA and 114.8% on MME-RealWorld-RS. Meanwhile, it achieves much higher inference efficiency, outperforming prior search-based methods by 20–44% in speed.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/kiki-zyq/ZoomSearch.git
cd ZoomSearch

# Install dependencies
pip install -r requirements.txt
```
## 🌟 Demo

### Supported Models

**Vision-Language Models:**
- Qwen2-VL series
- LLaVA / LLaVA-OneVision series

**Search Models:**
- [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP)
- [GeoRSCLIP](https://github.com/om-ai-lab/RS5M)
- [DGTRS-CLIP](https://github.com/MitsuiChen14/DGTRS)
- [VisRAG](https://github.com/OpenBMB/VisRAG)

### Example Usage
```bash
python demo.py \
  --model llava_onevision_qwen2_7b_ov \
  --model_path lmms-lab/llava-onevision-qwen2-7b-ov \
  --search_model_path openbmb/VisRAG-Ret \
  --image_path demo/dota_v2_dota_v2_dota_v2_P7036.png \
  --json_path demo/dota_v2_dota_v2_dota_v2_P7036.json \
  --zoom \
  --save_intermediate
```

**Key Options:**
- `--zoom`: Enable **ZoomSearch** for coarse-to-fine region localization on high-resolution images.
- `--save_intermediate`: Save all intermediate results from zooming and retrieval.

> 💡 For CLIP-family models, place the weights under [`ZoomSearch/checkpoints/`](checkpoints/read.md).

> 💡 **Recommended:** `llava_onevision_qwen2_7b_ov` + `openbmb/VisRAG-Ret` for best performance.

## 🛰️ Evaluation

### Example Usage
```bash
python batch_eval.py \
  --model llava_onevision_qwen2_7b_ov \
  --model_path lmms-lab/llava-onevision-qwen2-7b-ov \
  --search_model_path openbmb/VisRAG-Ret \
  --dataset MME-RealWorld \
  --subset "Remote Sensing" \
  --zoom \
  --save_intermediate \
  --output_dir ./outputs/mme_realworld_rs
```

This runs evaluation on the **MME-RealWorld** dataset (Remote Sensing subset).
## 📜 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{ZoomSearch,
  title={Look Where It Matters: Training-Free Ultra-HR Remote Sensing VQA via Adaptive Zoom Search},
  author={Zhou, Yunqi and Jiang, Chengjie and Yuan, Chun and Li, Jing},
  journal={arXiv preprint arXiv:2511.20460},
  year={2025}
}
```


## 💖 Acknowledgements

We sincerely thank the teams behind **[LRS-VQA](https://github.com/VisionXLab/LRS-VQA)** and **[MME-RealWorld](https://github.com/MME-Benchmarks/MME-RealWorld)** datasets,  
as well as **[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)**, for their open-source resources that supported our research!
