# 🔍 Search Model Checkpoints

## 1️⃣ RemoteCLIP

### Download

Download pretrained weights from the [RemoteCLIP HuggingFace Repository](https://huggingface.co/chendelong/RemoteCLIP).

### Usage
```bash
--search_model_path checkpoints/RemoteCLIP/RemoteCLIP-ViT-L-14.pt
```

---

## 2️⃣ GeoRSCLIP

### Download

Download from the [RS5M HuggingFace Repository](https://huggingface.co/Zilun/GeoRSCLIP).

### Usage
```bash
--search_model_path checkpoints/GeoRSCLIP/RS5M_ViT-H-14.pt
```

---

## 3️⃣ DGTRS-CLIP

### Download

Download from HuggingFace:
- [DGTRS-CLIP-ViT-B-16](https://huggingface.co/MitsuiChen14/DGTRS-CLIP-ViT-B-16)
- [DGTRS-CLIP-ViT-L-14](https://huggingface.co/MitsuiChen14/DGTRS-CLIP-ViT-L-14)

### Usage
```bash
--search_model_path checkpoints/DGTRS-CLIP/DGTRS-CLIP-ViT-B-16.pt
```

---

## 📁  Directory Structure

After setup, your `checkpoints/` folder should look like:
```
checkpoints/
├── README.md
├── RemoteCLIP/
│   ├── RemoteCLIP-RN50.pt
│   ├── RemoteCLIP-ViT-B-32.pt
│   └── RemoteCLIP-ViT-L-14.pt
├── GeoRSCLIP/
│   ├── RS5M_ViT-B-32.pt
│   ├── RS5M_ViT-L-14.pt
│   └── RS5M_ViT-H-14.pt
└── DGTRS-CLIP/
    ├── DGTRS-CLIP-ViT-B-16.pt
    └── DGTRS-CLIP-ViT-L-14.pt
```