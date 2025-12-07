

from torchvision import transforms


def _convert_to_rgb(image):
    return image.convert('RGB')


def get_preprocess(image_resolution=224, is_train=False, subset_name="clip"):
    """
    Get image preprocessing transform for RS5M-CLIP model.
    
    Args:
        image_resolution: Target image resolution (default: 224)
        is_train: Whether to use training augmentation (default: False)
        subset_name: Normalization preset, one of:
            - "clip": Standard CLIP normalization
            - "imagenet": ImageNet normalization
            - "rs5m": RS5M dataset normalization (recommended for RS5M-CLIP)
            - "pub11": PUB11 dataset normalization
            - "rs3": RS3 dataset normalization
            - "geometa": GeoMeta dataset normalization
    
    Returns:
        torchvision.transforms.Compose: Image preprocessing pipeline
    """
    if subset_name == "clip":
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    elif subset_name == "imagenet":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    elif subset_name == "rs5m":
        normalize = transforms.Normalize(
            mean=[0.406, 0.423, 0.390], 
            std=[0.188, 0.175, 0.185]
        )
    elif subset_name == "pub11":
        normalize = transforms.Normalize(
            mean=[0.445, 0.469, 0.441], 
            std=[0.208, 0.193, 0.213]
        )
    elif subset_name == "rs3":
        normalize = transforms.Normalize(
            mean=[0.350, 0.356, 0.316], 
            std=[0.158, 0.147, 0.143]
        )
    elif subset_name == "geometa":
        normalize = transforms.Normalize(
            mean=[0.320, 0.322, 0.285], 
            std=[0.179, 0.168, 0.166]
        )
    else:
        raise ValueError(f"Unknown subset_name: {subset_name}. "
                        f"Choose from ['clip', 'imagenet', 'rs5m', 'pub11', 'rs3', 'geometa']")

    if is_train:
        preprocess = transforms.Compose([
            transforms.RandomResizedCrop(
                image_resolution,
                interpolation=transforms.InterpolationMode.BICUBIC,
                scale=(0.9, 1.0)
            ),
            _convert_to_rgb,
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize(
                size=image_resolution,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_resolution),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
    
    return preprocess