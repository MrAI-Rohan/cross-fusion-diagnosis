import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(data_cfg, mode, seed=42):
    if mode not in ["train", "val"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'.")
    
    normalization = data_cfg.get("normalization", "imagenet")
    if normalization == "imagenet":
        normalize = A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    elif normalization == "standard":
        normalize = A.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(1.0, 1.0, 1.0),
            max_pixel_value=255.0
        )
    transform_cfg = data_cfg.get(f"{mode}_transform", None)
    if transform_cfg in [None, False]:
        return A.Compose([
            normalize,
            ToTensorV2()
        ], seed=seed)

    transform_map = {
        "hflip": A.HorizontalFlip,
        "vflip": A.VerticalFlip,
        "rotate90": A.RandomRotate90,
        "brightness_contrast": A.RandomBrightnessContrast,
        "gauss_noise": A.GaussNoise,
        "blur": A.Blur,
        "elastic": A.ElasticTransform,
        "grid_distortion": A.GridDistortion,
        "shift_scale_rotate": A.ShiftScaleRotate,
        "resize": A.Resize,
        "center_crop": A.CenterCrop,
    }

    transforms = []

    for name, prob in transform_cfg.items():
        if name in ["resize", "center_crop"] and prob:
            sz = data_cfg["patch_size"]
            transforms.append(transform_map[name](height=sz, width=sz, p=1.0))
        elif name in transform_map and isinstance(prob, (int, float)) and prob > 0:
            transforms.append(transform_map[name](p=prob))

    # Always normalize + tensor
    transforms.append(normalize)

    transforms.append(ToTensorV2())

    return A.Compose(transforms, seed=seed)
