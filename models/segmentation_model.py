import torch

import timm
import torch.nn as nn

from cfenet import CFENet
from decoders.unet import SwinUNet
from decoders.upernet import SwinUPerNet
from decoders.deeplabv3plus import SwinDeepLabV3Plus


def check_hook(module, inputs, output):
    def check_tensor(name, tensor):
        if not torch.is_tensor(tensor):
            return

        if not torch.isfinite(tensor).all():
            print(
                f"[{module.__class__.__name__}] {name}: "
                f"non-finite values detected "
                f"(min={tensor.min().item()}, max={tensor.max().item()})"
            )
            return

        max_abs = tensor.abs().max().item()
        if max_abs > 60000:   # close to FP16 limit (~65504)
            print(
                f"[{module.__class__.__name__}] {name}: "
                f"near FP16 overflow (max abs={max_abs:.2f})"
            )

    if torch.is_tensor(output):
        check_tensor("output", output)

    elif isinstance(output, (tuple, list)):
        for i, tensor in enumerate(output):
            check_tensor(f"output[{i}]", tensor)

class SegmentationModel(nn.Module):
    """This class will only work properly if the encoder and decoder work correctly together already.
        This doesn't verify anything, just a binding class.
        
        The CFENet is designed only for 4 feature maps, so it will throw an error if more are given.    
    """
    def __init__(self, encoder, decoder, cfenet=False, encoder_channels=None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        if not cfenet and encoder_channels is None:
            raise Exception("Pass the encoder_channels to use CFENet")

        self.cfenet = CFENet(encoder_channels) if cfenet else None
        if self.cfenet:
            self.cfenet.register_forward_hook(check_hook)
    
    def forward(self, x, decoder_precision=None, cfenet_precision=None):
        features = self.encoder(x)
        permuted_features = [f.permute(0, 3, 1, 2) for f in features]

        if self.cfenet is not None:
            if cfenet_precision == "fp32":
                with torch.autocast(device_type="cuda", enabled=False):    
                    permuted_features = [f.float() for f in permuted_features]
                    permuted_features = self.cfenet(permuted_features)
            else:
                permuted_features = self.cfenet(permuted_features)
        
        # To prevent overflow in UPerNet, the decoder is run in FP32 precision if specified.
        if decoder_precision == "fp32":
            with torch.autocast(device_type="cuda", enabled=False):
                features_fp32 = [f.float() for f in permuted_features]
                decoder_output = self.decoder(features_fp32)
        else:
            decoder_output = self.decoder(permuted_features)
    
        return decoder_output


def build_model(config):
    model_cfg = config["model"]
    if model_cfg["encoder"] == "swin_t":
        encoder_channels = [96, 192, 384, 768]

    encoder = timm.create_model(
                model_cfg["timm_name"],
                pretrained=model_cfg.get("pretrained", True),
                features_only=True)
    
    if model_cfg["decoder"] == "unet":
        decoder = SwinUNet(encoder_channels)
    elif model_cfg["decoder"] == "upernet":
        decoder = SwinUPerNet(encoder_channels)
    elif model_cfg["decoder"] == "deeplabv3plus":
        decoder = SwinDeepLabV3Plus(encoder_channels, aspp_index=3, high_res_index=0)
    else:
        raise ValueError("Invalid decoder: {model_cfg['decoder']}, choose from [unet, upernet, deeplabv3plus]")
    
    if model_cfg["cfenet"] not in [True, False]:
        raise ValueError("cfenet must be a bool")

    return SegmentationModel(encoder, decoder, model_cfg["cfenet"], encoder_channels,)
