import time
import wandb
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryStatScores

from model import build_model
from loss_function import BCEDiceLoss
from optimizer_factory import build_optimizer
from scheduler_factory import build_scheduler
from utils import compute_metrics, convert_numerics

class SegmentationModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Save config to checkpoints
        self.save_hyperparameters(config)

        # Build model
        self.model = build_model(config)

        self.loss_fn = BCEDiceLoss()

        self.train_stats = BinaryStatScores(threshold=0.5)
        self.val_stats = BinaryStatScores(threshold=0.5)

    def forward(self, x, decoder_precision=None, cfenet_precision=None):
        return self.model(x, decoder_precision=decoder_precision, cfenet_precision=cfenet_precision)

    def training_step(self, batch, batch_idx):

        # if batch_idx < 2:
        #     imgs, masks, *_ = batch
        #     print(f"TRAIN batch {batch_idx} | img mean: {imgs.mean():.6f} std: {imgs.std():.6f} | mask sum: {masks.sum():.0f}")        

        images, masks, _, _, _ = batch

        decoder_fp32 = "fp32" if self.config["training"].get("decoder_force_fp32", False) else None
        cfenet_fp32 = "fp32" if self.config["training"].get("cfenet_force_fp32", False) else None
        
        preds = self(images, decoder_precision=decoder_fp32, cfenet_precision=cfenet_fp32)

        loss = self.loss_fn(preds, masks)

        # So few bad loss values doesn't poison the whole curve
        if not torch.isfinite(loss["total_loss"]):
            print(f"Non-finite training loss at epoch {self.current_epoch}, batch {batch_idx}")
            self.optimizers().zero_grad()
            return None

        self.train_stats.update(torch.sigmoid(preds).detach(), masks.int())

        self.log("train_loss_bce", loss["bce_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_dice", loss["dice_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_total", loss["total_loss"], on_step=True, on_epoch=True, prog_bar=True)

        # print(loss)

        # if batch_idx % 15 == 0:
        #     print(f"step {batch_idx}: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, "
        #         f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB peak")

        return loss["total_loss"]

    def validation_step(self, batch, batch_idx):

        # if batch_idx < 2:
        #     imgs, masks, *_ = batch
        #     with torch.no_grad():
        #         preds = torch.sigmoid(self(imgs))
            # print(f"VAL batch {batch_idx} | img mean: {imgs.mean():.6f} std: {imgs.std():.6f} | mask sum: {masks.sum():.0f} | pred mean: {preds.mean():.6f}")

        images, masks, _, _, _ = batch

        # Validation in FP16.
        preds = self(images)
        loss = self.loss_fn(preds, masks)

        if not torch.isfinite(preds).all():
            print(
                f"Non-finite validation predictions "
                f"epoch={self.current_epoch} "
                f"batch={batch_idx}"
            )
            return
        if not torch.isfinite(loss["total_loss"]):
            print(
                f"Non-finite validation loss "
                f"epoch={self.current_epoch} "
                f"batch={batch_idx}"
            )
            return

        self.val_stats.update(torch.sigmoid(preds).detach(), masks.int())

        self.log("val_loss_bce", loss["bce_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss_dice", loss["dice_loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss_total", loss["total_loss"], on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0 and self.trainer.is_global_zero:
            self.log_images(images, masks, preds)

    def configure_optimizers(self):
        cfenet_enabled = self.config["model"]["cfenet"]

        encoder_lr = float(self.config['training']['encoder_lr'])
        decoder_lr = float(self.config['training']['decoder_lr'])

        param_groups = [
            {'params': self.model.encoder.parameters(), 'lr': encoder_lr},
            {'params': self.model.decoder.parameters(), 'lr': decoder_lr}
        ]

        if cfenet_enabled:
            cfenet_lr = float(self.config['training']['cfenet_lr'])
            param_groups.append({'params': self.model.cfenet.parameters(), 'lr': cfenet_lr})

        optimizer = build_optimizer(param_groups, self.config)
        scheduler = build_scheduler(optimizer, self.config)

        if scheduler is None:
            return optimizer

        if self.config["scheduler"]["name"] == "reduce_on_plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.config["scheduler"]["monitor"]
                }
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
        self.skipped_steps = 0
        self._prev_scale = None
        print(f"\nEpoch {self.current_epoch + 1} started")

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        encoder_lr = float(self.optimizers().param_groups[0]["lr"])
        decoder_lr = float(self.optimizers().param_groups[1]["lr"])

        tp, fp, tn, fn, _ = self.train_stats.compute()

        metrics = compute_metrics(tp.float(), fp.float(), fn.float(), tn.float())

        self.log("train_pos_iou", metrics["pos_iou"])
        self.log("train_neg_iou", metrics["neg_iou"])
        self.log("train_mean_iou", metrics["mean_iou"])
        self.log("train_precision", metrics["precision"])
        self.log("train_recall", metrics["recall"])
        self.log("train_f1", metrics["f1"])
        self.log("train_accuracy", metrics["accuracy"])
        self.log("encoder_lr", encoder_lr)
        self.log("decoder_lr", decoder_lr)

        if len(self.optimizers().param_groups) > 2:
            cfenet_lr = float(self.optimizers().param_groups[2]["lr"])
            self.log("cfenet_lr", cfenet_lr)
        
        # Log number of dismissed batches by scaler
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        if scaler is not None:
            self.log('skipped_steps', self.skipped_steps)

        self.train_stats.reset()

        print(f"Epoch {self.current_epoch + 1} finished "
            f"| Time: {epoch_time:.2f}s "
            f"| Train BIoU: {metrics['pos_iou']:.4f}"
            f"| Encoder LR: {encoder_lr:.2e}"
            f"| Decoder LR: {decoder_lr:.2e}")
        if len(self.optimizers().param_groups) > 2:
            print(f"| CFENet LR: {cfenet_lr:.2e}")

    def on_validation_epoch_end(self):
        tp, fp, tn, fn, _ = self.val_stats.compute()

        metrics = compute_metrics(tp.float(), fp.float(), fn.float(), tn.float())

        self.log("val_pos_iou", metrics["pos_iou"])
        self.log("val_neg_iou", metrics["neg_iou"])
        self.log("val_mean_iou", metrics["mean_iou"])
        self.log("val_precision", metrics["precision"])
        self.log("val_recall", metrics["recall"])
        self.log("val_f1", metrics["f1"])
        self.log("val_accuracy", metrics["accuracy"])

        self.val_stats.reset()

        print(f"Val BIoU: {metrics['pos_iou']:.4f}")

    def on_load_checkpoint(self, checkpoint):
        fixed = convert_numerics(checkpoint)
        checkpoint.clear()
        checkpoint.update(fixed)

    def on_before_optimizer_step(self, optimizer):
        # Log scaler deets to inspect gradient magnitudes and dismissed batches if grads explode.
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)

        if scaler is not None:
            current_scale = scaler.get_scale()
            
            if self._prev_scale is not None and current_scale < self._prev_scale:
                self.skipped_steps += 1
            self._prev_scale = current_scale

            self.log('loss_scale', current_scale, on_step=True, on_epoch=False)
        
        grad_norm = torch.nn.utils.get_total_norm(
            [p for p in self.model.parameters() if p.grad is not None]
        )
        self.log('train_grad_norm', grad_norm, on_step=True, on_epoch=False)

    def log_images(self, images, masks, preds):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()

        images = images.cpu()
        masks = masks.cpu()
        preds = preds.cpu()

        log_list = []

        for i in range(min(3, images.shape[0])):
            img = images[i] * std + mean
            img = img.clamp(0, 1)
            gt = masks[i]
            pr = preds[i]

            log_list.append(
                wandb.Image(
                    img,
                    masks={
                        "ground_truth": {"mask_data": gt.squeeze().numpy()},
                        "prediction": {"mask_data": pr.squeeze().numpy()}
                    }
                )
            )

        self.logger.experiment.log({"predictions": log_list, "epoch": self.current_epoch})

