import argparse
import glob
import os
from datetime import datetime
from typing import Dict

import accelerate
import cv2
import numpy as np
import torch
from aim import Image as aim_image
from aim import Run
from loguru import logger
from segments.export import colorize
from tqdm import tqdm

from config import create_cfg, show_config
from dataset import create_data_loader, get_test_transform
from loss import SegmentLoss
from modeling import EMA, UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the model after training",
        default=False,
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Place to save the prediction file",
        default="prediction",
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def read_test_img(img_path: str, img_size: int) -> np.ndarray:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return img


def create_str_loss(loss: Dict[str, torch.Tensor], use_ssl: bool = True) -> str:
    if not use_ssl and "dis_loss" in loss:
        del loss["dis_loss"]

    loss_list = []
    for key, value in loss.items():
        loss_list.append(f"{key}: {value.item():.5f}")
    return ", ".join(loss_list)


class Trainer:
    def __init__(self, cfg):
        os.makedirs(cfg.PROJECT_DIR, exist_ok=True)
        self.accelerator = accelerate.Accelerator()
        if self.accelerator.is_main_process:
            show_config(cfg)
            self.tracker = Run(
                experiment=f"seg-only-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
            )
            self.tracker["hparams"] = dict(cfg)

        self.cfg = cfg
        self.device = self.accelerator.device
        model = UNet(img_ch=cfg.MODEL.IN_CHANNELS, output_ch=cfg.DATA.NUM_CLASSES)
        model.to(self.device)

        if cfg.TRAIN.USE_SSL:
            ema_model = UNet(
                img_ch=cfg.MODEL.IN_CHANNELS, output_ch=cfg.DATA.NUM_CLASSES
            )
            ema_model = ema_model.to(self.device)
            self.ema_updater = EMA(
                src_model=model,
                ema_model=ema_model,
                beta=cfg.TRAIN.EMA.BETA,
                update_after_step=cfg.TRAIN.EMA.UPDATE_AFTER_STEP,
                update_every=cfg.TRAIN.EMA.UPDATE_EVERY,
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        loss_fn = SegmentLoss(
            num_classes=cfg.DATA.NUM_CLASSES,
            threshold=cfg.TRAIN.THRESHOLD,
            temperature=cfg.TRAIN.TEMPERATURE,
            unlabel_weight=cfg.TRAIN.UNLABEL_WEIGHT,
            use_ssl=cfg.TRAIN.USE_SSL,
        )

        with self.accelerator.main_process_first():
            train_loader, val_loader = create_data_loader(
                root=self.cfg.DATA.ROOT,
                batch_size=self.cfg.DATA.BATCH_SIZE,
                num_workers=self.cfg.DATA.NUM_WORKERS,
                num_classes=cfg.DATA.NUM_CLASSES,
                img_size=cfg.DATA.IMG_SIZE,
                use_ssl=cfg.TRAIN.USE_SSL,
            )

        (
            self.model,
            self.optimizer,
            self.loss_fn,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            model, optimizer, loss_fn, train_loader, val_loader
        )
        if cfg.TRAIN.USE_SSL:
            for param in ema_model.parameters():
                param.requires_grad = False
            self.ema_model = self.accelerator.prepare(ema_model)
            self.ema_model.eval()
        self.min_loss = float("inf")
        self.current_epoch = 0
        self.prev_best = None

        if self.cfg.MODEL.CHECKPOINT is not None:
            with self.accelerator.main_process_first():
                self.load_from_checkpoint()

    def load_from_checkpoint(self):
        checkpoint = self.cfg.MODEL.CHECKPOINT
        if not os.path.exists(checkpoint):
            logger.warning(f"Checkpoint {checkpoint} not found. Skipping...")
            return
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model"])
        if self.cfg.TRAIN.USE_SSL:
            if checkpoint["ema_model"] is not None:
                self.accelerator.unwrap_model(self.ema_model).load_state_dict(
                    checkpoint["ema_model"]
                )
            if checkpoint["ema_updater"] is not None:
                self.ema_updater.load_state_dict(checkpoint["ema_updater"])
        self.optimizer.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_epoch = checkpoint["epoch"] + 1
        if self.accelerator.is_main_process:
            logger.info(
                f"Checkpoint loaded from {self.cfg.MODEL.CHECKPOINT}, continue training or validate..."
            )
        del checkpoint

    def denormalize_to_numpy(self, img: torch.Tensor) -> torch.Tensor:
        img = (img * 0.5) + 0.5
        img = img.permute(1, 2, 0)
        return (img * 255).cpu().numpy().astype("uint8")

    def draw_color_to_numpy(self, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.argmax(dim=0).cpu().numpy()
        color_mask = colorize(mask)
        return color_mask

    def _train_one_epoch(self):
        self.model.train()

        for loader_idx, (
            img,
            label,
            weak_unlabeled_img,
            strong_unlabeled_img,
        ) in enumerate(self.train_loader, 1):
            if self.cfg.TRAIN.USE_SSL:
                with torch.inference_mode():
                    weak_pred = self.ema_model(weak_unlabeled_img)
            with self.accelerator.accumulate(self.model):
                if self.cfg.TRAIN.USE_SSL:
                    pred, strong_pred = self.model(
                        torch.cat([img, strong_unlabeled_img], dim=0)
                    ).chunk(2, dim=0)
                    if self.current_epoch < 10:
                        weak_pred = None
                    loss = self.loss_fn(pred, label, strong_pred, weak_pred)
                else:
                    pred = self.model(img)
                    loss = self.loss_fn(pred, label)
                self.accelerator.backward(loss["total_loss"])
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.cfg.TRAIN.GRAD_CLIP
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.accelerator.sync_gradients and self.cfg.TRAIN.USE_SSL:
                self.accelerator.wait_for_everyone()
                self.ema_updater.update()

            if self.accelerator.is_main_process and (
                (loader_idx % self.cfg.TRAIN.LOG_EVERY_STEP == 0)
                or (loader_idx == len(self.train_loader))
            ):
                logger.info(
                    f"Epoch [{self.current_epoch}/{self.cfg.TRAIN.EPOCHS}] | Step [{loader_idx}/{len(self.train_loader)}] | {create_str_loss(loss, use_ssl=self.cfg.TRAIN.USE_SSL)}"
                )
                for key, value in loss.items():
                    self.tracker.track(
                        value.item(),
                        name=f"train_{key}",
                        step=(self.current_epoch - 1) * len(self.train_loader)
                        + loader_idx,
                    )

    def validate(self):
        total_loss = 0
        use_model = self.ema_model if self.cfg.TRAIN.USE_SSL else self.model
        use_model.eval()
        for loader_idx, (img, label) in enumerate(self.val_loader, 1):
            with torch.inference_mode():
                pred = use_model(img)
            loss = self.loss_fn(pred, label)
            batch_loss = self.accelerator.gather_for_metrics(loss["total_loss"])
            total_loss += batch_loss.mean().item()
            if self.accelerator.is_main_process and (
                (loader_idx % self.cfg.TRAIN.LOG_EVERY_STEP == 0)
                or (loader_idx == len(self.val_loader))
            ):
                logger.info(
                    f"Step [{loader_idx}/{len(self.val_loader)}] | {create_str_loss(loss, use_ssl=self.cfg.TRAIN.USE_SSL)}"
                )
                for key, value in loss.items():
                    self.tracker.track(
                        value.item(),
                        name=f"val_{key}",
                        step=(self.current_epoch - 1) * len(self.train_loader)
                        + loader_idx,
                    )
                target_img = self.denormalize_to_numpy(img[0])
                target_label = self.draw_color_to_numpy(label[0])
                target_pred = self.draw_color_to_numpy(pred[0])
                self.tracker.track(
                    aim_image(
                        np.hstack([target_img, target_label, target_pred]),
                        caption="Image, GT, Prediction",
                    ),
                    name=f"val_img_{loader_idx}",
                    epoch=self.current_epoch,
                )
        total_loss /= len(self.val_loader)
        if self.accelerator.is_main_process and total_loss < self.min_loss:
            logger.info(
                f"New best model found with loss: {total_loss:.5f}, save to best_model_epoch_{self.current_epoch}.pth"
            )
            if self.cfg.TRAIN.SAVE_BEST_ONLY and os.path.exists(
                os.path.join(
                    self.cfg.PROJECT_DIR, f"best_model_epoch_{self.prev_best}.pth"
                )
            ):
                os.remove(
                    os.path.join(
                        self.cfg.PROJECT_DIR, f"best_model_epoch_{self.prev_best}.pth"
                    )
                )
            self.min_loss = total_loss
            self.prev_best = self.current_epoch
            torch.save(
                {
                    "model": self.accelerator.unwrap_model(self.model).state_dict(),
                    "optimizer": self.optimizer.optimizer.state_dict(),
                    "epoch": self.current_epoch,
                    "ema_model": self.ema_model.state_dict()
                    if self.cfg.TRAIN.USE_SSL
                    else None,
                    "ema_updater": (
                        self.ema_updater.state_dict()
                        if self.cfg.TRAIN.USE_SSL
                        else None
                    ),
                },
                os.path.join(
                    self.cfg.PROJECT_DIR, f"best_model_epoch_{self.current_epoch}.pth"
                ),
            )

    def train(self):
        self.min_loss = float("inf")
        for epoch in range(1, self.cfg.TRAIN.EPOCHS + 1):
            self.current_epoch = epoch
            self._train_one_epoch()
            if epoch % self.cfg.TRAIN.VAL_FREQ == 0:
                self.accelerator.wait_for_everyone()
                self.validate()

    def test(
        self,
        save_path: str = "prediction",
        target_width: int = 640,
        target_height: int = 360,
    ):
        if not self.accelerator.is_main_process:
            return
        os.makedirs(save_path, exist_ok=True)
        img_list = list(glob.glob(os.path.join(self.cfg.DATA.ROOT, "images", "*.jpg")))
        filter_img_list = sorted(
            [img for img in img_list if int(os.path.basename(img).split(".")[0]) > 900]
        )
        batch_size = self.cfg.DATA.BATCH_SIZE
        test_transform = get_test_transform(cfg.DATA.IMG_SIZE)
        model = self.accelerator.unwrap_model(
            self.ema_model if self.cfg.TRAIN.USE_SSL else self.model
        )
        model.eval()

        for idx in tqdm(range(0, len(filter_img_list), batch_size)):
            end_idx = min(idx + batch_size, len(filter_img_list))
            img_paths = filter_img_list[idx:end_idx]
            img_list = [
                test_transform(read_test_img(img, self.cfg.DATA.IMG_SIZE))
                for img in img_paths
            ]
            img_list = torch.stack(img_list).to(self.device)
            with torch.inference_mode():
                out_list = model(img_list)
            out_list = (
                torch.nn.functional.interpolate(
                    out_list,
                    size=(target_height, target_width),
                )
                .argmax(dim=1)
                .cpu()
                .numpy()
                .astype("uint8")
            )
            for predict_mask, img_path in zip(out_list, img_paths):
                cv2.imwrite(
                    os.path.join(
                        save_path,
                        os.path.basename(img_path).replace(".jpg", "_mask.png"),
                    ),
                    predict_mask,
                )
                color_img = colorize(predict_mask)
                cv2.imwrite(
                    os.path.join(
                        save_path,
                        os.path.basename(img_path).replace(".jpg", "_color.png"),
                    ),
                    color_img,
                )

    def close(self):
        self.accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    cfg = create_cfg()
    cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    trainer = Trainer(cfg)
    if not args.test:
        trainer.train()
        trainer.close()
    if args.test:
        trainer.test(args.save_path)
