# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2025-05-15 20:06:57
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-09-06 17:56:15
# @Email:  root@haozhexie.com

import logging

import torch

import utils.datasets
import utils.distributed
import utils.helpers


def test(cfg, test_data_loader=None, policy=None):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if test_data_loader is None:
        test_dataset = utils.datasets.get_dataset(
            cfg.DATASET.NAME,
            split="test",
            pin_memory=cfg.DATASET.PIN_MEMORY,
            delta_action=cfg.DATASET.USE_DELTA_ACTION,
            required_features=cfg.DATASET.REQUIRED_FEATURES,
            image_transforms=utils.datasets.ImageTransforms(cfg.DATASET.IMG_SIZE),
            delta_timestamps=utils.helpers.get_delta_timestamps(
                cfg.POLICY, cfg.DATASET.DELTA_TIMESTAMPS
            ),
        )
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            num_workers=cfg.CONST.N_WORKERS,
            pin_memory=cfg.DATASET.PIN_MEMORY,
            shuffle=False,
        )
    if policy is None:
        policy = utils.helpers.get_policy(
            cfg.POLICY,
            test_data_loader.dataset.meta,
            cfg.DATASET.IMG_SIZE,
            cfg.DATASET.REQUIRED_FEATURES,
        )
        logging.info("Recovering from %s ..." % (cfg.CONST.CKPT))
        policy = policy.from_pretrained(cfg.CONST.CKPT)
        if torch.cuda.is_available():
            policy = torch.nn.DataParallel(policy).cuda()

        policy.device = policy.module.config.device

    l1_loss = torch.nn.L1Loss()
    policy.eval()
    n_samples = len(test_data_loader)
    test_losses = utils.average_meter.AverageMeter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            batch = {
                k: (
                    v.to(policy.device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in batch.items()
            }
            # Fix: Remove the additional dimension for task
            if isinstance(batch["task"], list) and isinstance(
                batch["task"][0], (tuple, list)
            ):
                batch["task"] = batch["task"][0]

            gt_actions = batch["action"]
            pred_actions = policy.module.predict_action_chunk(batch)

            test_losses.update(l1_loss(pred_actions, gt_actions).item())
            if utils.distributed.is_master():
                logging.info(
                    "Test[%d/%d] Losses = %.4f"
                    % (batch_idx + 1, n_samples, test_losses.val())
                )

    return test_losses
