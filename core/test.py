# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2025-05-15 20:06:57
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-07-09 19:29:51
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
            required_features=cfg.DATASET.REQUIRED_FEATURES,
            delta_timestamps=utils.helpers.get_delta_timestamps(
                cfg.CONST.POLICY_NAME, cfg.DATASET.DELTA_TIMESTAMPS
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
            cfg.CONST.POLICY_NAME,
            test_data_loader.dataset.meta,
            cfg.DATASET.REQUIRED_FEATURES,
        )
        logging.info("Recovering from %s ..." % (cfg.CONST.CKPT))
        policy = policy.from_pretrained(cfg.CONST.CKPT)
        policy.device = policy.config.device

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

            actions = []
            for _ in range(policy.config.n_action_steps):
                actions.append(policy.select_action(batch))

            test_losses.update(
                l1_loss(torch.vstack(actions).unsqueeze(0), batch["action"]).item()
            )
            if utils.distributed.is_master():
                logging.info(
                    "Test[%d/%d] Losses = %.4f"
                    % (batch_idx + 1, n_samples, test_losses.val())
                )

    return test_losses
