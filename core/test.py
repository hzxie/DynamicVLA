# -*- coding: utf-8 -*-
#
# @File:   test.py
# @Author: Haozhe Xie
# @Date:   2025-05-15 20:06:57
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-06-18 19:16:41
# @Email:  root@haozhexie.com

import logging

import torch
import utils.datasets
import utils.distributed
import utils.helpers


def test(cfg, test_data_loader=None, policy=None):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    local_rank = utils.distributed.get_rank()
    if test_data_loader is None:
        test_dataset = utils.datasets.get_dataset(
            cfg.DATASET.NAME,
            split="test",
            pin_memory=cfg.DATASET.PIN_MEMORY,
            delta_timestamps=cfg.DATASET.DELTA_TIMESTAMPS,
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
            cfg.CONST.POLICY_NAME, test_data_loader.dataset.meta
        )
        logging.info("Recovering from %s ..." % (cfg.CONST.CKPT))
        policy = policy.from_pretrained(cfg.CONST.CKPT)
        if torch.cuda.is_available():
            policy = torch.nn.DataParallel(policy).cuda()

    policy.eval()
    n_samples = len(test_data_loader)
    test_losses = utils.average_meter.AverageMeter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            batch = {
                k: (v.to(policy.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            loss, _ = policy.forward(batch)
            test_losses.update(loss.item())
            if utils.distributed.is_master():
                logging.info(
                    "Test[%d/%d] Losses = %.4f"
                    % (batch_idx + 1, n_samples, test_losses.val())
                )

    return test_losses
