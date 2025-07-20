# -*- coding: utf-8 -*-
#
# @File:   train.py
# @Author: Haozhe Xie
# @Date:   2025-05-15 20:06:33
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-07-19 00:55:41
# @Email:  root@haozhexie.com

import logging
import os
import shutil
import time

import diffusers.optimization
import torch

import core.test
import utils.average_meter
import utils.datasets
import utils.distributed
import utils.helpers
import utils.summary_writer


def train(cfg):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    local_rank = utils.distributed.get_rank()

    # Set up datasets
    # train_dataset = lerobot.common.datasets.lerobot_dataset.LeRobotDataset(
    train_dataset = utils.datasets.get_dataset(
        cfg.DATASET.NAME,
        split="train",
        pin_memory=cfg.DATASET.PIN_MEMORY,
        delta_action=cfg.DATASET.USE_DELTA_ACTION,
        required_features=cfg.DATASET.REQUIRED_FEATURES,
        image_transforms=utils.datasets.ImageTransforms(
            cfg.DATASET.IMG_SIZE, cfg.TRAIN.IMAGE_TRANSFORMS
        ),
        delta_timestamps=utils.helpers.get_delta_timestamps(
            cfg.POLICY.NAME, cfg.POLICY.CHUNK_SIZE, cfg.DATASET.DELTA_TIMESTAMPS
        ),
    )
    test_dataset = utils.datasets.get_dataset(
        cfg.DATASET.NAME,
        split="test",
        pin_memory=cfg.DATASET.PIN_MEMORY,
        delta_action=cfg.DATASET.USE_DELTA_ACTION,
        required_features=cfg.DATASET.REQUIRED_FEATURES,
        image_transforms=utils.datasets.ImageTransforms(cfg.DATASET.IMG_SIZE),
        delta_timestamps=utils.helpers.get_delta_timestamps(
            cfg.POLICY.NAME, cfg.POLICY.CHUNK_SIZE, cfg.DATASET.DELTA_TIMESTAMPS
        ),
    )
    train_sampler = None
    test_sampler = None
    if torch.cuda.is_available():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, rank=local_rank, shuffle=True, drop_last=True
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, rank=local_rank, shuffle=False, drop_last=True
        )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.N_WORKERS,
        pin_memory=cfg.DATASET.PIN_MEMORY,
        sampler=train_sampler,
        persistent_workers=True,
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.N_WORKERS,
        pin_memory=cfg.DATASET.PIN_MEMORY,
        sampler=test_sampler,
        persistent_workers=True,
    )

    # Set up the policy
    policy = utils.helpers.get_policy(
        cfg.POLICY.NAME,
        train_dataset.meta,
        cfg.DATASET.IMG_SIZE,
        cfg.POLICY.CHUNK_SIZE,
        cfg.DATASET.REQUIRED_FEATURES,
    )
    if utils.distributed.is_master():
        logging.info(
            "Using policy: %s with config %s" % (cfg.POLICY.NAME, policy.config)
        )
        logging.info(
            "#Parameters: %s/%s"
            % (
                utils.helpers.get_formatted_big_number(
                    utils.helpers.get_n_parameters(policy, trainable_only=True)
                ),
                utils.helpers.get_formatted_big_number(
                    utils.helpers.get_n_parameters(policy, trainable_only=False)
                ),
            )
        )

    init_epoch = 0
    if "CKPT" in cfg.CONST:
        logging.info("Loading pretrained model from %s ..." % cfg.CONST.CKPT)
        # Save the normalizers to enable migration to the new datasets
        normalizers = {
            n: getattr(policy, n)
            for n in ["normalize_inputs", "normalize_targets", "unnormalize_outputs"]
        }
        policy.config.device = "cuda:%d" % local_rank
        policy = policy.from_pretrained(cfg.CONST.CKPT, config=policy.config)
        for k, v in normalizers.items():
            setattr(policy, k, v)

    if torch.cuda.is_available():
        policy = torch.nn.parallel.DistributedDataParallel(
            policy.to(local_rank),
            device_ids=[local_rank],
            find_unused_parameters=True,
        )

    # Set up the optimizer
    n_batches = len(train_data_loader)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=cfg.TRAIN.OPTIMIZER.LR,
        eps=cfg.TRAIN.OPTIMIZER.EPS,
        weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        betas=cfg.TRAIN.OPTIMIZER.BETAS,
    )
    lr_scheduler = diffusers.optimization.get_scheduler(
        name=cfg.TRAIN.LR_SCHEDULER.NAME,
        optimizer=optimizer,
        num_warmup_steps=cfg.TRAIN.LR_SCHEDULER.N_WARMUP_STEPS,
        num_training_steps=cfg.TRAIN.N_EPOCHS * n_batches,
    )

    # Set up folders for logs, snapshot and checkpoints
    if utils.distributed.is_master():
        output_dir = os.path.join(cfg.DIR.OUTPUT, "%s", cfg.CONST.EXP_NAME)
        cfg.DIR.CHECKPOINTS = output_dir % "checkpoints"
        cfg.DIR.LOGS = output_dir % "logs"
        os.makedirs(cfg.DIR.CHECKPOINTS, exist_ok=True)
        # Summary writer
        tb_writer = utils.summary_writer.SummaryWriter(cfg)
        # Log current config
        tb_writer.add_config(cfg.TRAIN)

    for epoch_idx in range(init_epoch, cfg.TRAIN.N_EPOCHS):
        epoch_start_time = time.perf_counter()
        batch_time = utils.average_meter.AverageMeter()
        data_time = utils.average_meter.AverageMeter()
        train_losses = utils.average_meter.AverageMeter()
        # Randomize the DistributedSampler
        if train_sampler:
            train_sampler.set_epoch(epoch_idx)

        # Training loop
        policy.train()
        batch_end_time = time.perf_counter()
        for batch_idx, batch in enumerate(train_data_loader):
            n_itr = epoch_idx * n_batches + batch_idx
            data_time.update(time.perf_counter() - batch_end_time)
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

            loss, _ = policy.forward(batch)
            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.update(loss.item())
            batch_time.update(time.perf_counter() - batch_end_time)
            batch_end_time = time.perf_counter()
            if utils.distributed.is_master():
                tb_writer.add_scalars({"Loss/Batch": train_losses.val()}, n_itr)
                logging.info(
                    "[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %.4f"
                    % (
                        epoch_idx + 1,
                        cfg.TRAIN.N_EPOCHS,
                        batch_idx + 1,
                        n_batches,
                        batch_time.val(),
                        data_time.val(),
                        train_losses.val(),
                    )
                )

        epoch_end_time = time.perf_counter()
        if utils.distributed.is_master():
            tb_writer.add_scalars({"Loss/Epoch/Train": train_losses.avg()}, epoch_idx)
            logging.info(
                "[Epoch %d/%d] EpochTime = %.3f (s) Losses = %.4f"
                % (
                    epoch_idx + 1,
                    cfg.TRAIN.N_EPOCHS,
                    epoch_end_time - epoch_start_time,
                    train_losses.avg(),
                )
            )

        # Evaluate the current model
        test_losses = core.test(
            cfg,
            test_data_loader=test_data_loader,
            policy=policy,
        )
        if utils.distributed.is_master():
            tb_writer.add_scalars({"Loss/Epoch/Test": test_losses.avg()}, epoch_idx)

        # Save the model checkpoint
        if utils.distributed.is_master():
            logging.info("Saving model checkpoint to %s ..." % cfg.DIR.CHECKPOINTS)
            policy.module.save_pretrained(cfg.DIR.CHECKPOINTS)
            if epoch_idx % cfg.TRAIN.CKPT_SAVE_FREQ == 0:
                shutil.copy(
                    os.path.join(cfg.DIR.CHECKPOINTS, "model.safetensors"),
                    os.path.join(
                        cfg.DIR.CHECKPOINTS, "model.epoch%04d.safetensors" % epoch_idx
                    ),
                )
