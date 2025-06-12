# -*- coding: utf-8 -*-
#
# @File:   dataset.py
# @Author: Haozhe Xie
# @Date:   2025-05-15 14:25:09
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-05-20 14:43:56
# @Email:  root@haozhexie.com

import torch


def get_dataset(cfg, dataset, split):
    if dataset == "DOM":
        return DomDataset(cfg.DOM, split)
    elif dataset == "OXE":
        return OxeDataset(cfg.OXE, split)
    else:
        raise ValueError("Unknown dataset: %s" % dataset)


class DomDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super(DomDataset, self).__init__()
        self.dataset_cfg = cfg
        self.split = split
        self.files = self._get_files()

    def __len__(self):
        pass

    def __getitem__(self):
        pass

    def _get_files(self):
        pass


class OxeDataset(torch.utils.data.IterableDataset):
    def __init__(self, cfg, split):
        super(OxeDataset, self).__init__()
        self.dataset_cfg = cfg
        self.split = split

        raise NotImplementedError
