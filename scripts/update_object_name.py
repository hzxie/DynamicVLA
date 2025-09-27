# -*- coding: utf-8 -*-
#
# @File:   update_object_name.py
# @Author: Haozhe Xie
# @Date:   2025-09-23 19:14:01
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-09-23 19:35:13
# @Email:  root@haozhexie.com

import os
import sys

root = sys.argv[1]
assert os.path.exists(root) and os.path.isdir(root)

files = os.listdir(root)
for f in files:
    assert "_" in f and f.endswith(".usd")
    tokens = f.split("_")
    prefix = "".join(tokens[:-1])
    suffix = int(tokens[-1][:-4])
    new_name = "%s%02d.usd" % (prefix, suffix)
    os.rename(os.path.join(root, f), os.path.join(root, new_name))
