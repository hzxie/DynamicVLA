# -*- coding: utf-8 -*-
#
# @File:   io.py
# @Author: Haozhe Xie
# @Date:   2025-10-21 10:13:21
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-21 10:28:25
# @Email:  root@haozhexie.com

import hashlib
import os
import pathlib

import pylibmc


class MCClient:
    def __init__(
        self, enabled: bool = False, servers: list[str] = ["127.0.0.1"]
    ) -> None:
        self.ttl = 86400  # 1 day
        self.client = (
            pylibmc.Client(
                servers,
                binary=True,
                behaviors={"tcp_nodelay": True, "ketama": True},
            )
            if enabled
            else None
        )

    def _get_mc_key(self, path: str | pathlib.Path) -> str:
        st = os.stat(path)
        sig = f"{st.st_ino}-{int(st.st_mtime)}-{st.st_size}"
        return hashlib.md5(sig.encode()).hexdigest()

    def _get_bytes(self, path: str | pathlib.Path) -> bytes:
        with open(path, "rb") as fp:
            return fp.read()

    def get(self, path: str | pathlib.Path) -> bytes:
        if self.client is None:
            return self._get_bytes(path)

        mc_key = self._get_mc_key(str(path))
        mc_value = self.client.get(mc_key)
        if mc_value is None:
            mc_value = self._get_bytes(path)
            self.client.set(mc_key, mc_value, time=self.ttl)

        return mc_value
