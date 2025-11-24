# -*- coding: utf-8 -*-
#
# @File:   memcached.py
# @Author: Haozhe Xie
# @Date:   2025-10-21 10:13:21
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-11-06 15:24:08
# @Email:  root@haozhexie.com

import hashlib
import json
import logging
import os
import pathlib
import time

import pylibmc


class MCClient:
    def __init__(
        self,
        enabled: bool = False,
        servers: list[str] = ["127.0.0.1"],
    ) -> None:
        self.ttl = 86400  # 1 day
        self.chunk_size = 512 * 1024  # 512 KB
        self.mc_cfg = (
            {
                "servers": servers,
                "binary": True,
                "behaviors": {"tcp_nodelay": True, "ketama": True},
                "pool_size": 4,
            }
            if enabled
            else None
        )
        self.pid = None
        self.mc_client = None

    def _get_file_content(self, path: str | pathlib.Path) -> bytes:
        with open(path, "rb") as fp:
            return fp.read()

    def _get_mc_client(self) -> pylibmc.ClientPool:
        assert self.mc_cfg is not None
        if self.mc_client is None or self.pid != os.getpid():
            self.pid = os.getpid()
            self.mc_client = pylibmc.ClientPool(
                pylibmc.Client(
                    self.mc_cfg["servers"],
                    binary=self.mc_cfg["binary"],
                    behaviors=self.mc_cfg["behaviors"],
                ),
                self.mc_cfg["pool_size"],
            )
        return self.mc_client

    def _get_mc_key(self, path: str | pathlib.Path) -> str:
        st = os.stat(path)
        sig = f"{st.st_ino}-{int(st.st_mtime)}-{st.st_size}"
        return hashlib.md5(sig.encode()).hexdigest()

    def _get_mc_chunk_keys(self, manifest_key: str, n_chunks: int) -> list[str]:
        return [f"{manifest_key}:{i:05}" for i in range(n_chunks)]

    def _get_mc_value(self, path: str | pathlib.Path) -> bytes | None:
        manifest_key = self._get_mc_key(str(path))
        # Ideally, all manifest and chunks should be in MemCached.
        # manifest_value = mc.get(manifest_key)
        manifest_value = self._mc_with_retry(lambda mc, key: mc.get(key), manifest_key)
        if manifest_value is not None:
            manifest_value = json.loads(manifest_value.decode())
            mc_chunk_keys = self._get_mc_chunk_keys(
                manifest_key, manifest_value["n_chunks"]
            )
            # mc_chunks = mc.get_multi(mc_chunk_keys)
            mc_chunks = self._mc_with_retry(
                lambda mc, keys: mc.get_multi(keys), mc_chunk_keys
            )
            if mc_chunks is not None and len(mc_chunks) == manifest_value["n_chunks"]:
                mc_value = b"".join([mc_chunks[k] for k in mc_chunk_keys])
                if hashlib.sha256(mc_value).hexdigest() == manifest_value["sha256"]:
                    return mc_value

        # Missing or corrupted in MemCached
        mc_value = self._get_file_content(path)
        n_chunks = (os.path.getsize(path) + self.chunk_size - 1) // self.chunk_size
        manifest_value = {
            "n_chunks": n_chunks,
            "sha256": hashlib.sha256(mc_value).hexdigest(),
        }
        mc_chunk_keys = self._get_mc_chunk_keys(manifest_key, n_chunks)
        chunks = {}
        for mci, mck in enumerate(mc_chunk_keys):
            chunks[mck] = mc_value[mci * self.chunk_size : (mci + 1) * self.chunk_size]
            # mc.set_multi(chunks, self.ttl)
            self._mc_with_retry(lambda mc, items: mc.set_multi(items, self.ttl), chunks)
            # mc.set(manifest_key, json.dumps(manifest_value).encode(), self.ttl)
            self._mc_with_retry(
                lambda mc, key, value: mc.set(key, value, self.ttl),
                manifest_key,
                json.dumps(manifest_value).encode(),
            )

        return mc_value

    def _mc_with_retry(self, fn: callable, *args: tuple) -> any:
        N_TRIES = 3

        for i in range(N_TRIES):
            try:
                with self._get_mc_client().reserve() as mc:
                    return fn(mc, *args)
            except (pylibmc.ConnectionError, pylibmc.ServerDown) as ex:
                logging.warning(
                    f"[{i + 1}/{N_TRIES}] MemCached connection error: {ex} while "
                    f"invoking {fn.__name__}."
                )
                self.pid = None
                self.mc_client = None
                time.sleep(0.1 * (i + 1))

    def get(self, path: str | pathlib.Path) -> bytes:
        if self.mc_cfg is None:
            return self._get_file_content(path)

        mc_value = self._get_mc_value(path)
        # Fallback to read from disk
        if mc_value is None:
            mc_value = self._get_file_content(path)

        return mc_value
