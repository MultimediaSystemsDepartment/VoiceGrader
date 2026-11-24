from __future__ import annotations

import os


def subprocess_environment(paths_file: str, gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["RLHF_PATHS_FILE"] = paths_file
    if gpu == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env