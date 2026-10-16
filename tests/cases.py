"""Shared test settings. Routing patterns live in sparsebench/workloads.py."""

import torch

from sparsebench.workloads import CASE_NAMES, routing_case

__all__ = ["CASE_NAMES", "DEVICES", "routing_case"]

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
