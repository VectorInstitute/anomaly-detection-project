"""Implementatation of the AI-VAD Model.

AI-VAD: Accurate and Interpretable Video Anomaly Detection

Paper https://arxiv.org/pdf/2212.00789.pdf
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import AiVad, AiVadLightning

__all__ = ["AiVad", "AiVadLightning"]
