"""PyTorch model for AI-VAD model implementation.

Paper https://arxiv.org/pdf/2212.00789.pdf
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor, nn

from anomalib.models.ai_vad.density import CombinedDensityEstimator
from anomalib.models.ai_vad.features import FeatureExtractor
from anomalib.models.ai_vad.flow import FlowExtractor
from anomalib.models.ai_vad.regions import RegionExtractor


class AiVadModel(nn.Module):
    """AI-VAD model.

    Args:
        box_score_thresh (float): Confidence threshold for region extraction stage.
        persons_only (bool): When enabled, only regions labeled as person are included.
        min_bbox_area (int): Minimum bounding box area. Regions with a surface area lower than this value are excluded.
        max_bbox_overlap (float): Maximum allowed overlap between bounding boxes.
        enable_foreground_detections (bool): Add additional foreground detections based on pixel difference between
            consecutive frames.
        foreground_kernel_size (int): Gaussian kernel size used in foreground detection.
        foreground_binary_threshold (int): Value between 0 and 255 which acts as binary threshold in foreground
            detection.
        n_velocity_bins (int): Number of discrete bins used for velocity histogram features.
        use_velocity_features (bool): Flag indicating if velocity features should be used.
        use_pose_features (bool): Flag indicating if pose features should be used.
        use_deep_features (bool): Flag indicating if deep features should be used.
        n_components_velocity (int): Number of components used by GMM density estimation for velocity features.
        n_neighbors_pose (int): Number of neighbors used in KNN density estimation for pose features.
        n_neighbors_deep (int): Number of neighbors used in KNN density estimation for deep features.
    """

    def __init__(
        self,
        # region-extraction params
        box_score_thresh: float = 0.8,
        persons_only: bool = False,
        min_bbox_area: int = 100,
        max_bbox_overlap: float = 0.65,
        enable_foreground_detections: bool = True,
        foreground_kernel_size: int = 3,
        foreground_binary_threshold: int = 18,
        # feature-extraction params
        n_velocity_bins: int = 8,
        use_velocity_features: bool = True,
        use_pose_features: bool = True,
        use_deep_features: bool = True,
        # density-estimation params
        n_components_velocity: int = 5,
        n_neighbors_pose: int = 1,
        n_neighbors_deep: int = 1,
    ):
        super().__init__()
        if not any((use_velocity_features, use_pose_features, use_deep_features)):
            raise ValueError("Select at least one feature type.")

        # initialize flow extractor
        self.flow_extractor = FlowExtractor()
        # initialize region extractor
        self.region_extractor = RegionExtractor(
            box_score_thresh=box_score_thresh,
            persons_only=persons_only,
            min_bbox_area=min_bbox_area,
            max_bbox_overlap=max_bbox_overlap,
            enable_foreground_detections=enable_foreground_detections,
            foreground_kernel_size=foreground_kernel_size,
            foreground_binary_threshold=foreground_binary_threshold,
        )
        # initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            n_velocity_bins=n_velocity_bins,
            use_velocity_features=use_velocity_features,
            use_pose_features=use_pose_features,
            use_deep_features=use_deep_features,
        )
        # initialize density estimator
        self.density_estimator = CombinedDensityEstimator(
            use_velocity_features=use_velocity_features,
            use_pose_features=use_pose_features,
            use_deep_features=use_deep_features,
            n_components_velocity=n_components_velocity,
            n_neighbors_pose=n_neighbors_pose,
            n_neighbors_deep=n_neighbors_deep,
        )

    def forward(self, batch: Tensor) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Forward pass through AI-VAD model.

        Args:
            batch (Tensor): Input image of shape (N, L, C, H, W)
        Returns:
            list[Tensor]: List of bbox locations for each image.
            list[Tensor]: List of per-bbox anomaly scores for each image.
            list[Tensor]: List of per-image anomaly scores.
        """
        self.flow_extractor.eval()
        self.region_extractor.eval()
        self.feature_extractor.eval()

        # 1. get first and last frame from clip
        first_frame = batch[:, 0, ...]
        last_frame = batch[:, -1, ...]

        # 2. extract flows and regions
        with torch.no_grad():
            flows = self.flow_extractor(first_frame, last_frame)
            regions = self.region_extractor(first_frame, last_frame)

        # 3. extract pose, appearance and velocity features
        features_per_batch = self.feature_extractor(first_frame, flows, regions)

        if self.training:
            return features_per_batch

        # 4. estimate density
        box_scores = []
        image_scores = []
        for features in features_per_batch:
            box, image = self.density_estimator(features)
            box_scores.append(box)
            image_scores.append(image)

        box_locations = [batch_item["boxes"] for batch_item in regions]
        return box_locations, box_scores, image_scores
