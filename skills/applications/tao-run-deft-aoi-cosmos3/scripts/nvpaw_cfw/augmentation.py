"""Deterministic, bbox-safe photometric augmentation for NVPAW SFT."""

from __future__ import annotations

import hashlib
import random
from typing import Any

from PIL import Image, ImageEnhance
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor


class NVPAWAugmentationSettings(BaseModel):
    """Native ``[augmentation]`` schema for the reviewed exp40 recipe."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    enabled: bool = Field(default=True, strict=True)
    seed: int = Field(default=314159, ge=0, strict=True)
    same_on_all_images: bool = Field(default=True, strict=True)
    color_jitter_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    brightness: float = Field(default=0.1, ge=0.0, le=1.0)
    contrast: float = Field(default=0.1, ge=0.0, le=1.0)
    saturation: float = Field(default=0.1, ge=0.0, le=1.0)
    hue: float = Field(default=0.02, ge=0.0, le=0.5)
    random_crop_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    horizontal_flip_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    vertical_flip_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    text_media_order_probability: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _bbox_safe_only(self) -> "NVPAWAugmentationSettings":
        forbidden = {
            "random_crop_probability": self.random_crop_probability,
            "horizontal_flip_probability": self.horizontal_flip_probability,
            "vertical_flip_probability": self.vertical_flip_probability,
            "text_media_order_probability": self.text_media_order_probability,
        }
        enabled = {name: value for name, value in forbidden.items() if value != 0.0}
        if enabled:
            raise ValueError(f"geometry and text/media reordering must remain disabled: {enabled}")
        return self


def _seed_for(data: dict[str, Any], base_seed: int) -> int:
    payload = (
        f"{base_seed}\0{int(data.get('_nvpaw_epoch', 0))}\0"
        f"{int(data.get('_nvpaw_source_index', 0))}\0color_jitter"
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _image_blocks(conversation: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if not isinstance(conversation, list):
        return blocks
    for message in conversation:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if (
                isinstance(block, dict)
                and block.get("type") == "image"
                and isinstance(block.get("image"), Image.Image)
            ):
                blocks.append(block)
    return blocks


class NVPAWPhotometricAugmentation(Augmentor):
    """Apply one seeded color-jitter draw identically to all sample images."""

    def __init__(
        self,
        enabled: bool = True,
        seed: int = 314159,
        same_on_all_images: bool = True,
        color_jitter_probability: float = 0.5,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.02,
        random_crop_probability: float = 0.0,
        horizontal_flip_probability: float = 0.0,
        vertical_flip_probability: float = 0.0,
        text_media_order_probability: float = 0.0,
    ) -> None:
        super().__init__(input_keys=["conversation"])
        self.settings = NVPAWAugmentationSettings.model_validate(
            {
                "enabled": enabled,
                "seed": seed,
                "same_on_all_images": same_on_all_images,
                "color_jitter_probability": color_jitter_probability,
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "hue": hue,
                "random_crop_probability": random_crop_probability,
                "horizontal_flip_probability": horizontal_flip_probability,
                "vertical_flip_probability": vertical_flip_probability,
                "text_media_order_probability": text_media_order_probability,
            }
        )

    def _parameters(self, data: dict[str, Any]) -> tuple[bool, float, float, float, float, list[int]]:
        settings = self.settings
        rng = random.Random(_seed_for(data, settings.seed))
        return (
            rng.random() < settings.color_jitter_probability,
            rng.uniform(max(0.0, 1.0 - settings.brightness), 1.0 + settings.brightness),
            rng.uniform(max(0.0, 1.0 - settings.contrast), 1.0 + settings.contrast),
            rng.uniform(max(0.0, 1.0 - settings.saturation), 1.0 + settings.saturation),
            rng.uniform(-settings.hue, settings.hue),
            rng.sample(range(4), k=4),
        )

    @staticmethod
    def _apply(
        image: Image.Image,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        order: list[int],
    ) -> Image.Image:
        def adjust_hue(value: Image.Image) -> Image.Image:
            if hue == 0.0:
                return value
            hue_channel, saturation_channel, value_channel = value.convert("HSV").split()
            shift = int(round(hue * 255.0))
            hue_channel = hue_channel.point(
                [(channel_value + shift) % 256 for channel_value in range(256)]
            )
            return Image.merge(
                "HSV", (hue_channel, saturation_channel, value_channel)
            ).convert("RGB")

        operations = (
            lambda value: ImageEnhance.Brightness(value).enhance(brightness),
            lambda value: ImageEnhance.Contrast(value).enhance(contrast),
            lambda value: ImageEnhance.Color(value).enhance(saturation),
            adjust_hue,
        )
        for operation_index in order:
            image = operations[operation_index](image)
        return image

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        settings = self.settings
        if not settings.enabled:
            return data
        if not settings.same_on_all_images:
            raise ValueError("same_on_all_images must be true for the reviewed recipe")
        apply_jitter, brightness, contrast, saturation, hue, order = self._parameters(data)
        if not apply_jitter:
            return data
        for block in _image_blocks(data.get("conversation")):
            block["image"] = self._apply(
                block["image"], brightness, contrast, saturation, hue, order
            )
        return data


_TOML_HOOK_MARKER = "_nvpaw_augmentation_toml_hook_installed"


def install_augmentation_toml_hook() -> None:
    """Map the native TOML section into the processor's augmentation LazyCall."""

    from cosmos_framework.configs.toml_config import sft_config, toml_config_helper

    if getattr(sft_config, _TOML_HOOK_MARKER, False):
        return
    upstream_schema = sft_config.SFTExperimentConfig

    class NVPAWSFTExperimentConfig(upstream_schema):
        augmentation: NVPAWAugmentationSettings = Field(
            default_factory=NVPAWAugmentationSettings
        )

    NVPAWSFTExperimentConfig.__name__ = "NVPAWSFTExperimentConfig"
    sft_config.SFTExperimentConfig = NVPAWSFTExperimentConfig
    toml_config_helper.PATH_REMAPS["vlm"][("augmentation",)] = (
        "dataloader_train",
        "processor",
        "augmentation",
    )
    setattr(sft_config, _TOML_HOOK_MARKER, True)
