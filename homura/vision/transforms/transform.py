# the design here is inspired by FAIR's fvcore
from __future__ import annotations

import random
import warnings
from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
from torchvision.transforms import functional as VF, transforms as VT, InterpolationMode

__all__ = ["TransformBase",
           "ConcatTransform",
           "GeometricTransformBase", "NonGeometricTransformBase",
           "RandomResizedCrop", "RandomCrop", "RandomRotation", "RandomHorizontalFlip", "CenterCrop", "RandomResize",
           "Normalize", "ColorJitter", "RandomGrayScale"]

TargetType = Literal["bbox", "mask"]

_get_image_size = VF.get_image_size if hasattr(VF, 'get_image_size') else VF._get_image_size


class HomuraTransformWarning(UserWarning):
    pass


class TransformBase(ABC):
    """ Base class of data augmentation transformations. Transform is expected to be used as drop-in
    replacements of torchvision's transforms. ::

    train_da = CenterCrop(224, task="segmentation") * ColorJitter(task="segmentation") + ...


    """
    supported_target_types = {"bbox", "mask"}

    def __init__(self,
                 target_type: TargetType):
        if target_type is not None and target_type not in TransformBase.supported_target_types:
            raise RuntimeError(f"Invalid target_type: {target_type}")
        self.target_type = target_type

    @staticmethod
    def ensure_tensor(t,
                      is_input: bool) -> torch.Tensor:
        # is_input may be useful for users to modify the behavior
        return t if isinstance(t, torch.Tensor) else VF.to_tensor(t)

    def __call__(self,
                 input: torch.Tensor,
                 target: torch.Tensor = None
                 ) -> torch.Tensor or tuple[torch.Tensor, torch.Tensor]:

        input = self.ensure_tensor(input, True)
        original_size = _get_image_size(input)
        if target is not None:
            target = self.ensure_tensor(target, False)

        params = self.get_params(input)
        input = self.apply_image(input, params)
        if target is None:
            if self.target_type is not None:
                warnings.warn(f"target is None, but target_type=={self.target_type}")
            return input

        if self.target_type == "bbox":
            target = self.apply_bbox(target, params, original_size)
        elif self.target_type == "mask":
            target = self.apply_mask(target, params)
        return input, target

    def __mul__(self,
                other: TransformBase
                ) -> ConcatTransform:
        """ Concat transformations in (other, self) order.

        Args:
            other: other transformation

        Returns: Concatenated transformations.

        """
        return ConcatTransform(other, self, target_type=self.target_type)

    def get_params(self,
                   image: torch.Tensor) -> Optional:
        return None

    @abstractmethod
    def apply_image(self,
                    image: torch.Tensor,
                    params
                    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def apply_coords(self,
                     coords: torch.Tensor,
                     original_wh: tuple[int, int],
                     params
                     ) -> torch.Tensor:
        # transform coordinates of shape Nx2
        pass

    def apply_bbox(self,
                   bbox: torch.Tensor,
                   params,
                   original_wh: tuple[int, int]
                   ) -> torch.Tensor:
        # see also fvcore
        # bbox: Nx4 float tensor of XYXY format in absolute coordinate

        bbox = bbox.clone()
        if bbox.dim() != 2 or bbox.size(1) != 4:
            raise ValueError(f"Invalid bbox shape, expected Nx4, but got {bbox.size()}")
        # (x0, y0, x1, y1) -> ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
        # bbox should be cpu tensor
        idx = torch.tensor([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = self.apply_coords(bbox.view(-1, 4)[:, idx].reshape(-1, 2), original_wh, params).view(-1, 4, 2)
        minxy, _ = coords.min(dim=1)
        maxxy, _ = coords.max(dim=1)

        return torch.cat((minxy, maxxy), dim=1)

    @abstractmethod
    def apply_mask(self,
                   mask: torch.Tensor,
                   params
                   ) -> torch.Tensor:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


# utils

class ConcatTransform(TransformBase):
    def __init__(self,
                 *transforms: TransformBase,
                 target_type: TargetType = None):
        super().__init__(target_type)
        self.transforms = transforms

        if target_type is not None:
            for transform in self.transforms:
                if getattr(transform, "target_type", None) != target_type:
                    warnings.warn(f"task of transform={transform} is inconsistent with others", HomuraTransformWarning)

    def __call__(self,
                 input: torch.Tensor,
                 target: torch.Tensor = None
                 ) -> (torch.Tensor, torch.Tensor):
        for transform in self.transforms:
            input, target = transform(input, target)
        return input, target

    def __repr__(self):
        # from torchvision's Compose
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string

    def apply_image(self, image: torch.Tensor, params) -> torch.Tensor:
        pass

    def apply_mask(self, mask: torch.Tensor, params) -> torch.Tensor:
        pass

    def apply_coords(self, coords: torch.Tensor, original_wh: torch.Tensor, params) -> torch.Tensor:
        pass

    def apply_bbox(self, bbox: torch.Tensor, params, original_wh: tuple[int, int]) -> torch.Tensor:
        pass


# geometric

class GeometricTransformBase(TransformBase, ABC):
    def apply_mask(self,
                   mask: torch.Tensor,
                   params
                   ) -> torch.Tensor:
        return self.apply_image(mask, params)


class RandomHorizontalFlip(GeometricTransformBase):
    def __init__(self,
                 p: float = 0.5,
                 target_type: TargetType = None
                 ):
        super().__init__(target_type)
        self.p = p

    def get_params(self, image) -> Optional:
        return random.random()

    def apply_coords(self,
                     coords: torch.Tensor,
                     original_wh,
                     params
                     ) -> torch.Tensor:
        if params < self.p:
            coords[:, 0] = original_wh[0] - coords[:, 0]
        return coords

    def apply_image(self,
                    image: torch.Tensor,
                    params
                    ) -> torch.Tensor:
        return VF.hflip(image) if params < self.p else image

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


def _crop_coods_(coords, top, left, h, w, output_h, output_w):
    # crop
    coords[:, 0] -= left
    coords[:, 1] -= top
    coords[:, 0].clamp_(0, w)
    coords[:, 1].clamp_(0, h)
    # scale
    coords[:, 0] *= output_w / w
    coords[:, 1] *= output_h / h

    return coords.round()


class RandomCrop(GeometricTransformBase):
    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=False,
                 fill=0,
                 padding_mode="constant",
                 mask_fill=255,
                 target_type: TargetType = None):
        super().__init__(target_type)
        self.size = VT._setup_size(size, "Invalid value for size (h, w)")
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode
        self.fill = fill
        self.mask_fill = mask_fill
        if self.padding is not None and self.target_type is not None:
            # when reflection padding is applied, what are the expected mask or bbox?
            raise RuntimeError("padding is unexpected for non-classification tasks")
        if self.target_type == "detection":
            warnings.warn(f"{self.__class__.__name__} expects coordinate origin is at left top. "
                          f"Inconsistency with this may cause unexpected results.",
                          HomuraTransformWarning)

    def get_params(self, image) -> tuple[int, ...]:
        return VT.RandomCrop.get_params(image, self.size)

    def __call__(self,
                 input: torch.Tensor,
                 target: torch.Tensor = None
                 ) -> (torch.Tensor, torch.Tensor):
        if self.padding is not None:
            input = VF.pad(input, self.padding, self.fill, self.padding_mode)

        if self.pad_if_needed:
            w, h = _get_image_size(input)
            eh, ew = self.size
            pw, ph = max(ew - w, 0), max(eh - h, 0)
            if pw > 0 or ph > 0:
                input = VF.pad(input, [0, 0, pw, ph], fill=self.fill)
                if self.target_type == "segmentation":
                    target = VF.pad(target, [0, 0, pw, ph], fill=self.mask_fill)

        return super().__call__(input, target)

    def apply_image(self,
                    image: torch.Tensor,
                    params
                    ) -> torch.Tensor:
        x, y, h, w = params
        return VF.crop(image, x, y, h, w)

    def apply_coords(self,
                     coords: torch.Tensor,
                     original_wh,
                     params
                     ) -> torch.Tensor:
        top, left, h, w = params
        coords = _crop_coods_(coords, top, left, h, w, self.size[0], self.size[1])
        return coords

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, pad={self.pad_if_needed})"


class RandomResize(GeometricTransformBase):
    def __init__(self,
                 min_size: int,
                 max_size: int = None,
                 target_type: TargetType = None):
        super().__init__(target_type)
        if max_size is not None and min_size > max_size:
            raise ValueError(f"Invalid size: min_size={min_size} > max_size={max_size}")
        self.min_size = min_size
        self.max_size = max_size

    def get_params(self,
                   image: torch.Tensor) -> Optional:
        if self.max_size is None:
            return self.min_size
        return random.randint(self.min_size, self.max_size)

    def apply_image(self,
                    image: torch.Tensor,
                    params
                    ) -> torch.Tensor:
        return VF.resize(image, params)

    def apply_mask(self,
                   mask: torch.Tensor,
                   params
                   ) -> torch.Tensor:
        return VF.resize(mask, params, interpolation=InterpolationMode.NEAREST)

    def apply_coords(self,
                     coords: torch.Tensor,
                     original_wh: tuple[int, int],
                     params
                     ) -> torch.Tensor:
        raise NotImplementedError()

    def __repr__(self):
        return f"RandomResize(min_size={self.min_size}, max_size={self.max_size})"


class RandomResizedCrop(GeometricTransformBase):
    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 target_type=None):
        super().__init__(target_type=target_type)
        self.size = VT._setup_size(size, "Invalid value for size (h, w)")
        self.scale = scale
        self.ratio = ratio

    def get_params(self,
                   image: torch.Tensor) -> Optional:
        return VT.RandomResizedCrop.get_params(image, self.scale, self.ratio)

    def apply_image(self,
                    image: torch.Tensor,
                    params
                    ) -> torch.Tensor:
        i, j, h, w = params
        return VF.resized_crop(image, i, j, h, w, self.size)

    def apply_mask(self,
                   mask: torch.Tensor,
                   params
                   ) -> torch.Tensor:
        i, j, h, w = params
        return VF.resized_crop(mask, i, j, h, w, self.size, interpolation=InterpolationMode.NEAREST)

    def apply_coords(self,
                     coords: torch.Tensor,
                     original_wh,
                     params
                     ) -> torch.Tensor:
        top, left, h, w = params
        coords = _crop_coods_(coords, top, left, h, w, self.size[0], self.size[1])
        return coords

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, scale={self.scale}, ratio={self.ratio})"


class RandomRotation(GeometricTransformBase):
    def __init__(self,
                 degrees,
                 fill=None,
                 mask_fill=255,
                 target_type=None):
        super().__init__(target_type=target_type)
        self.degrees = VT._setup_angle(degrees, "degrees", (2,))
        self.fill = fill
        self.mask_fill = mask_fill
        if self.target_type == "detection":
            warnings.warn("Rotated bbox may exceeds image area. Please check it carefully.", HomuraTransformWarning)

    def get_params(self,
                   image: torch.Tensor) -> Optional:
        return VT.RandomRotation.get_params(self.degrees)

    def apply_image(self,
                    image: torch.Tensor,
                    params
                    ) -> torch.Tensor:
        angle = params
        return VF.rotate(image, angle, fill=self.fill)

    def apply_mask(self,
                   mask: torch.Tensor,
                   params
                   ) -> torch.Tensor:
        angle = params
        return VF.rotate(mask, angle, fill=self.mask_fill)

    def apply_coords(self,
                     coords: torch.Tensor,
                     original_wh,
                     params
                     ) -> torch.Tensor:
        original_wh = torch.tensor(original_wh, dtype=torch.float).view(1, 2)
        rad = torch.deg2rad(torch.tensor(params, dtype=torch.float))
        # rotation matrix
        rot = torch.stack([torch.cos(rad), -torch.sin(rad), torch.sin(rad), torch.cos(rad)]).view(2, 2)
        center = original_wh / 2
        coords -= center
        coords @= rot
        coords += center
        return coords.round()

    def __repr__(self):
        return f"{self.__class__.__name__}(degrees={self.degrees})"


class CenterCrop(GeometricTransformBase):
    def __init__(self,
                 size,
                 target_type=None):
        super().__init__(target_type)
        self.size = VT._setup_size(size, "Invalid size for (h, w) for size")

    def apply_image(self,
                    image: torch.Tensor,
                    params
                    ) -> torch.Tensor:
        return VF.center_crop(image, self.size)

    def apply_coords(self,
                     coords: torch.Tensor,
                     original_wh,
                     params
                     ) -> torch.Tensor:
        w, h = original_wh
        eh, ew = self.size
        crop_top = int((h - eh + 1) * 0.5)
        crop_left = int((w - ew + 1) * 0.5)
        coords = _crop_coods_(coords, crop_top, crop_left, eh, ew, eh, ew)
        return coords

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


# non geometric

class NonGeometricTransformBase(TransformBase, ABC):
    def apply_coords(self,
                     coords: torch.Tensor,
                     original_wh: torch.Tensor,
                     params
                     ) -> torch.Tensor:
        pass

    def apply_mask(self,
                   mask: torch.Tensor,
                   params
                   ) -> torch.Tensor:
        return mask

    def apply_bbox(self,
                   bbox: torch.Tensor,
                   params,
                   original_wh: tuple[int, int]
                   ) -> torch.Tensor:
        # because no-geometric transform does not affect bounding boxes
        return bbox


class Normalize(NonGeometricTransformBase):
    def __init__(self,
                 mean: list[float],
                 std: list[float],
                 target_type: TargetType = None):
        super().__init__(target_type)
        self.mean = mean
        self.std = std

    def apply_image(self,
                    image: torch.Tensor,
                    params
                    ) -> torch.Tensor:
        return VF.normalize(image, self.mean, self.std)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class RandomGrayScale(NonGeometricTransformBase):
    def __init__(self,
                 p: float = 0.5,
                 target_type: TargetType = None):
        super().__init__(target_type)
        self._impl = VT.RandomGrayscale(p)

    def apply_image(self,
                    image: torch.Tensor,
                    params
                    ) -> torch.Tensor:
        return self._impl(image)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self._impl.p}"


class ColorJitter(NonGeometricTransformBase):
    def __init__(self,
                 brightness=0,
                 contrast=0,
                 saturation=0,
                 hue=0,
                 target_type: TargetType = None):
        super().__init__(target_type)
        self._impl = VT.ColorJitter(brightness, contrast, saturation, hue)

    def apply_image(self,
                    image: torch.Tensor,
                    params
                    ) -> torch.Tensor:
        return self._impl(image)

    def __repr__(self):
        return f"{self.__class__.__name__}(brightness={self._impl.brightness}, contrast={self._impl.contrast}, " \
               f"saturation={self._impl.saturation}, hue={self._impl.hue})"
