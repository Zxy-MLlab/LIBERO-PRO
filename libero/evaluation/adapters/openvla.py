"""Environment-side adapter for the official OpenVLA REST server."""

from io import BytesIO
from typing import Any, Dict

import numpy as np


class OpenVLAAdapter:
    """Translate between LIBERO values and OpenVLA's native `/act` contract."""

    PREPROCESS_SERVER = "server"
    PREPROCESS_OFFICIAL_LIBERO = "official_libero"
    _PREPROCESS_MODES = {
        PREPROCESS_SERVER,
        PREPROCESS_OFFICIAL_LIBERO,
    }

    OFFICIAL_CENTER_CROP_SCALE = 0.9

    def __init__(
        self,
        image_preprocess: str = PREPROCESS_OFFICIAL_LIBERO,
        center_crop: bool = False,
    ) -> None:
        if image_preprocess not in self._PREPROCESS_MODES:
            raise ValueError(
                "OpenVLA image_preprocess must be one of {}; got {!r}".format(
                    sorted(self._PREPROCESS_MODES),
                    image_preprocess,
                )
            )
        if not isinstance(center_crop, bool):
            raise ValueError("OpenVLA center_crop must be true or false")
        if center_crop and image_preprocess != self.PREPROCESS_OFFICIAL_LIBERO:
            raise ValueError(
                "OpenVLA center_crop requires image_preprocess=official_libero"
            )
        self.image_preprocess = image_preprocess
        self.center_crop = center_crop

    @staticmethod
    def _official_libero_image(image: np.ndarray) -> np.ndarray:
        """Apply OpenVLA's LIBERO JPEG-95 and 224px Lanczos sequence.

        Upstream uses TensorFlow for these two CPU operations.  Pillow keeps
        the environment install small and follows the same sequence, although
        the two libraries are not guaranteed to produce bit-identical pixels.
        """

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - installation guidance
            raise ImportError(
                "official_libero image preprocessing requires Pillow; install "
                "libero/evaluation/requirements.txt"
            ) from exc

        encoded = BytesIO()
        Image.fromarray(image, mode="RGB").save(
            encoded,
            format="JPEG",
            quality=95,
        )
        encoded.seek(0)
        with Image.open(encoded) as decoded:
            rgb = decoded.convert("RGB")
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            resized = rgb.resize((224, 224), resample=resampling)
            return np.array(resized, dtype=np.uint8, copy=True, order="C")

    @classmethod
    def _official_center_crop(cls, image: np.ndarray) -> np.ndarray:
        """Approximate OpenVLA's optional 0.9-area LIBERO center crop.

        Upstream applies TensorFlow's bilinear crop-and-resize after the
        JPEG/Lanczos 224px preprocessing.  Pillow keeps this CPU-side client
        lightweight while preserving the same crop geometry and interpolation
        family; exact pixels can differ slightly between libraries.
        """

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - installation guidance
            raise ImportError(
                "official_libero center cropping requires Pillow; install "
                "libero/evaluation/requirements.txt"
            ) from exc

        height, width = image.shape[:2]
        side_fraction = float(np.sqrt(cls.OFFICIAL_CENTER_CROP_SCALE))
        left = width * (1.0 - side_fraction) / 2.0
        top = height * (1.0 - side_fraction) / 2.0
        right = width - left
        bottom = height - top
        cropped = Image.fromarray(image, mode="RGB").crop(
            (left, top, right, bottom)
        )
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        resized = cropped.resize((width, height), resample=resampling)
        return np.array(resized, dtype=np.uint8, copy=True, order="C")

    def adapt_observation(
        self,
        observation: Any,
        instruction: str,
        unnorm_key: str,
    ) -> Dict[str, Any]:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("OpenVLA instruction must be a non-empty string")
        if not isinstance(unnorm_key, str) or not unnorm_key.startswith("libero_"):
            raise ValueError(
                "OpenVLA unnorm_key must be a checkpoint key starting with "
                "'libero_'"
            )

        image = np.asarray(observation.agentview_rgb)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                "OpenVLA agentview image must have RGB HWC shape; got {}".format(
                    image.shape
                )
            )

        # Official OpenVLA LIBERO evaluation rotates the simulator image by
        # 180 degrees, performs a JPEG round trip, and resizes to 224px.
        image = np.ascontiguousarray(image[::-1, ::-1], dtype=np.uint8)
        if self.image_preprocess == self.PREPROCESS_OFFICIAL_LIBERO:
            image = self._official_libero_image(image)
            if self.center_crop:
                image = self._official_center_crop(image)
        return {
            "image": image,
            "instruction": instruction,
            "unnorm_key": unnorm_key,
        }

    def actions_from_model_output(self, model_output: Any) -> np.ndarray:
        # The official deploy.py returns the NumPy action itself.  On failure it
        # returns the JSON string "error", normally with HTTP status 200.
        if isinstance(model_output, str):
            if model_output == "error":
                raise RuntimeError(
                    "OpenVLA server inference failed; inspect the GPU server logs"
                )
            raise ValueError("unexpected string returned by OpenVLA server")

        action = np.asarray(model_output, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(
                "OpenVLA server must return one action with shape [7]; got {}".format(
                    action.shape
                )
            )
        if not np.isfinite(action).all():
            raise ValueError("OpenVLA action contains NaN or infinite values")
        if action[-1] < 0.0 or action[-1] > 1.0:
            raise ValueError("OpenVLA native gripper action must be in [0, 1]")

        action = action.copy()
        # This exactly follows the official OpenVLA LIBERO post-processing:
        # map [0,1] to [-1,+1], binarize with sign(), then invert because
        # LIBERO uses -1=open and +1=close.  The first six dimensions always
        # pass through unchanged.
        action[-1] = -np.sign(2.0 * action[-1] - 1.0)
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError("OpenVLA action is outside LIBERO's [-1, 1] range")
        return np.ascontiguousarray(action[None, :], dtype=np.float32)
