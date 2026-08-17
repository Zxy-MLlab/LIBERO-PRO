import numpy as np

from libero.evaluation.live_preview import LivePreviewServer, encode_rgb_png


def test_png_encoder_and_preview_interface():
    image = np.zeros((4, 5, 3), np.uint8)
    assert encode_rgb_png(image).startswith(b"\x89PNG\r\n\x1a\n")
    preview = LivePreviewServer(host="127.0.0.1", port=0, refresh_hz=10)
    preview.publish(agentview_rgb=image, wrist_rgb=image, status={"phase": "test"})
    assert preview._state.status()["phase"] == "test"
