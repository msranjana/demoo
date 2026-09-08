"""Smoke/fire detection via a llama-server GGUF vision model."""

import json
import urllib.error
import urllib.request

import config
from detectors.vlm_smoke_fire_base import VlmSmokeFireDetectorBase


class GgufVlmSmokeFireDetector(VlmSmokeFireDetectorBase):
    """Calls a running llama-server instance with a GGUF vision model."""

    def __init__(
        self,
        model_id=None,
        label=None,
        server_url=None,
        timeout=None,
        **kwargs,
    ):
        super().__init__(
            model_id=model_id or config.GGUF_MODEL_ID,
            label=label or "gguf-smolvlm2-256m",
            **kwargs,
        )
        self.server_url = (server_url or config.GGUF_SERVER_URL).rstrip("/")
        self.timeout = timeout or config.GGUF_TIMEOUT
        self._endpoint = f"{self.server_url}/v1/chat/completions"

    def on_start(self):
        self.log(
            f"using GGUF model {self.model_id} via {self.server_url} "
            f"(start with: llama-server -hf {self.model_id})"
        )
        self._check_server()

    def _check_server(self):
        try:
            with urllib.request.urlopen(f"{self.server_url}/health", timeout=5) as response:
                if response.status != 200:
                    self.log(f"llama-server health check returned {response.status}", level="WARNING")
        except urllib.error.URLError as error:
            self.log(
                f"llama-server not reachable at {self.server_url}: {error}. "
                "GGUF detector will retry on each frame.",
                level="WARNING",
            )

    def _ask(self, frame):
        image_base64 = self.to_base64(frame)
        if not image_base64:
            return ""

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "max_tokens": 32,
            "temperature": 0,
        }

        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))

        choices = body.get("choices") or []
        if not choices:
            return ""

        message = choices[0].get("message") or {}
        return (message.get("content") or "").strip()
