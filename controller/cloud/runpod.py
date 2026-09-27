"""RunPod's REST API, the parts the studio uses.

API v2 (https://api.runpod.io/v2/...). Not v1: RunPod retires v1 on 15 November
2026, and a machine lifecycle that stops working on a date nobody wrote down is
exactly the kind of failure that leaves a pod billing with nobody watching it.

Deliberately thin: one method per call, dicts in and out, errors raised as
`RunPodError` with the provider's own message. The decisions -- which GPU, when
to start, when to stop -- live in `controller.cloud.manager`, where they can be
tested without a network.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

BASE = os.environ.get("AI_STUDIO_RUNPOD_API", "https://api.runpod.io")
TIMEOUT = httpx.Timeout(30.0, connect=10.0)
# The CUDA runner image is built on CUDA 12.8; a host with an older driver
# starts the container and then fails the first time torch touches the card.
MIN_CUDA = "12.8"


class RunPodError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class RunPod:
    def __init__(self, api_key: str, base: str | None = None) -> None:
        self._headers = {"Authorization": "Bearer " + api_key}
        self._base = (base or BASE).rstrip("/")

    async def _call(self, method: str, path: str, **kw: Any) -> Any:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.request(method, self._base + path, headers=self._headers, **kw)
        except httpx.HTTPError as e:
            raise RunPodError(0, "RunPod could not be reached (%s)." % e.__class__.__name__) from e
        if r.status_code == 204 or not r.content:
            if r.is_success:
                return None
        try:
            body = r.json()
        except ValueError:
            body = {"message": r.text[:300]}
        if not r.is_success:
            msg = body.get("message") or body.get("error") or body.get("detail") or r.reason_phrase
            if r.status_code in (401, 403):
                msg = "RunPod refused the API key (%s). Check that it is valid and allowed to manage pods." % msg
            raise RunPodError(r.status_code, str(msg))
        return body

    # --------------------------------------------------------------- catalog
    async def gpus(self, cloud: str = "SECURE") -> list[dict]:
        """Every GPU type with its price and current stock on `cloud` (SECURE or COMMUNITY)."""
        body = await self._call("GET", "/v2/catalog/gpus", params={
            "include": "AVAILABILITY", "product": "POD", "count": 1,
            "cloud": cloud, "minCudaVersion": MIN_CUDA})
        return list((body or {}).get("gpus") or [])

    # ------------------------------------------------------------- templates
    async def list_templates(self) -> list[dict]:
        body = await self._call("GET", "/v2/templates")
        if isinstance(body, dict):
            return list(body.get("templates") or [])
        return list(body or [])

    async def create_template(self, body: dict) -> dict:
        return await self._call("POST", "/v2/templates", json=body)

    async def update_template(self, template_id: str, body: dict) -> dict:
        return await self._call("PATCH", "/v2/templates/%s" % template_id, json=body)

    # ------------------------------------------------------------------ pods
    async def create_pod(self, *, name: str, gpu_id: str, cloud: str,
                         env: dict[str, str], template_id: str | None = None,
                         image: str | None = None, disk_gb: int | None = None,
                         volume_gb: int | None = None, mount: str = "/data") -> dict:
        """A pod from the studio's template, or spelled out in full without one.

        With a template the container settings come from it, and the body
        carries only what differs per pod: the GPU, the cloud, and the env
        that must never sit in a template -- RunPod merges env per key, with
        the body's values winning.
        """
        body: dict[str, Any] = {
            "name": name,
            "cloud": cloud,
            "gpu": {"id": gpu_id, "count": 1, "minCudaVersion": MIN_CUDA},
            "env": env,
        }
        if template_id:
            body["templateId"] = template_id
        else:
            body.update({
                "image": image,
                "disk": int(disk_gb or 40),
                "mounts": {"persistent": {"size": int(volume_gb or 80), "path": mount}},
                # The runner dials out to the studio; nothing needs to reach it.
                "ports": [],
            })
        return await self._call("POST", "/v2/pods", json=body)

    async def get_pod(self, pod_id: str) -> dict | None:
        try:
            return await self._call("GET", "/v2/pods/%s" % pod_id)
        except RunPodError as e:
            if e.status == 404:
                return None
            raise

    async def list_pods(self) -> list[dict]:
        body = await self._call("GET", "/v2/pods")
        if isinstance(body, dict):
            return list(body.get("pods") or body.get("items") or [])
        return list(body or [])

    async def action(self, pod_id: str, action: str) -> None:
        await self._call("POST", "/v2/pods/%s/action" % pod_id, json={"action": action})

    async def terminate(self, pod_id: str) -> None:
        try:
            await self._call("DELETE", "/v2/pods/%s" % pod_id)
        except RunPodError as e:
            if e.status != 404:        # already gone is what we wanted
                raise
