"""Rented GPUs (RunPod): settings, the offers with estimates, the pods, the spend.

Administrators only, all of it: it spends money on the studio's account. The
decisions themselves are made by `controller.cloud.manager`; these routes show
them and let a person start or stop a pod by hand.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Request

from .. import db
from ..cloud import manager
from ..cloud.runpod import RunPodError
from .security import require_admin

router = APIRouter(prefix="/api/cloud")
FLEET = None                     # set by the app, as for the other routers


@router.get("/machine")
async def machine(request: Request) -> dict:
    """The cloud as one choosable machine, for the run wizard. Any signed-in user.

    Only what planning needs -- the largest GPU on offer and the price cap --
    never the key, the spend or the pods.
    """
    from .security import current_user
    current_user(request)
    return {"machine": await manager.virtual_runner()}


@router.get("")
async def status(request: Request) -> dict:
    require_admin(request)
    return manager.MANAGER.status()


@router.put("/settings")
async def put_settings(request: Request, payload: dict = Body(...)) -> dict:
    user = require_admin(request)
    try:
        return manager.save_settings(payload, user["id"])
    except (TypeError, ValueError) as e:
        raise HTTPException(400, str(e)) from e


@router.get("/gpus")
async def gpus(request: Request, cloud: str = "any", job: str = "") -> dict:
    """What can be rented now, and -- for a run -- whether it fits and what it would cost."""
    require_admin(request)
    s = manager.settings()
    if not s["api_key_set"]:
        return {"gpus": [], "note": "Add a RunPod API key to see what can be rented."}
    try:
        offers = await manager.offers(cloud if cloud in manager.POLICIES else "any", refresh=True)
    except RunPodError as e:
        raise HTTPException(502, "RunPod: %s" % e) from e
    j = db.get_job(job) if job else None
    for o in offers:
        o["within_price_cap"] = o["price"] <= s["max_price_per_hour"]
        if j and FLEET is not None:
            ok, why = manager.fits(FLEET, j, o)
            o["fits"], o["why_not"] = ok, why
            if ok:
                o["estimate"] = manager.estimate(j, o)
    offers.sort(key=lambda o: (o["cloud"], -o["speed"], o["price"]))
    return {"gpus": offers, "spend_today": round(manager.spend(), 2), "daily_cap": s["daily_cap"]}


@router.post("/pods")
async def start_pod(request: Request, payload: dict = Body(...)) -> dict:
    user = require_admin(request)
    try:
        return await manager.MANAGER.start_by_hand(str(payload.get("gpu_id") or ""),
                                                   str(payload.get("cloud") or "SECURE").upper(), user["id"])
    except (ValueError, RunPodError) as e:
        raise HTTPException(400, str(e)) from e


@router.post("/pods/{pod_id}/drain")
async def drain_pod(request: Request, pod_id: str) -> dict:
    """Give a pod back the careful way: checkpoints to the studio first, then delete."""
    require_admin(request)
    try:
        await manager.MANAGER.drain_by_hand(pod_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if FLEET is not None:
        FLEET.wake()
    return {"ok": True}


@router.delete("/pods/{pod_id}")
async def delete_pod(request: Request, pod_id: str) -> dict:
    """Delete at once. Checkpoints still on the pod are lost -- for a stuck pod."""
    require_admin(request)
    try:
        await manager.MANAGER.terminate_now(pod_id)
    except (ValueError, RunPodError) as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True}
