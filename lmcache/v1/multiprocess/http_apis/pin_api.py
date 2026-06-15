# SPDX-License-Identifier: Apache-2.0
"""HTTP endpoints to permanently pin / unpin KV cache tokens in L1.

These endpoints allow operators to prevent eviction of specific token
sequences from the L1 (CPU DRAM) cache.  Once pinned, the corresponding
KV cache chunks are excluded from LRU eviction until explicitly
unpinned.

Example usage (curl)::

    # Pin tokens
    curl -X POST http://localhost:8080/pin \\
        -H "Content-Type: application/json" \\
        -d '{
              "model_name": "meta-llama/Llama-3.1-8B",
              "world_size": 1,
              "token_ids": [1, 2, 3, 4, 5],
              "request_id": "manual-pin-001"
            }'

    # Unpin tokens
    curl -X POST http://localhost:8080/unpin \\
        -H "Content-Type: application/json" \\
        -d '{
              "model_name": "meta-llama/Llama-3.1-8B",
              "world_size": 1,
              "token_ids": [1, 2, 3, 4, 5],
              "request_id": "manual-pin-001"
            }'
"""

# Standard
from typing import Any

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey, ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey

logger = init_logger(__name__)

router = APIRouter()


class PinRequest(BaseModel):
    """HTTP request body for ``/pin`` and ``/unpin`` endpoints."""

    model_name: str
    world_size: int
    token_ids: list[int]
    request_id: str = "http-pin"
    cache_salt: str = ""
    worker_id: int | None = None


@router.post("/pin")
async def pin(request: Request, body: PinRequest) -> Any:
    """Permanently pin tokens in L1 cache.

    Pinned chunks are excluded from LRU eviction until explicitly
    unpinned via ``/unpin``.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(status_code=503, content={"error": "engine not initialized"})

    ctx = engine.context

    # --- Debug info ---
    logger.info(
        "pin request: model_name=%s, world_size=%s, worker_id=%s, "
        "request_id=%s, token_ids_len=%s, cache_salt=%r",
        body.model_name, body.world_size, body.worker_id,
        body.request_id, len(body.token_ids), body.cache_salt,
    )

    ipc_key = IPCCacheEngineKey(
        model_name=body.model_name,
        world_size=body.world_size,
        worker_id=body.worker_id,
        token_ids=tuple(body.token_ids),
        start=0,
        end=len(body.token_ids),
        request_id=body.request_id,
        cache_salt=body.cache_salt,
    )
    chunk_hashes = ctx.token_hasher.compute_chunk_hashes(body.token_ids)
    logger.info("pin: computed %d chunk hashes", len(chunk_hashes))
    if not chunk_hashes:
        return {"pinned_tokens": 0, "pinned_chunks": 0}

    obj_keys = ipc_key_to_object_keys(ipc_key, chunk_hashes)
    logger.info("pin: generated %d obj_keys", len(obj_keys))
    if obj_keys:
        sample = obj_keys[0]
        logger.info(
            "pin: first obj_key: chunk_hash=%s, model_name=%s, kv_rank=%s, cache_salt=%r",
            sample.chunk_hash.hex()[:16],
            sample.model_name,
            sample.kv_rank,
            sample.cache_salt,
        )

    # Check against L1Manager._objects
    l1_objs = ctx.storage_manager._l1_manager._objects
    logger.info("pin: L1Manager._objects has %d entries", len(l1_objs))

    # Check first few obj_keys against L1
    for i, ok in enumerate(obj_keys[:5]):
        exists = ok in l1_objs
        logger.info(
            "pin: obj_key[%d] %s in L1 (chunk_hash=%s, kv_rank=%s, cache_salt=%r)",
            i,
            "EXISTS" if exists else "NOT FOUND",
            ok.chunk_hash.hex()[:16],
            ok.kv_rank,
            ok.cache_salt,
        )

    # If first key doesn't exist, dump a few L1 keys for comparison
    if obj_keys and obj_keys[0] not in l1_objs:
        logger.info("pin: --- sample of L1 keys (first 3) ---")
        for j, (lk, lv) in enumerate(list(l1_objs.items())[:3]):
            logger.info(
                "pin: L1 key[%d]: chunk_hash=%s, model_name=%s, kv_rank=%s, cache_salt=%r",
                j,
                lk.chunk_hash.hex()[:16],
                lk.model_name,
                lk.kv_rank,
                lk.cache_salt,
            )

    hit_chunks = ctx.storage_manager.pin(obj_keys)
    pinned_tokens = hit_chunks * ctx.chunk_size

    logger.info(
        "HTTP pin: pinned %d tokens (%d chunks) for request %s",
        pinned_tokens,
        hit_chunks,
        body.request_id,
    )
    return {"pinned_tokens": pinned_tokens, "pinned_chunks": hit_chunks}


@router.post("/unpin")
async def unpin(request: Request, body: PinRequest) -> Any:
    """Release permanent pin references for tokens.

    Once all pin references for a chunk are released, it becomes
    eligible for LRU eviction again.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(status_code=503, content={"error": "engine not initialized"})

    ctx = engine.context
    ipc_key = IPCCacheEngineKey(
        model_name=body.model_name,
        world_size=body.world_size,
        worker_id=body.worker_id,
        token_ids=tuple(body.token_ids),
        start=0,
        end=len(body.token_ids),
        request_id=body.request_id,
        cache_salt=body.cache_salt,
    )

    chunk_hashes = ctx.token_hasher.compute_chunk_hashes(body.token_ids)
    if not chunk_hashes:
        return {"unpinned_chunks": 0}

    obj_keys = ipc_key_to_object_keys(ipc_key, chunk_hashes)
    ctx.storage_manager.unpin(obj_keys)

    logger.info(
        "HTTP unpin: released %d chunks for request %s",
        len(chunk_hashes),
        body.request_id,
    )
    return {"unpinned_chunks": len(chunk_hashes)}


@router.get("/pin/status")
async def pin_status(request: Request) -> Any:
    """Debug: dump L1 cache status for diagnosing pin issues."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(status_code=503, content={"error": "engine not initialized"})

    ctx = engine.context
    l1_objs = ctx.storage_manager._l1_manager._objects
    permanent_pins = ctx.storage_manager._l1_manager._permanent_pins
    sm_status = ctx.storage_manager._l1_manager.report_status()

    # Sample keys from L1
    key_samples = []
    for j, lk in enumerate(list(l1_objs.keys())[:10]):
        key_samples.append({
            "index": j,
            "chunk_hash": lk.chunk_hash.hex()[:16],
            "model_name": lk.model_name,
            "kv_rank": lk.kv_rank,
            "cache_salt": lk.cache_salt,
        })

    return {
        "l1_object_count": len(l1_objs),
        "permanent_pin_count": len(permanent_pins),
        "status": sm_status,
        "key_samples": key_samples,
    }
