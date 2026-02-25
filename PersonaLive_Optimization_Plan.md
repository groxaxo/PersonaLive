# PersonaLive Optimization Plan

## Objective

Reduce online latency and jitter first, then improve throughput and startup behavior with low-risk changes.

## Files Reviewed

- `webcam/vid2vid.py`
- `webcam/vid2vid_trt.py`
- `webcam/util.py`
- `webcam/connection_manager.py`
- `src/liveportrait/motion_extractor.py`
- `src/wrapper.py`
- `src/pipelines/pipeline_pose2vid.py`
- `src/modeling/engine_model.py`
- `inference_offline.py`

## Priority Order

### P0 (highest impact)

1. Avoid cross-process CUDA tensor passing in online mode.
2. Replace input queue busy-wait with blocking queue reads.
3. Bound websocket output queue to prevent latency buildup.
4. Remove redundant pose detector calls on first online chunk.

### P1

1. Reduce unnecessary forced CUDA synchronization in TensorRT path.
2. Defer CPU conversion boundaries where possible.
3. Clean up offline startup loading pattern.

### P2

1. Attention bank memory/caching refinements.
2. `torch.compile` experiments (per GPU/driver profile).
3. Decode chunk auto-tuning.

---

## Phase 1 (Implemented)

### 1) Online IPC fix: move frame preprocess/device transfer into worker

**Files**

- `webcam/vid2vid.py`
- `webcam/vid2vid_trt.py`

**What changed**

- `accept_new_params` now enqueues CPU tensors directly.
- Worker process now stacks frames, moves batch to GPU once, normalizes (`/255`, `*2-1`), and converts HWC -> NCHW.

**Why**

- Prevents expensive cross-process GPU tensor handling.
- Reduces context/IPC overhead and improves stability under load.

### 2) Queue polling fix: remove busy-wait

**File**

- `webcam/util.py`

**What changed**

- Replaced `qsize()` + `sleep(0.01)` loop with blocking `queue.get(timeout=0.1)`.
- Preserved `prefer_latest` behavior by draining with `get_nowait()` after minimum batch arrives.

**Why**

- Lower CPU overhead and less latency jitter.

### 3) Output queue bounding

**File**

- `webcam/connection_manager.py`

**What changed**

- Added bounded output queue (`asyncio.Queue(maxsize=120)`) via `ConnectionManager(output_queue_maxsize=120)` default.

**Why**

- Prevents unbounded queue growth and stale-frame latency accumulation.

### 4) Pose extractor first-window batching

**File**

- `src/liveportrait/motion_extractor.py`

**What changed**

- `interpolate_kps_online` now runs one detector forward pass for `[ref + motion]` and slices outputs.

**Why**

- Removes redundant first-frame detector calls.

---

## Phase 2 (Implemented)

### 1) Reduce blocking sync overhead in TensorRT prefill path

**Files**

- `src/modeling/engine_model.py`
- `src/wrapper_trt.py`

**What changed**

- Added optional `synchronize` flag to `EngineModel.__call__` and `EngineModel.prefill` (default remains `True`).
- Switched non-critical TRT prefill sites in wrapper to `synchronize=False`.
- Combined first-frame prefill calls into one call in TRT wrapper.

**Why**

- Keeps correctness defaults while removing unnecessary host-side blocking in prefill-heavy paths.

### 2) Decode boundary cleanup (safe)

**Files**

- `src/wrapper.py`
- `src/pipelines/pipeline_pose2vid.py`

**What changed**

- Replaced hardcoded `0.18215` with `vae.config.scaling_factor` fallback.
- Used explicit CPU transfer path with non-blocking transfer hints at decode boundary.

**Why**

- Improves maintainability and makes decode scaling robust across VAE variants.

### 3) Lightweight runtime profiling hooks

**Files**

- `webcam/vid2vid.py`
- `webcam/vid2vid_trt.py`

**What changed**

- Added optional rolling latency telemetry controlled by env var:
  - `PERSONALIVE_PROFILE=1`
- Logs p50/p95 total loop latency plus preprocess/inference p50 every 30 loops.

**Why**

- Gives immediate observability for regression checks and tuning.

---

## Recommended Validation

1. Run online mode with `--acceleration xformers` for 3-5 minutes and check:
   - smoother FPS
   - less output queue growth
   - lower p95 latency
2. Repeat with `--acceleration tensorrt`.
3. Validate reset/restart behavior during active stream.

## Suggested Metrics

- Input queue depth over time
- Output queue depth over time
- End-to-end latency (capture -> websocket send), p50/p95/p99
- Delivered FPS
- GPU utilization and VRAM

---

## Next Phase (not implemented yet)

1. Trim unnecessary `torch.cuda.synchronize()` usage in TRT path:
   - `src/modeling/engine_model.py`
   - `src/wrapper_trt.py`
2. Refactor offline startup to reduce fragmented load path:
   - `inference_offline.py`
3. Reduce decode boundary conversions where safe:
   - `src/pipelines/pipeline_pose2vid.py`
   - `src/wrapper.py`

## Rollback Guidance

- Revert by file group if any regression appears:
  - queue/IPC changes
  - motion extractor batching
  - output queue bounding
