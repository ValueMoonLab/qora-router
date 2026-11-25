# qora-router

**The OpenRouter for your own GPU.** One endpoint, many local models, one GPU.

> **Project status: frozen at v0.2.0 (2026-11).** This is a personal/learning project, kept as-is.
> An honest competitive survey found that [llama-swap](https://github.com/mostlygeek/llama-swap) (6k+ stars, MIT)
> already does on-demand local model swapping with a mature feature set (aliases, API keys, groups, TTL, peers, Web UI),
> llama.cpp ships a built-in router mode with LRU eviction, and RustingStrata itself hot-swaps models by name natively.
> For production use, prefer llama-swap or [Superfluid](https://github.com/basecompute/superfluid) (agent-aware QoS,
> park/resume, multi-machine fleets, Windows/Linux/macOS). qora-router remains useful as a zero-dependency,
> single-file glue layer for heterogeneous engines (including non-OpenAI protocols) on a home LAN. No further features are planned.

qora-router is a fully automatic router for local inference engines. Pick a model name per request and it loads, swaps, or attaches to whatever backend serves that model - including models from different engines. It switches **both models and engines** on demand, so OpenAI-compatible clients never see a "model not loaded" error.

```
Client -> POST /v1/chat/completions {"model": "qwen3.6-uncensored"}
              |
        qora-router (:8090)
              |
   +----------+------------+-------------------+------------------+
   |                       |                   |                  |
 RustingStrata          Inkling          llama.cpp / Ollama   vLLM / SGLang
 GGUF hot-swap          cold start        HTTP, zero-downtime  HTTP, zero-downtime
 ~40s (measured)        ~4min (measured)  model id changes     model id changes
```

## Why

A 12 GB GPU cannot hold two large models at once. Instead of picking one engine and sticking with it, qora-router lets you **name the model you want per request** and handles everything else automatically - no scripts, no manual load/unload, no engine restarts by hand:

- **Model switch, same RustingStrata engine** (two GGUF quants): **~40 seconds**, hot-swap via `/api/load`, no process restart.
- **Model switch, same HTTP engine** (for example two LM Studio or Ollama model ids): **~0 seconds**, only the `model` field in the request body changes.
- **Engine switch** (RustingStrata -> Inkling): **~4 minutes**, old engine stopped, VRAM released, new engine cold-started - all triggered by the model name in the request.
- **Attach to an external engine** (LM Studio, Jan, GPT4All, an already-running Ollama): no launch cost; the router never stops a server it did not start.
- Same model consecutive requests: **0 seconds**, fully transparent.

Clients see a standard OpenAI API. No special SDK, no retry logic, no "model not loaded" errors.

## Quick start

```bash
pip install tokenizers   # only needed for the Inkling backend; RS/HTTP-only setups need zero deps

cp config.example.json config.json
# edit config.json: set rs_exe / rs_models_dir (and ink_* if you use Inkling),
# or register HTTP engines under a "backends" map.

python scheduler.py --config config.json
```

List every supported engine and its launch hints:

```bash
python scheduler.py --list-engines
```

Then from any OpenAI-compatible client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8090/v1", api_key="not-needed")

r = client.chat.completions.create(
    model="qwen3.6-uncensored",
    messages=[{"role": "user", "content": "Explain compound interest in one sentence."}]
)

# Switch engine by name - the bridge queues the request while it swaps
r = client.chat.completions.create(
    model="inkling-2-mini",
    messages=[{"role": "user", "content": "Write a Python decorator for retry with backoff."}]
)
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat. Pick model by name. |
| `/v1/models` | GET | List all configured models |
| `/health` | GET | Current backend, active model, swap count, request count |
| `/engines` | GET | Supported-engine catalog as JSON (also `/v1/engines`) |

## Supported engines

`--list-engines` prints ready-to-copy launch hints for all of these, and `GET /engines` returns the same catalog as JSON. "Switch cost" is the cost of changing the model or replacing the engine; changing the model id inside one already-running HTTP server is always ~0s.

| Engine | Type | Switch cost | Notes |
|--------|------|-------------|-------|
| llama.cpp (`llama-server`) | `http` | server restart (not measured) | `/v1/chat/completions`; CUDA, Metal, Vulkan or CPU; one model per process |
| Ollama | `http` | engine-managed load/unload (not measured) | `/v1/chat/completions`; ids like `llama3.2:3b`; CUDA, ROCm, Metal or CPU |
| vLLM | `http` | server restart (not measured) | `/v1/chat/completions`; NVIDIA CUDA, ROCm on Linux |
| SGLang | `http` | server restart (not measured) | `/v1/chat/completions`; NVIDIA CUDA (Linux) |
| LMDeploy | `http` | server restart (not measured) | `/v1/chat/completions`; NVIDIA CUDA |
| LM Studio | `http` (external) | manual in GUI (not measured) | `/v1/chat/completions`; attach only; CUDA, Metal or Vulkan |
| KoboldCpp | `http` | server restart (not measured) | `/v1/chat/completions`; CPU, CUDA, ROCm or Vulkan |
| Text Generation Inference (TGI) | `http` | server restart (not measured) | `/v1/chat/completions`; CUDA, ROCm, Gaudi or CPU |
| TensorRT-LLM (`trtllm-serve`) | `http` | server restart (not measured) | `/v1/chat/completions`; NVIDIA |
| MLX (`mlx_lm.server`) | `http` | server restart (not measured) | `/v1/chat/completions`; macOS Apple Silicon only |
| LocalAI | `http` | engine-managed load (not measured) | `/v1/chat/completions`; CPU, CUDA, ROCm or Intel |
| Jan | `http` (external) | manual in app (not measured) | `/v1/chat/completions`; attach only |
| GPT4All | `http` (external) | manual in app (not measured) | `/v1/chat/completions`; attach only |
| Mistral.rs | `http` | server restart (not measured) | `/v1/chat/completions`; CUDA, Metal or CPU |
| RustingStrata | `rs` | ~40s GGUF hot-swap (measured) | Native; `/v1/chat/completions`; NVIDIA CUDA or CPU |
| Inkling | `inkling` | ~4min cold start (measured) | Native GEN stdin/stdout; requires `tokenizers` |

Only the RustingStrata and Inkling numbers were measured, on the hardware listed below. Everything else is marked "not measured" on purpose.

## Configuration

See `config.example.json`. Its `_examples` key contains copy-paste templates for llama.cpp, Ollama and vLLM; it is ignored by the scheduler.

### Backends

Backends live under a `backends` map. Each entry has a `type` of `rs`, `inkling` or `http`:

```json
{
  "backends": {
    "llamacpp": {
      "type": "http",
      "exe": "llama-server",
      "args": ["-m", "/path/to/model.gguf", "--host", "127.0.0.1",
               "--port", "8085", "-ngl", "99"],
      "base_url": "http://127.0.0.1:8085",
      "health_path": "/health",
      "startup_timeout": 180
    },
    "ollama": {
      "type": "http",
      "exe": "ollama",
      "args": ["serve"],
      "base_url": "http://127.0.0.1:11434",
      "health_path": "/api/tags"
    }
  },
  "models": {
    "llama-3.2-3b": {
      "backend": "llamacpp",
      "request": {"model": "llama-3.2-3b"}
    },
    "qwen3": {
      "backend": "ollama",
      "request": {"model": "qwen3:8b"}
    }
  }
}
```

- `type: "http"` starts `exe` with `args` and polls `base_url + health_path` until ready. If `exe` is empty the router **attaches** to an already-running server and never stops it.
- `type: "rs"` uses the native RustingStrata server and its `/api/load` hot-swap.
- `type: "inkling"` uses the native Inkling GEN stdin/stdout protocol.

Legacy flat configuration (`rs_exe`, `rs_models_dir`, `ink_exe`, ...) still works. When a backend entry has no `type`, it is inferred: a `gguf`/`models_dir` field means `rs`, a `bridge_port`/`tokenizer` field means `inkling`, and a `base_url`/`exe` field means `http`.

### Models

Each model entry is one public model name:

| Field | Meaning |
|-------|---------|
| `backend` | Backend name from `backends` (or `rs` / `inkling`). May be omitted: `gguf` implies `rs`, `bridge_port` implies `inkling`, otherwise the single configured HTTP backend is used. |
| `gguf` | GGUF filename for an `rs` model |
| `request` | For `http` models, a dict merged into the chat request body. Use `{"model": "server-model-id"}` to select the engine's model id; any other OpenAI fields here override the client defaults. |
| `description` | Shown by `/v1/models` and `/health` |

The `request` merge is what makes zero-downtime switching possible: if two public models share one HTTP backend, qora-router only changes the `model` field between requests.

Only bind GPU engines to loopback. All external traffic should go through the scheduler; otherwise two clients can load models at the same time and exhaust VRAM. External HTTP servers are never stopped by qora-router, so do not run two GPU-heavy external servers that both reserve all VRAM.

## Architecture

```
ThreadingHTTPServer
  /v1/chat/completions   /v1/models   /health   /engines
          |
          v
      Scheduler  -> threading.Lock (one swap at a time)
          |
          +-- RSBackend        HTTP /api/load hot-swap
          +-- InklingEngine    GEN stdin/stdout
          +-- HTTPBackend      any OpenAI-compatible server
                  - managed: launch exe + args, health poll, stop on switch
                  - external: attach only, never stopped
```

- **VRAM exclusivity**: for native GPU engines, only one backend is resident at a time. Switching stops the old process, waits for VRAM release, then starts the new one.
- **Zero-downtime HTTP switching**: two models on the same HTTP backend just change the request body; no process restart.
- **Request queuing**: a lock serializes swaps. Concurrent requests for the same model pass through; requests for a different model wait.
- **Attach mode**: if a server is already running at a backend's `base_url`, the scheduler uses it and leaves its lifecycle alone.

## Backends

### RustingStrata (GGUF)
- Native OpenAI HTTP server
- Hot-swap via `POST /api/load {"model": "filename.gguf"}` - measured **37-42s** on RTX 3060 12GB
- Supports any GGUF model in `--models-dir`

### Inkling (strata-inkling)
- stdin/stdout line protocol (`GEN <max_tokens> <k=v...> <comma-separated-ids>`)
- Cold start ~4 minutes (10.5 GB model load + expert streaming)
- Requires the `tokenizers` library for prompt encoding

### HTTP engines (everything else)
- Any server that implements `POST /v1/chat/completions`
- Per-model `request` body merge, for example `{"model": "llama3.2:3b"}`
- Managed mode (`exe` set) or external attach mode (`exe` empty)

## Requirements

- Python 3.10+ (pure standard library; `tokenizers` only for Inkling)
- Windows, Linux or macOS. Windows-specific process flags are guarded.
- At least one engine: RustingStrata binary, Inkling binary, or any OpenAI-compatible local server
- NVIDIA GPU with CUDA for the measured RustingStrata/Inkling numbers (tested: RTX 3060 12GB); other engines bring their own hardware requirements

## Measured performance (RTX 3060 12GB)

| Operation | Time |
|-----------|------|
| RS hot-swap (IQ2_M -> Q2_K_XL) | 37-42s |
| RS -> Inkling cold start | ~240s |
| Same-model request (RS uncensored, short answer) | 0.7-1.3s |
| Same HTTP engine, different model id (request-body only) | ~0s |
| LAN client -> scheduler -> RS | 0.7s |

## Non-goals

- No streaming (SSE) yet - responses are returned whole.
- No multi-GPU placement; one GPU, one resident GPU-heavy backend.
- No authentication - put it behind a reverse proxy / firewall if you expose it beyond localhost.

## License

Apache-2.0
