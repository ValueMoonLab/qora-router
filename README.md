# qora-router

**The OpenRouter for your own GPU.** One endpoint, many local models, one GPU. A fully automatic model router that runs multiple local LLM backends behind a single OpenAI-compatible API. It switches **both models and engines** on demand — GGUF hot-swap within an engine, full engine swap across backends — moving them in and out of VRAM automatically so your clients never see an error.

```
Client ──► POST /v1/chat/completions {"model": "qwen3.6-uncensored"}
              │
      qora-router (:8090)
              │
              ├── RustingStrata (GGUF, hot-swap ~40s)
              │     ├── qwen3.6-uncensored  (IQ2_M, ~100 tok/s)
              │     └── qwen3.6             (Q2_K_XL)
              │
              └── Inkling (looped MoE, cold start ~4min)
                    └── inkling-2-mini      (SQ-10, ~37 tok/s)
```

## Why

A 12 GB GPU can't hold two large models at once. Instead of picking one and sticking with it, qora-router lets you **name the model you want per-request** and handles everything else automatically — no scripts, no manual load/unload, no engine restarts by hand:

- **Model switch, same engine** (e.g. two GGUF quants): **~40 seconds**, hot-swap, no process restart
- **Engine switch** (GGUF ↔ Inkling): **~4 minutes**, old engine stopped, VRAM released, new engine cold-started — all triggered by the model name in your request
- Same model consecutive requests: **0 seconds**, fully transparent

Clients see a standard OpenAI API. No special SDK, no retry logic, no "model not loaded" errors.

## Quick start

```bash
pip install tokenizers   # only needed for the Inkling backend; RS-only setups need zero deps

cp config.example.json config.json
# edit config.json: set rs_exe / rs_models_dir (and ink_* if you use Inkling)

python scheduler.py --config config.json
```

Then from any OpenAI-compatible client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8090/v1", api_key=***

r = client.chat.completions.create(
    model="qwen3.6-uncensored",
    messages=[{"role": "user", "content": "Explain compound interest in one sentence."}]
)

# Switch model by name — the bridge queues the request while it swaps
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
| `/health` | GET | Current backend, swap count, request count |

## Configuration

See `config.example.json`. Key fields:

| Field | Meaning |
|-------|---------|
| `port` / `host` | Listen address (default `8090` / `0.0.0.0`) |
| `rs_exe` | Path to the RustingStrata binary |
| `rs_models_dir` | Directory containing your GGUF files |
| `rs_addr` | Loopback address the RS backend binds (default `127.0.0.1:8084`) |
| `rs_ctx` / `rs_parallel` | Context size / parallel slots for RS |
| `ink_exe` / `ink_model` / `ink_tokenizer` / `ink_chat_template` | Inkling engine + model paths (optional) |
| `models` | Map of public model name → backend (`rs` or `inkling`) + GGUF filename |
| `default_model` | Model used when the request omits one |
| `swap_timeout` | Max seconds to wait for a backend swap (default 300) |

Only bind the RS backend to loopback: all external traffic should go through the scheduler, otherwise two clients can load models simultaneously and exhaust VRAM.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              ThreadingHTTPServer                 │
│         /v1/chat/completions  /health           │
└────────────────────┬────────────────────────────┘
                     │
              ┌──────▼──────┐
              │  Scheduler  │  ← threading.Lock (one swap at a time)
              │  routes by  │
              │ model name  │
              └──┬───────┬──┘
                 │       │
        ┌────────▼──┐ ┌──▼────────────┐
        │ RSBackend │ │ InklingEngine │
        │ HTTP fwd  │ │ GEN protocol  │
        │ /api/load │ │ stdin/stdout  │
        └───────────┘ └───────────────┘
```

- **VRAM exclusivity**: only one backend resident at a time. Switching stops the old process, waits for VRAM release, then starts the new one.
- **Request queuing**: a lock serializes swaps. Concurrent requests for the same model pass through; requests for a different model wait.
- **Attach mode**: if a RustingStrata instance is already running on `rs_addr`, the scheduler attaches to it instead of spawning a duplicate.

## Backends

### RustingStrata (GGUF)
- Native OpenAI HTTP server
- Hot-swap via `POST /api/load {"model": "filename.gguf"}` — measured **37–42s** on RTX 3060 12GB
- Supports any GGUF model in `--models-dir`

### Inkling (strata-inkling)
- stdin/stdout line protocol (`GEN <max_tokens> <k=v...> <comma-separated-ids>`)
- Cold start ~4 minutes (10.5 GB model load + expert streaming)
- Requires the `tokenizers` library for prompt encoding

## Requirements

- Python 3.10+
- `tokenizers` (Inkling backend only)
- NVIDIA GPU with CUDA (tested: RTX 3060 12GB)
- [RustingStrata](https://github.com/RustingSword/RustingStrata) binary
- Strata Inkling engine binary (optional)

## Measured performance (RTX 3060 12GB)

| Operation | Time |
|-----------|------|
| RS hot-swap (IQ2_M ↔ Q2_K_XL) | 37–44s |
| RS → Inkling cold start | ~240s |
| Same-model request (RS uncensored, short answer) | 0.7–1.3s |
| LAN client → scheduler → RS | 0.7s |

## Non-goals

- No streaming (SSE) yet — responses are returned whole.
- No multi-GPU placement; one GPU, one resident backend.
- No authentication — put it behind a reverse proxy / firewall if you expose it beyond localhost.

## License

Apache-2.0
