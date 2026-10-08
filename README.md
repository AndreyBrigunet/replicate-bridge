# Replicate OpenAI Bridge v1.1.0

- Docker startup sends zero requests to Replicate.
- Keepalive always starts OFF.
- `POST /admin/keepalive/start` enables keepalive and schedules a warm-up ping.
- Real `/v1/chat/completions` traffic works independently of keepalive and never enables it.
- Real requests have priority over automatic keepalive.
- Real 429 responses are retried after Replicate's cooldown.
- Keepalive 429 responses are skipped rather than retried aggressively.
- Predictions are created asynchronously and polled to completion; `Prefer: wait` is not used.

Endpoints:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /admin/keepalive/status`
- `POST /admin/keepalive/start`
- `POST /admin/keepalive/stop`
- `POST /admin/keepalive/ping`


Keepalive
```bash
chmod +x keepalive
```

```bash
./keepalive start
./keepalive stop
./keepalive status
```
