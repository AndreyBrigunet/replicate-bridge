# Replicate OpenAI Bridge

OpenAI-compatible bridge for a Replicate community model.

## What it does

- `POST /v1/chat/completions`
- `GET /v1/models`
- Replicate native SSE -> OpenAI SSE
- Smart keep-alive: real requests reset the timer
- Start/stop/status/manual-ping admin endpoints
- Persistent keep-alive enabled/disabled state

## Admin endpoints

- `GET /admin/keepalive/status`
- `POST /admin/keepalive/start`
- `POST /admin/keepalive/stop`
- `POST /admin/keepalive/ping`

## Scope

Text chat is supported. OpenAI native tool/function calling is not emulated.
