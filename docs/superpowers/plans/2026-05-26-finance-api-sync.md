# Finance API Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a finance API module to the existing Render FastAPI service so the budget planner can persist one shared household snapshot in Render Postgres.

**Architecture:** Keep the existing chatbot/SEO/calendar app as the only Render web service. Add a separate `finance_api` package mounted from `main.py` under `/finance`, backed by Postgres through `DATABASE_URL`, with PIN-based login issuing signed bearer tokens. The frontend keeps IndexedDB as fallback and syncs the same app snapshot through the new API when configured.

**Tech Stack:** FastAPI, Pydantic, itsdangerous, psycopg 3, pytest, React, Vite, TypeScript, IndexedDB fallback.

---

## File Structure

- `finance_api/__init__.py`: package marker and exported router.
- `finance_api/config.py`: finance settings from environment.
- `finance_api/db.py`: Postgres connection, schema initialization, and test-friendly connection override.
- `finance_api/auth.py`: PIN hash verification and signed token helpers.
- `finance_api/models.py`: request/response models for login and snapshot sync.
- `finance_api/repository.py`: household/snapshot SQL operations.
- `finance_api/router.py`: `/finance` routes.
- `tests/test_finance_api.py`: API behavior tests with an isolated SQLite-compatible fake repository where needed or mocked repository calls.
- `main.py`: include the finance router and run finance schema initialization at startup without blocking the existing app.
- `requirements.txt`: add `psycopg[binary]` and `pytest`.
- Frontend `src/sync/*`: API client and sync repository adapter.
- Frontend `src/App.tsx`: hydrate from remote when configured, save to remote after local updates.

## Backend Tasks

### Task 1: Finance Auth and Models

**Files:**
- Create: `finance_api/auth.py`
- Create: `finance_api/config.py`
- Create: `finance_api/models.py`
- Test: `tests/test_finance_api.py`

- [ ] Write a failing test named `test_finance_login_accepts_configured_pin_and_returns_token` that sets `FINANCE_PIN=109428`, posts `{"pin": "109428"}` to `/finance/auth/login`, and expects a JSON token string.
- [ ] Write a failing test named `test_finance_login_rejects_wrong_pin` that posts `{"pin": "000000"}` and expects HTTP 401.
- [ ] Implement `FinanceSettings` with `pin`, `token_secret`, `token_max_age_seconds`, and `database_url`.
- [ ] Implement `create_access_token`, `verify_access_token`, and `verify_pin`.
- [ ] Run `python -m pytest tests/test_finance_api.py -q` and confirm these tests pass.

### Task 2: Postgres Snapshot Repository

**Files:**
- Create: `finance_api/db.py`
- Create: `finance_api/repository.py`
- Test: `tests/test_finance_api.py`

- [ ] Write a failing test named `test_snapshot_requires_bearer_token` that calls `GET /finance/snapshot` with no auth and expects HTTP 401.
- [ ] Write a failing test named `test_snapshot_round_trip_persists_latest_snapshot` that logs in, `PUT`s a sample snapshot, then `GET`s it back with matching `snapshot`, `revision`, and `updatedAt`.
- [ ] Implement schema initialization:
  - `finance_households(id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now())`
  - `finance_snapshots(household_id TEXT PRIMARY KEY REFERENCES finance_households(id), snapshot JSONB NOT NULL, revision INTEGER NOT NULL DEFAULT 1, updated_at TIMESTAMPTZ NOT NULL DEFAULT now())`
- [ ] Implement `get_snapshot` and `save_snapshot` in `finance_api/repository.py`.
- [ ] Run `python -m pytest tests/test_finance_api.py -q` and confirm the repository-backed tests pass.

### Task 3: Mount API Without Breaking Existing App

**Files:**
- Modify: `main.py`
- Modify: `requirements.txt`
- Test: `tests/test_finance_api.py`

- [ ] Write a failing test named `test_finance_health_reports_database_configuration` that calls `/finance/health` and expects `{"ok": true, "databaseConfigured": false}` when `DATABASE_URL` is absent.
- [ ] Include the finance router in `main.py`.
- [ ] Call finance schema initialization during startup only when `DATABASE_URL` is set; log and continue if it fails so chatbot/SEO routes remain alive.
- [ ] Add `psycopg[binary]` and `pytest` to `requirements.txt`.
- [ ] Run `python -m pytest tests/test_finance_api.py -q`.

## Frontend Tasks

### Task 4: API Client and Store Sync

**Files:**
- Create: `src/sync/financeApi.ts`
- Create: `src/sync/financeApi.test.ts`
- Modify: `src/App.tsx`

- [ ] Write a failing Vitest test named `logs in and stores a remote finance token` using a mocked `fetch`.
- [ ] Write a failing Vitest test named `loads and saves a remote snapshot with bearer auth` using a mocked `fetch`.
- [ ] Implement `loginToFinanceApi`, `loadRemoteSnapshot`, `saveRemoteSnapshot`, and local token persistence.
- [ ] Update `App.tsx` so `VITE_FINANCE_API_BASE_URL` enables remote load/save while preserving IndexedDB fallback.
- [ ] Run `npm.cmd run test -- --run src/sync/financeApi.test.ts src/App.test.tsx`.

### Task 5: Deploy and Verify

**Files:**
- Backend Render env vars.
- Frontend built `finances/` folder in `Alcuin_Abraham_Resume_Website`.

- [ ] Set Render env vars on the existing service:
  - `DATABASE_URL` from `budget-planner-db`.
  - `FINANCE_PIN=109428`.
  - `FINANCE_TOKEN_SECRET` as a generated random secret.
  - `FINANCE_ALLOWED_ORIGINS=https://alcuinabraham.com,http://127.0.0.1:5173`.
- [ ] Push backend branch and deploy existing Render service.
- [ ] Build frontend with `VITE_FINANCE_API_BASE_URL=https://ai-international-services-chatbot.onrender.com/finance`.
- [ ] Copy the build into the frontend repo `finances/`, commit, and push.
- [ ] Smoke test live:
  - `/finance/health` returns ok.
  - login with PIN `109428` succeeds.
  - saving a budget planner snapshot survives page refresh on another browser context.
