# Patrimonio Operations

## Deploy / Start

Build and start the beta stack:

```bash
docker compose up --build -d
```

Before deploying, create a `.env` from `.env.example` and set strong values for:

- `POSTGRES_PASSWORD`
- `PATRIMONIO_PASSWORD`
- `PATRIMONIO_SESSION_SECRET`

For an exposed webpage, also set:

- `APP_AUTH_ENABLED=true`
- `APP_COOKIE_SECURE=true` behind HTTPS
- `APP_COOKIE_SAMESITE=strict`
- `APP_RESTORE_ENABLED=false` unless you are intentionally restoring

Check service status:

```bash
docker compose ps
```

Check the app:

```bash
curl http://127.0.0.1:8000/health
```

The web container runs migrations automatically by default through `RUN_MIGRATIONS=true`.
For non-Docker deployments, run migrations manually before starting Uvicorn:

```bash
alembic upgrade head
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Import

Inspect Excel workbooks without importing:

```bash
python import_excel.py --path "./" --dry-run
```

Import workbooks and rebuild the development database:

```bash
python import_excel.py --path "./" --do-import
```

Import workbooks without rebuilding the database:

```bash
python import_excel.py --path "./" --do-import --keep-existing-db
```

The rebuild path is intended only for development. Production should use migrations and non-destructive imports.

## Migrations

Apply database migrations:

```bash
alembic upgrade head
```

Create a future migration after model changes:

```bash
alembic revision --autogenerate -m "describe change"
```

## Backup

Download a JSON backup:

```bash
curl -o patrimonio-export.json http://127.0.0.1:8000/export
```

The export includes owners, assets, effective investing category config, positions, ownership splits, historical asset valuations, and structured net-asset valuation line items.

For PostgreSQL-level backups:

```bash
docker compose exec db pg_dump -U patr -d patrimony > patrimonio.sql
```

## Restore

The API restore endpoint replaces the current database contents with a JSON export.
It is disabled by default in authenticated deployments. Set `APP_RESTORE_ENABLED=true` only for the restore window, use only with a verified backup, then disable it again.

```bash
curl -X POST http://127.0.0.1:8000/restore \
  -H "Content-Type: application/json" \
  -H "X-CSRF-Token: <token-from-auth-status>" \
  --data @restore-payload.json
```

Where `restore-payload.json` has this shape:

```json
{
  "confirm_restore": true,
  "backup": {
    "version": 1
  }
}
```

In practice, place the full `/export` JSON object inside `backup`.
For normal use, prefer the Admin tab restore panel after logging in; it sends the CSRF header automatically.

## Audit

Recent API mutation events:

```bash
curl http://127.0.0.1:8000/audit-log/
```

When `APP_AUTH_ENABLED=false`, audit actors are recorded as `api`. When `APP_AUTH_ENABLED=true`, audit actors use the logged-in username.

## Deployment Rehearsal Checklist

- Build: `docker compose up --build -d`
- Confirm DB and web are healthy: `docker compose ps`
- Open `/dashboard`
- Login with `PATRIMONIO_USERNAME` / `PATRIMONIO_PASSWORD` when auth is enabled
- Confirm mutating API requests without `X-CSRF-Token` are rejected when auth is enabled
- Confirm `/health` returns `{"status":"ok"}`
- Confirm `/investing-assets/` returns categories
- Create a test owner or asset, then confirm `/audit-log/`
- Confirm the Admin tab shows Audit log and Price history
- Export backup with `/export`
- Restore into a disposable database before trusting restore in production, either through `POST /restore` or the Admin tab restore panel
- Confirm dashboard totals after restore
