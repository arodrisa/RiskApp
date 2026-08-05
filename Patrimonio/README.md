# Patrimonio

Personal/family asset-control app for Antonio, Patri/Patricia, shared positions, and family-owned company assets.

## Beta Scope

This beta supports:

- Importing the existing Excel workbooks as bootstrap data.
- Project users with owner, administrator, editor, and viewer access levels, plus CSRF protection and login rate limiting.
- Maintaining owners and assets from the dashboard.
- Creating a new dated snapshot from the previous date as a template.
- Editing positions in a selector-based snapshot table.
- Saving common/shared positions once, with position-level ownership splits.
- Viewing Antonio, Patricia, and aggregate dashboards with positions, charts, and history tables.
- Exporting a JSON backup of owners, assets, investing category config, positions, and position-level ownership splits.
- Configuring stock assets with a price provider and ticker/ID, then applying latest prices to snapshot rows.
- Entering structured market-minus-debt or company net-assets valuation line items from the snapshot table.
- Category-level `Investing_Assets` configuration for the Investing KPI and default investing chart view.
- React + TypeScript frontend with safer snapshot replacement review, dirty draft banner, row-level validation, undo remove, stock price audit/history, chart settings, dashboard deltas, largest movers, stable chart colors, line-chart hover values, and an Admin tab for audit/restore.

Still intentionally light for beta:

- For public exposure, deploy behind HTTPS and set `APP_COOKIE_SECURE=true`; private-network/VPN access is still recommended.
- Alembic migrations are present for the beta schema, but the app still keeps a development startup schema helper for compatibility with older local databases.
- Excel import is transitional. The dashboard is intended to become the main data-entry path.

## Requirements

- Python 3.8+
- Dependencies from `requirements.txt`
- Optional PostgreSQL via Docker Compose

## Install

```bash
python -m pip install -r requirements.txt
```

## Run Locally

Build the React frontend first:

```bash
cd frontend
npm install
npm run build
cd ..
```

Then run the API:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Open:

```text
http://127.0.0.1:8001/dashboard
```

Health check:

```text
http://127.0.0.1:8001/health
```

JSON backup/export:

```text
http://127.0.0.1:8001/export
```

The dashboard also has an `Export backup` button in the top toolbar.
API restore is available at `POST /restore` with `confirm_restore=true` only when `APP_RESTORE_ENABLED=true`; use it only with a verified export. The Admin tab exposes audit, price history, and guarded restore when logged in.

For frontend-only development, run FastAPI on port `8001`, then:

```bash
cd frontend
npm run dev
```

Open the Vite URL. API calls are proxied to FastAPI.

## Import Excel Data

Dry run:

```bash
python import_excel.py --path "./" --dry-run
```

Import:

```bash
python import_excel.py --path "./" --do-import
```

Import rules:

- `Patrimonio ARS.xlsx` imports Antonio positions.
- `Patrimonio Patri.xlsx` imports Patri/Patricia positions.
- `Patrimonio Comun.xlsx` is skipped.
- `Resumen` sheets are skipped.
- Rows without `Tipo Activo`, blank `NAV`, or zero `NAV` are skipped.
- Excel `NAV` is treated as total position value.

## Optional PostgreSQL

Start the database:

```bash
docker-compose up -d
```

The Docker web container runs `alembic upgrade head` before starting Uvicorn by default. Set `RUN_MIGRATIONS=false` if migrations are handled separately.

Example `DATABASE_URL`:

```text
DATABASE_URL=postgresql://patr:patrpass@db:5432/patrimony
```

Then run the API with that environment variable set.

Apply migrations before starting the app on a server:

```bash
alembic upgrade head
```

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Frontend build check:

```bash
cd frontend
npm run build
```

## Data Model Notes

- `Asset` is a catalog/dimension table.
- Stock assets can store `price_provider` and `price_symbol`; the first provider implemented is Yahoo.
- `InvestingAsset` stores whether each category is treated as invested. Cash/Casa default to not invested; other categories default to invested.
- `Position` is the dated snapshot fact table.
- Direct Antonio/Patri holdings can use `Position.owner_id`.
- Common/shared/company holdings should be ownerless positions with `PositionOwnership` split rows.
- The legacy `Ownership` table is only kept to migrate old asset-level ownership data.
