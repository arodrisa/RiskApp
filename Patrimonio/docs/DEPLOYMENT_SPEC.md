# Patrimonio Deployment Specification

This document describes the current beta architecture, runtime requirements, backend API, frontend behavior, data model, and deployment assumptions for preparing a server deployment.

## Application Overview

Patrimonio is a snapshot-based personal and family asset-control app. It tracks dated asset positions for Antonio, Patricia/Patri, common/shared holdings, and family-owned companies.

The Excel files are only a bootstrap import source. The intended future workflow is to maintain owners, assets, prices, ownership splits, and dated snapshots through the web interface.

## Runtime Stack

- Backend: FastAPI
- ASGI server: Uvicorn
- ORM: SQLAlchemy 1.4
- Schema validation: Pydantic 1.x
- Database: SQLite by default, PostgreSQL supported
- Frontend: React + TypeScript built with Vite and served by FastAPI
- Excel import: `openpyxl`
- Price provider: Yahoo chart endpoint through Python standard library HTTP calls

Python dependencies are pinned in `requirements.txt`.

## Process Model

Default command:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The included `Dockerfile` uses:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The Docker entrypoint runs `alembic upgrade head` before Uvicorn when `RUN_MIGRATIONS=true` (the default). It retries migrations while the database is starting.

The app exposes:

- `/` -> dashboard HTML
- `/dashboard` -> dashboard HTML
- `/ui/...` -> built React static assets
- `/health` -> health check
- `/audit-log/` -> recent mutation audit entries
- `/restore` -> replace database contents from a JSON backup, requires confirmation
- `/investing-assets/` -> category-level investing classification
- `/auth/status`, `/auth/login`, `/auth/logout` -> cookie-session authentication
- `/docs` -> FastAPI OpenAPI UI
- `/openapi.json` -> OpenAPI schema

## Environment Variables

### `DATABASE_URL`

Optional. Defaults to:

```text
sqlite:///./patrimonio.db
```

PostgreSQL example:

```text
postgresql://patr:patrpass@db:5432/patrimony
```

For server deployment, PostgreSQL is preferred over SQLite if multiple users, backups, container replacement, or reliable persistence are required.

### `RUN_MIGRATIONS`

Optional. Defaults to `true` in the Docker entrypoint.

- `true`: run `alembic upgrade head` before starting Uvicorn.
- `false`: skip automatic migrations, useful when migrations are run separately by release automation.

### Authentication Environment Variables

- `APP_AUTH_ENABLED`: set `true` for deployed environments.
- `PATRIMONIO_USERNAME`: admin username.
- `PATRIMONIO_PASSWORD`: admin password.
- `PATRIMONIO_SESSION_SECRET`: long random secret used to sign session cookies.
- `PATRIMONIO_SESSION_TTL_SECONDS`: session lifetime, default `43200`.
- `APP_COOKIE_SECURE`: set `true` when serving over HTTPS.
- `APP_COOKIE_SAMESITE`: cookie SameSite policy, defaults to `strict`.
- `APP_LOGIN_RATE_LIMIT_ATTEMPTS`: failed login attempts allowed per client/user window, default `5`.
- `APP_LOGIN_RATE_LIMIT_WINDOW_SECONDS`: login rate-limit window, default `300`.
- `APP_RESTORE_ENABLED`: set `true` only when restore must be available; defaults to disabled for authenticated deployments.

## Docker

The repository includes:

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

The Docker image builds the React frontend in a Node stage, copies `frontend/dist` into the Python image, and serves it through FastAPI.
The image entrypoint applies Alembic migrations before starting the web process unless `RUN_MIGRATIONS=false`.
The Docker build context excludes local databases, Excel workbooks, Python caches, and frontend build/dependency folders.

Current Compose services:

- `db`: PostgreSQL 15
- `web`: FastAPI app on host port `8000`

Persistent database volume:

```text
db_data:/var/lib/postgresql/data
```

Current Compose database credentials:

```text
POSTGRES_USER=patr
POSTGRES_PASSWORD=patrpass
POSTGRES_DB=patrimony
```

These should be replaced with server-managed secrets before production use.

Compose includes:

- a PostgreSQL `pg_isready` health check
- a web `/health` health check
- `web.depends_on.db.condition=service_healthy`

For deployment, replace all `change-me` fallback values with server-managed secrets or environment variables.

## Database Schema

### `owners`

Stores owner dimension records.

- `id`: integer primary key
- `name`: unique owner name
- `type`: `person`, `company`, or another string value

Owners can be deleted only when they have no positions or ownership splits.

### `assets`

Stores asset catalog/dimension records. Ownership is intentionally not stored here because the same asset can be held by Antonio, Patricia, or common/shared accounts.

- `id`: integer primary key
- `name`: asset name
- `category`: imported from Excel `Tipo Activo`
- `asset_type`: examples include `stock`, `fund`, `bond`, `cash`, `real_estate`, `company`
- `valuation_method`: examples include `market_direct`, `price_provider`, `market_minus_debt`, `company_net_assets`
- `price_provider`: `manual` or `yahoo`
- `price_symbol`: ticker or provider identifier, for example `SAN.MC`
- `is_investment`: legacy/internal field retained for older data compatibility; the beta UI uses category-level `investing_assets`
- `is_shared`: legacy/helper boolean
- `created_at`: creation timestamp

Assets can be deleted only if related records do not block deletion at database/API level.

### `investing_assets`

Stores category-level investing configuration for the Investing KPI and default investing chart visibility.

- `category`: string primary key matching `assets.category`
- `is_invested`: boolean

Default behavior when no override row exists:

- Not invested: `cash`, `caja`, `efectivo`, `Casa`
- Invested: all other categories

The React Data Entry tab exposes this as `Investing_Assets`. Asset rows do not expose an investing selector.

### `positions`

Stores dated snapshot fact rows.

- `id`: integer primary key
- `asset_id`: foreign key to `assets`
- `owner_id`: nullable foreign key to `owners`
- `as_of_date`: snapshot date
- `quantity`: quantity held
- `value`: total position value/NAV in EUR-equivalent reporting terms
- `broker`: broker/bank/account/source label
- `source`: `import`, `manual`, or another source string
- `created_at`: creation timestamp

Direct owner positions use `owner_id`. Common/shared/company-attributable positions use `owner_id = null` plus `position_ownership` split rows.

### `position_ownership`

Stores position-level ownership splits.

- `position_id`: foreign key to `positions`
- `owner_id`: foreign key to `owners`
- `share`: decimal share between `0` and `1`

Total split shares cannot exceed `1.0`. If a common/shared position has no explicit split, the backend can default to family ownership shares.

### `asset_valuations`

Historical valuation table kept in the model.

- `id`: integer primary key
- `asset_id`: foreign key to `assets`
- `as_of_date`: valuation date
- `value`: valuation amount
- `source`: valuation source
- `created_at`: creation timestamp

Current dashboard snapshot workflows primarily use `positions.value`.

### `company_valuation_items`

Stores company net-assets valuation line items for company-style assets.

- `id`: integer primary key
- `asset_id`: foreign key to `assets`
- `as_of_date`: valuation date
- `item_type`: `asset` or `liability`
- `name`: line item name
- `amount`: line item amount
- `created_at`: creation timestamp

Net valuation is calculated as:

```text
sum(asset amounts) - sum(liability amounts)
```

### `ownership`

Legacy asset-level ownership table retained only to migrate older databases. New data should use `position_ownership`.

## Schema Creation And Migrations

For deployment, apply Alembic migrations before starting the app:

```bash
alembic upgrade head
```

The current migration chain is:

- `0001_initial_schema`
- `0002_beta_schema_updates`

The app still keeps development compatibility checks at startup:

1. `models.Base.metadata.create_all(bind=engine)` creates missing tables.
2. `ensure_beta_schema(engine)` adds beta columns such as `assets.price_provider` and `assets.price_symbol` if missing.
3. `migrate_legacy_asset_ownership_to_positions()` migrates legacy ownership into position-level ownership.

Those startup helpers are useful for old local SQLite databases, but server deployments should treat Alembic as the canonical schema path.

## Backend API

### System

- `GET /health`
  - Returns `{ "status": "ok" }`.

- `GET /auth/status`
  - Returns whether authentication is enabled, whether the current cookie is authenticated, the CSRF token for the current session, and whether restore is enabled.

- `POST /auth/login`
  - Body: `username`, `password`.
  - Rate limited by client/user.
  - Sets an HTTP-only session cookie and a CSRF cookie on success.

- `POST /auth/logout`
  - Clears the session cookie.
  - Requires `X-CSRF-Token` when authentication is enabled.

- `GET /export`
  - Downloads a JSON backup of owners, assets, effective investing category config, positions, and ownership splits.

- `POST /restore`
  - Replaces database contents from a version `1` JSON backup.
  - Body: `confirm_restore=true`, `backup`.
  - Requires `APP_RESTORE_ENABLED=true`.
  - Requires `X-CSRF-Token` when authentication is enabled.

- `GET /audit-log/`
  - Lists recent API mutation audit entries.

### Owners

- `GET /owners/`
  - Lists owners.

- `POST /owners/`
  - Creates an owner.
  - Body: `name`, optional `type`.

- `PUT /owners/{owner_id}`
  - Updates owner fields.

- `DELETE /owners/{owner_id}`
  - Deletes an owner only when there are no dependent positions or splits.

### Assets

- `GET /assets/`
  - Lists assets.

- `POST /assets/`
  - Creates an asset catalog record.
  - Rejects duplicate asset names for manual API entry.

- `GET /assets/duplicates`
  - Lists existing duplicate asset-name groups for cleanup/review.

- `PUT /assets/{asset_id}`
  - Updates an asset catalog record.

- `DELETE /assets/{asset_id}`
  - Deletes an asset.

### Investing Assets

- `GET /investing-assets/`
  - Lists all known categories with their effective `is_invested` value.
  - Categories without stored overrides are returned with the backend default.

- `PUT /investing-assets/{category}`
  - Creates or updates one category override.
  - Body: `is_invested`.

- `GET /assets/{asset_id}/company-valuation?as_of_date=YYYY-MM-DD`
  - Returns persisted asset/liability line items and calculated net value.

- `PUT /assets/{asset_id}/company-valuation`
  - Replaces company valuation line items for one asset/date.
  - Body: `as_of_date`, `items`.

### Prices

- `GET /prices/quote`
  - Parameters:
    - `asset_id`, or
    - `provider` plus `symbol`
  - Returns provider, symbol, latest price, currency, and quote date/time when available.
  - `manual` provider does not return live prices.
  - `yahoo` requires outbound internet access from the server.
  - Successful asset-based quote lookups are stored in `price_history`.

- `GET /prices/history`
  - Lists stored provider quote history.

### Positions

- `GET /positions/`
  - Optional parameters: `as_of_date`, `asset_id`, `skip`, `limit`.

- `POST /positions/`
  - Creates one position.

- `PUT /positions/{position_id}`
  - Updates one position.

- `DELETE /positions/{position_id}`
  - Deletes one position.

- `GET /positions/snapshot`
  - Optional parameter: `as_of_date`.
  - Returns dashboard/data-entry snapshot rows with asset, owner, broker, quantity, value, and ownership shares.

- `POST /positions/bulk`
  - Saves a snapshot draft.
  - Supports `replace_snapshot`.
  - Rows with `value <= 0` are ignored or delete existing rows.
  - Duplicate rows for the same date, asset, owner, and broker are merged.
  - When `replace_snapshot = true`, positions for the target date that are missing from the payload are deleted.

### Ownership Splits

- `GET /positions/ownership`
  - Lists all position ownership splits.

- `GET /positions/{position_id}/ownership`
  - Lists one position split.

- `PUT /positions/{position_id}/ownership`
  - Replaces one position split.
  - Shares must be between `0` and `1`.
  - Total shares must not exceed `1`.

### Dashboard

- `GET /dashboard/summary`
  - Optional parameter: `as_of_date`.
  - Returns total value, position count, and breakdowns by asset, category, broker, and owner.

- `GET /dashboard/details`
  - Optional parameter: `as_of_date`.
  - Returns selected-date detail rows.

- `GET /dashboard/dates`
  - Returns available snapshot dates.

- `GET /dashboard/history`
  - Returns historical dashboard points with summary and detail rows.

## Frontend Specification

The primary frontend is now the React app in `frontend/`.

It is a React + TypeScript browser app using the FastAPI JSON endpoints. The production build is generated with Vite into `frontend/dist`.

Local frontend commands:

```bash
cd frontend
npm install
npm run dev
npm run build
```

FastAPI serves the built React app from:

```text
/
/dashboard
```

Built frontend assets are served under:

```text
/ui/
```

The legacy static HTML dashboard remains at `app/static/dashboard.html` as a fallback if `frontend/dist/index.html` does not exist.

### Global Header

- Selects the current dashboard `as_of_date`.
- Refreshes all data.
- Exports JSON backup.

### Tabs

- `Data entry`
- `Antonio`
- `Patricia`
- `Aggregate`

### Data Entry Tab

Primary workflow:

1. Choose a new target date.
2. Load the selected date or the previous available date as a template.
3. Edit rows in a spreadsheet-like table.
4. Optionally apply configured stock prices.
5. Save the snapshot.

Snapshot columns:

- `Category`
- `Name`
- `Owner`
- `Quantity`
- `Value`
- `Broker`

Rules:

- `Category`, `Name`, and `Owner` use selectors from catalog data.
- Common/shared rows use no direct `owner_id` and can store split percentages.
- Rows with no value or zero value should not be kept as active positions.
- Unsaved draft changes are tracked in browser state.

Catalog sections:

- Owners can be created, modified, and deleted.
- Assets can be created, modified, and deleted.
- Stock assets can store provider/ticker settings for price lookup.
- Company net-assets valuation opens a modal with Assets and Liabilities tables. The calculated net value is applied back to the snapshot row.
- The Investing selector defaults to all assets except cash and home/Casa, and can be overridden per asset.

### Antonio And Patricia Tabs

Each owner tab shows:

- Total assets metric
- Investing metric
- Position count
- Current positions table filtered by selected date and owner
- Historical absolute table
- Historical percent-vs-start table
- Total assets by category line chart
- Latest NAV by category pie chart
- Investing by category line chart
- Latest investing NAV by category pie chart

### Aggregate Tab

The aggregate tab shows:

- Total assets metric
- Investing metric
- Total position count
- Owner count
- Aggregated positions table
- Historical absolute table
- Historical percent-vs-start table
- Category and broker/bank charts
- The same total/investing line and pie chart structure as the owner tabs

### Chart Behavior

Line charts:

- Display category lines plus a `Total` line.
- Default behavior:
  - All categories visible on the left axis.
  - `Total` visible on the right axis.
  - Units are EUR.
- Each line chart has its own settings button.
- Settings include:
  - Palette
  - Units: EUR, `% vs start`, or `Index 100`
  - Show/hide series
  - Assign each series to left or right axis
- Axes are drawn directly on the chart.
- Line charts do not display a legend.

Pie charts:

- Each pie chart has its own settings button.
- Settings include:
  - Palette
  - Show/hide categories
- Legend is displayed to the right of the chart.
- Legend contains category names only.
- Values/percentages are drawn inside the pie chart.

Bar charts:

- Aggregate category and broker/bank bar charts also have their own settings button.
- Settings include:
  - Palette
  - Show/hide bars

### Tables

All dashboard and catalog tables support sorting by column in ascending or descending order.

## Import Specification

Import command:

```bash
python import_excel.py --path "./" --do-import
```

Dry run:

```bash
python import_excel.py --path "./" --dry-run
```

Import rules:

- Import `Patrimonio ARS.xlsx` as Antonio data.
- Import `Patrimonio Patri.xlsx` as Patri/Patricia data.
- Do not import `Patrimonio Comun.xlsx`.
- Do not import `Resumen` sheets.
- Only import rows where `Tipo Activo` has a value.
- Skip rows where `NAV` is blank or zero.
- `Tipo Activo` maps to asset `category`.
- `Activo` maps to asset `name`.
- `Cantidad` maps to position `quantity`.
- `NAV` maps to position total `value`.
- `Broker` maps to position `broker`.

## Security And Production Gaps

The beta has optional single-admin cookie-session authentication and an audit log. Enable it in deployed environments with:

```text
APP_AUTH_ENABLED=true
PATRIMONIO_USERNAME=<admin-user>
PATRIMONIO_PASSWORD=<strong-password>
PATRIMONIO_SESSION_SECRET=<long-random-secret>
```

Before exposing the app beyond a trusted private network, still add or verify:

- HTTPS termination and `APP_COOKIE_SECURE=true`.
- CSRF protection is implemented for cookie-authenticated mutating requests; verify reverse proxies preserve `X-CSRF-Token`.
- Role-based authorization if more than one user/admin will use the app.
- Secure database credentials through environment secrets.
- Regular database backups.
- `alembic upgrade head` as part of release/startup.
- Server logging and request error monitoring.
- Restore tested first in a disposable database.

## Health And Backup

Deployment systems can use:

```text
GET /health
```

for liveness checks.

Use:

```text
GET /export
```

for application-level JSON backup. Database-level backups are still recommended for server operations.
