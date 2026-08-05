# Patrimonio handover for the next coding agent

## Current cleanup note

The project has been refactored so Excel is treated as a bootstrap/import source, not the long-term source of truth. Future work should assume the app will need UI workflows to create and edit owners, assets, positions, and valuation inputs directly.

Current modeling direction:

- `Asset` is now a static dimension/catalog record: name, category, asset type, valuation method, optional price provider config, shared flag, created date.
- `InvestingAsset` is category-level configuration for whether a category counts toward Investing. It replaces the old UI idea of per-asset investing overrides.
- `PriceHistory` stores successful asset-based provider quote lookups for audit/history.
- `AuditLog` stores API mutation activity. With `APP_AUTH_ENABLED=true`, audit actors use the logged-in username; otherwise they default to `api`.
- Dated holding facts live in `Position`: date, asset, optional direct owner, broker/bank, quantity, value, and source.
- Shared/common ownership is stored on each position through `PositionOwnership`, not on `Asset`. This is important because Antonio and Patri can hold the same asset in different banks/accounts with different splits.
- Ownerless common positions default to the family split Antonio 50% / Patri or Patricia 50% unless explicit position-level shares are saved.
- Excel `NAV` is the total position value and is imported into `Position.value`.
- Excel `Cantidad` is historical quantity and is imported into `Position.quantity`.
- Excel rows with blank, missing, or zero `NAV` are not imported as positions.
- Excel `NAV` is no longer imported into `AssetValuation`; that table is reserved for future manual valuation workflows.
- Dashboard summaries are exact-date snapshots and include owner, asset, category, and broker/bank breakdowns.
- Importable Excel sheets are sorted by detected date, but missing holdings are left absent rather than represented as zero-value positions.

## 1. Purpose of the application

This repository is a lightweight MVP for managing a family or business portfolio. The app imports Excel workbooks into a relational model, stores historical valuations and dated position snapshots, and exposes a small dashboard for reviewing portfolio value by date, asset, and owner.

The current workflow is:

1. Place one or more Excel workbooks in the repository root.
2. Run the importer to inspect or load the data.
3. Serve the FastAPI dashboard and query the API for snapshots and owner-level detail rows.

Important repository assets:

- Sample workbooks: [Patrimonio ARS.xlsx](../Patrimonio%20ARS.xlsx), [Patrimonio Patri.xlsx](../Patrimonio%20Patri.xlsx), and [Patrimonio Comun.xlsx](../Patrimonio%20Comun.xlsx)
- Local SQLite database: [patrimonio.db](../patrimonio.db)

## 2. Technology stack

- Python 3.10+ (the container base image uses 3.10)
- FastAPI for the HTTP API and static dashboard serving
- SQLAlchemy 1.4 + SQLite by default, with PostgreSQL-compatible configuration via DATABASE_URL
- Pydantic for request/response schemas
- OpenPyXL for Excel import
- Uvicorn for running the app
- Docker Compose for a PostgreSQL database option
- Unittest for regression tests

## 3. Repository structure

```text
.
├── app/
│   ├── __pycache__/
│   ├── static/
│   │   └── dashboard.html
│   ├── crud.py
│   ├── database.py
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   └── valuation.py
├── docs/
│   └── COPILOT_HANDOVER.md
├── tests/
│   └── test_positions.py
├── import_excel.py
├── Dockerfile
├── docker-compose.yml
├── proposal.md
├── README.md
├── requirements.txt
├── .env.example
├── patrimonio.db
├── Patrimonio ARS.xlsx
├── Patrimonio Comun.xlsx
├── Patrimonio Patri.xlsx
```

## 4. Features already implemented

### Excel import

The importer is implemented in [import_excel.py](../import_excel.py) and already supports:

- Reading .xlsx workbooks with OpenPyXL
- Inspecting workbook sheets in dry-run mode
- Skipping the shared workbook named like “Comun”
- Skipping sheets named “Resumen”
- Filtering rows so only detailed rows with a populated “Tipo Activo”-style category field are imported
- Creating or updating assets, owners, and positions
- Inferring dates from sheet names or header cells

Key functions:

- find_table(ws, start_row=1)
- inspect_file(path)
- normalize_key(value)
- get_row_value(row, keys)
- detect_name(row)
- try_parse_date(value)
- detect_quantity(row)
- detect_category(row)
- detect_net_asset_value(row)
- detect_owner_name(row)
- detect_sheet_date(ws, sheet_name, headers)
- import_file(path, db, dry_run=True)
- main(path, dry_run=True, do_import=False)

### Data model and persistence

The SQLAlchemy model lives in [app/models.py](../app/models.py). It contains:

- Owner
- Asset
- InvestingAsset
- PriceHistory
- AuditLog
- AssetValuation
- Position
- PositionOwnership
- Ownership, legacy migration bridge only

Key design points:

- Assets are catalog/dimension rows and do not own owner allocation.
- Assets may include stock price configuration through `price_provider` and `price_symbol`.
- Investing classification is stored by category in `investing_assets`; cash/Casa default to not invested, other categories default to invested.
- Positions are snapshot rows keyed by asset, date, owner/account context, and broker/bank.
- Direct Antonio/Patri holdings can use `Position.owner_id`.
- Common/shared holdings use `PositionOwnership` rows so each position can have its own split.
- Asset valuations are reserved for future manual valuation workflows.

### CRUD and dashboard queries

The business logic layer is in [app/crud.py](../app/crud.py). It implements:

- create_owner(db, name, type='person')
- get_owner_by_name(db, name)
- list_owners(db, skip, limit)
- create_asset(db, asset_data)
- create_asset_valuation(db, asset, as_of_date, value, source='import')
- create_position(db, asset, as_of_date, quantity, value, source='import')
- replace_position_ownership(db, position, shares)
- get_position_ownership(db, position)
- _resolve_positions_for_date(db, as_of_date=None, asset_id=None)
- get_available_dates(db)
- get_positions(db, as_of_date=None, asset_id=None, skip=0, limit=100)
- get_dashboard_details(db, as_of_date=None)
- get_dashboard_summary(db, as_of_date=None)
- list_investing_assets(db)
- upsert_investing_asset(db, category, is_invested)
- migrate_legacy_asset_ownership_to_positions(db)
- list_assets(db, skip, limit)

The dashboard queries are explicitly date-aware and now resolve exact-date positions rather than relying on “latest before date” logic.

### API and dashboard

The FastAPI app is in [app/main.py](../app/main.py).

Available routes:

- GET /
- GET /dashboard
- POST /owners/
- GET /owners/
- DELETE /owners/{owner_id}
- POST /assets/
- GET /assets/
- DELETE /assets/{asset_id}
- GET /investing-assets/
- PUT /investing-assets/{category}
- GET /prices/quote
- GET /positions/
- GET /positions/snapshot
- POST /positions/bulk
- GET /positions/ownership
- GET /positions/{position_id}/ownership
- PUT /positions/{position_id}/ownership
- GET /dashboard/summary
- GET /dashboard/details
- GET /dashboard/dates

The primary UI is the React + TypeScript frontend in [frontend/src/App.tsx](../frontend/src/App.tsx), built with Vite into `frontend/dist` and served by FastAPI. [app/static/dashboard.html](../app/static/dashboard.html) remains only as a fallback if the React build is unavailable. The React UI loads the summary, details, and available date list from the API and renders:

- total value
- position count
- selected date
- owner totals
- asset totals
- selected-date detail rows with columns Asset / Owner / Category / Quantity / Value
- data-entry tab that copies the previous date as a template for a new snapshot
- table saves from the dashboard use `replace_snapshot=true`, so omitted rows are deleted for that exact date
- the snapshot table is first in the Data Entry tab and uses selectors for Category, Name, and Owner
- new assets should be created in the asset catalog; stock assets can store `price_provider` and `price_symbol`
- the dashboard can apply configured stock prices to snapshot values through `/prices/quote`
- the dashboard has unsaved-change protection for the snapshot table
- `/export` downloads a JSON backup of owners, assets, positions, and position-level ownership splits
- `/restore` can replace the database from a version-1 JSON backup when `confirm_restore=true`
- `/auth/status`, `/auth/login`, and `/auth/logout` provide optional cookie-session auth
- Admin tab shows audit log, price history, and guarded restore
- position-level ownership split modal for common/shared rows
- asset and owner catalog maintenance, including delete actions
- `Investing_Assets` category table for investing classification
- chart settings, stable colors per category, pie percentages, and line-chart hover values

### Tests

Regression tests are in [tests/test_positions.py](../tests/test_positions.py). The current suite covers:

- position persistence
- dashboard summary aggregation
- exact-date position resolution
- available date discovery
- Excel header mapping
- importer behavior for detailed sheets
- category-level investing defaults and API overrides

The current test run status is verified: 39 tests ran and all passed.

## 5. Current work in progress

The most recent implementation focus was preparing the beta for deployment and safer day-to-day editing:

- Alembic migrations now cover the beta schema, audit log, and price history tables.
- Optional cookie-session authentication is available through environment variables.
- Cookie-authenticated mutating requests use double-submit CSRF protection through `X-CSRF-Token`.
- Login attempts are rate limited in memory per client/user.
- Restore is disabled by default in authenticated deployments unless `APP_RESTORE_ENABLED=true`.
- API mutations are recorded in the audit log.
- JSON export/restore is available, with restore guarded by an explicit confirmation flag and Admin-tab confirmation text.
- The Admin tab exposes audit log, price history, and restore workflows.

## 6. Architecture and design decisions

### Backend structure

The backend is intentionally simple and file-oriented for an MVP:

- FastAPI handles routing and static file serving.
- SQLAlchemy models describe the domain entities.
- CRUD functions encapsulate persistence logic and query composition.
- The importer is a standalone script that uses the same CRUD layer as the API.

### Data model decisions

The application stores data in a snapshot-oriented structure:

- Assets are stable catalog rows.
- Position rows represent the portfolio state at a specific date.
- PositionOwnership rows allocate shared/common position value to Antonio, Patri/Patricia, or other owners.
- The legacy Ownership table is only a migration bridge for older databases that stored ownership on assets.

This was chosen because the source workbooks contain date-based snapshots rather than a simple current-state ledger.

### Import logic approach

The importer uses heuristics rather than a strict schema contract because the Excel files are not normalized. It searches for likely header rows based on common field names such as “Tipo Activo”, “Activo”, “Cantidad”, “NAV”, and “Broker”. This keeps the importer flexible but also means it can be fragile when workbook structure changes.

## 7. Database structure

The canonical deployment schema path is Alembic:

- `0001_initial_schema`
- `0002_beta_schema_updates`
- `0003_audit_log`
- `0004_price_history`

The app still calls `Base.metadata.create_all` and `ensure_beta_schema` at startup as a development compatibility layer for old local SQLite databases.

### Tables

- owners
  - id
  - name (unique)
  - type

- assets
  - id
  - name
  - category
  - asset_type
  - valuation_method
  - price_provider
  - price_symbol
  - is_investment, legacy/internal
  - is_shared
  - created_at

- investing_assets
  - category
  - is_invested

- asset_valuations
  - id
  - asset_id (FK to assets.id)
  - as_of_date
  - value
  - source
  - created_at

- positions
  - id
  - asset_id (FK to assets.id)
  - owner_id (nullable FK to owners.id)
  - as_of_date
  - quantity
  - value
  - broker
  - source
  - created_at

- position_ownership
  - position_id (FK to positions.id)
  - owner_id (FK to owners.id)
  - share

- company_valuation_items
  - id
  - asset_id (FK to assets.id)
  - as_of_date
  - item_type
  - name
  - amount
  - created_at

- ownership, legacy only
  - owner_id (FK to owners.id)
  - asset_id (FK to assets.id)
  - share

### Configuration

The database connection comes from [app/database.py](../app/database.py):

- Default local database: sqlite:///./patrimonio.db
- Override via DATABASE_URL for PostgreSQL or other backends

## 8. API endpoints

The API is defined in [app/main.py](../app/main.py).

### Core routes

- POST /owners/
  - Creates an owner.
  - Rejects duplicates by owner name.

- GET /owners/
  - Lists owners with pagination.

- DELETE /owners/{owner_id}
  - Deletes unused owners.
  - Blocks deletion when positions or position ownership rows still reference the owner.

- POST /assets/
  - Creates an asset.

- GET /assets/
  - Lists assets with pagination.

- DELETE /assets/{asset_id}
  - Deletes an asset and its dependent positions.

- GET /positions/
  - Lists positions.
  - Supports as_of_date, asset_id, skip, and limit.

- GET /positions/snapshot
  - Returns editable selected-date rows, including position-level ownership shares.

- POST /positions/bulk
  - Saves a complete selected-date table from the data-entry UI.
  - Supports copied previous-date templates and explicit ownership shares for common rows.

- GET /positions/ownership
  - Lists saved position-level ownership splits.

- GET /positions/{position_id}/ownership
  - Reads one position's split.

- PUT /positions/{position_id}/ownership
  - Replaces one position's split after validating owners and share totals.

### Dashboard routes

- GET /dashboard/summary
  - Returns total value, position count, as_of_date, by_asset, and by_owner.
  - Accepts an optional as_of_date query string.

- GET /dashboard/details
  - Returns the selected-date detail rows for the dashboard table.
  - Each row includes asset_name, owner_name, category, quantity, and value.

- GET /dashboard/dates
  - Returns the distinct available snapshot dates from the positions table.

### Static routes

- GET /
- GET /dashboard
  - Serves [app/static/dashboard.html](../app/static/dashboard.html)

## 9. Authentication and authorization approach

Authentication is intentionally simple for the beta:

- `APP_AUTH_ENABLED=true` enables login protection for data APIs and the dashboard UI.
- `/auth/status`, `/auth/login`, and `/auth/logout` implement the login flow.
- Sessions are HMAC-signed HTTP-only cookies using `PATRIMONIO_SESSION_SECRET`.
- Mutating requests require `X-CSRF-Token` matching the CSRF cookie.
- Credentials come from `PATRIMONIO_USERNAME` and `PATRIMONIO_PASSWORD`.
- Local development defaults to auth disabled unless the environment enables it.

There is still no role-based authorization or multi-user account model. Treat the current login as a single-admin guard for private deployment.

## 10. Important dependencies

Defined in [requirements.txt](../requirements.txt):

- fastapi==0.95.2
- uvicorn[standard]==0.20.0
- SQLAlchemy==1.4.49
- alembic==1.11.1
- openpyxl==3.1.2
- python-dotenv==1.0.0
- pydantic==1.10.9
- psycopg2-binary==2.9.7

## 11. Commands to install, run, test, and build

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows PowerShell
python -m pip install -r requirements.txt
```

### Run the API locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Import the Excel files (inspect only)

```bash
python import_excel.py --path "./" --dry-run
```

### Import the Excel files into the database

```bash
python import_excel.py --path "./" --do-import
```

### Run the tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Run PostgreSQL via Docker Compose

```bash
docker-compose up -d
docker-compose ps
```

### Build the container image

```bash
docker build -t patrimonio .
```

## 12. Known bugs and technical debt

The code is functional for the MVP, but the following issues and debt should be noted:

- Alembic migrations exist for the current beta schema. The app still has startup schema helpers for development compatibility; production should run `alembic upgrade head` explicitly.
- The importer is heuristic-based and can break if the workbook headers or sheet structure change significantly.
- Asset matching is currently name-based in [app/crud.py](../app/crud.py), which can cause collisions or incorrect updates when two assets share the same name.
- The primary dashboard is now React + TypeScript, but [app/static/dashboard.html](../app/static/dashboard.html) is still kept as a fallback and should not be treated as the main UI.
- Authentication is single-admin only; there is no role-based authorization or user-management UI.
- Dates are handled as naive datetime values in the model and import logic; timezone handling is not implemented.
- The dashboard summary uses the latest available position when as_of_date is not provided; that fallback is operationally convenient but should be made explicit in the UI/UX.
- Imported Antonio/Patri workbooks create direct-owner positions. Manual common/company positions should use ownerless rows with position-level ownership shares.

## 13. Decisions still pending

The following decisions should be resolved before the project grows beyond the MVP:

1. Authorization strategy
   - keep single-admin access or introduce roles/users
   - define who can restore backups or delete catalog records

2. Production schema policy
   - decide when to disable startup `create_all`/beta schema helpers and rely only on Alembic

3. Import schema standardization
   - should the importer support a stricter mapping configuration instead of heuristics?
   - should workbook-specific adapters be introduced?

4. Frontend strategy
   - continue React/Vite and eventually remove the static HTML fallback?

5. Production deployment strategy
   - PostgreSQL in Docker/VM or managed database
   - backup and restore strategy
   - environment-based configuration management

## 14. Next recommended tasks

The next logical tasks for the continuation agent are:

1. Add role-based authorization if the app will have more than one admin/user.
2. Decide when to disable startup `create_all`/beta schema helpers in production and rely only on Alembic.
3. Add more API tests, especially around import edge cases and dashboard endpoints.
4. Harden the importer for additional workbook variants and sheet naming patterns.
5. Add a clearer default-date UX for the dashboard when no date is selected.
6. Consider introducing a more explicit asset identity model instead of name-based matching.
7. Replace in-memory login throttling with shared storage if the app runs multiple web workers/containers.

## 15. Relevant files by area

- Application entrypoint and routing: [app/main.py](../app/main.py)
- SQLAlchemy models: [app/models.py](../app/models.py)
- Database session setup: [app/database.py](../app/database.py)
- Query and CRUD layer: [app/crud.py](../app/crud.py)
- Pydantic schemas: [app/schemas.py](../app/schemas.py)
- Dashboard UI: [frontend/src/App.tsx](../frontend/src/App.tsx)
- Static fallback UI: [app/static/dashboard.html](../app/static/dashboard.html)
- Excel import logic: [import_excel.py](../import_excel.py)
- Valuation helpers: [app/valuation.py](../app/valuation.py)
- Regression tests: [tests/test_positions.py](../tests/test_positions.py)
- Environment and dependency config: [requirements.txt](../requirements.txt), [.env.example](../.env.example), [docker-compose.yml](../docker-compose.yml), [Dockerfile](../Dockerfile)
- Project notes and original plan: [proposal.md](../proposal.md), [README.md](../README.md)

## 16. Practical note for the next agent

The repository is already in a working MVP state. The most important thing to keep in mind is that the importer and dashboard were recently tuned to the current workbook format and date-based snapshot requirements. If new Excel layouts appear, the importer logic will likely be the first place to adapt.
