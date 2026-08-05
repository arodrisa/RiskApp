# Patrimonio Decisions

This document records domain decisions confirmed while building the MVP.

## Excel Import Rules

- `Patrimonio ARS.xlsx` maps to owner `Antonio`.
- `Patrimonio Patri.xlsx` maps to owner `Patri`.
- `Patrimonio Comun.xlsx` is not imported. It is only a reference for understanding aggregation.
- Sheets named `Resumen` are not imported.
- Only rows with a populated `Tipo Activo` value are imported.
- Column mapping:
  - `Tipo Activo` -> asset category
  - `Activo` -> asset name
  - `Cantidad` -> historical position quantity
  - `NAV` -> historical total position value
  - `Broker` -> broker or bank

## Snapshot Rules

- The dashboard shows exact selected-date snapshots only.
- `NAV` is always the total position value, not unit price.
- `Cantidad` is historical and belongs to the position snapshot.
- Market value should not be treated as static asset data.
- If `NAV` is blank, missing, or zero, the row is not imported as a position.
- If a holding existed before and disappears from a later dated sheet, no position is created for that later date.
- The same asset name can appear in separate holdings by owner, broker, or source.

## Current Data Model Direction

- `Asset` is a dimension-like table for mostly static identity data.
- `Position` stores historical snapshot facts: date, asset, owner, broker, quantity, and value.
- Excel `NAV` is imported into `Position.value`, not `AssetValuation`.
- `AssetValuation` remains available for future manual valuation workflows, such as real estate or company valuation components.
- Destructive database reset is allowed only for development import workflows and must not be enabled in production.
