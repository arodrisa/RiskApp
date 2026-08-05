import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type * as React from 'react';
import {
  Download,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Settings,
  Trash2,
  X,
} from 'lucide-react';
import { api } from './api';
import {
  Asset,
  AuditLog,
  AuthStatus,
  ChartMode,
  ChartPalette,
  ChartSettings,
  CompanyValuationItem,
  DashboardDetailRow,
  DashboardHistoryPoint,
  DashboardSummary,
  EntityOwnership,
  InvestingAsset,
  Owner,
  PriceAuditRow,
  PriceHistory,
  PriceQuote,
  ProjectInvitationResult,
  ProjectUser,
  RestoreResult,
  SnapshotRow,
  SortDirection,
  SortState,
} from './types';

const TOTAL_SERIES_KEY = '__total';
const ownerAliases: Record<string, string[]> = {
  Antonio: ['Antonio'],
  Patricia: ['Patri', 'Patricia'],
};

const palettes: Record<ChartPalette, string[]> = {
  calm: ['#5dd3b3', '#7aa7ff', '#f4c95d', '#f28f8f', '#b994ff', '#7bd88f', '#f0a6ca', '#8bd3e6'],
  vivid: ['#00a7dd', '#e7a900', '#e45778', '#43c89a', '#9b7dea', '#d96d18', '#2fa866', '#cf62d8'],
  mono: ['#f6f7fb', '#cbd5e1', '#94a3b8', '#64748b', '#475569', '#334155', '#1f2937', '#111827'],
};

const defaultChartSettings: ChartSettings = {
  palette: 'calm',
  mode: 'currency',
  hiddenSeries: [],
  seriesAxes: { [TOTAL_SERIES_KEY]: 'secondary' },
};

const assetTypes = [
  ['', 'Other'],
  ['stock', 'Stock'],
  ['fund', 'Fund'],
  ['bond', 'Bond'],
  ['cash', 'Cash'],
  ['real_estate', 'Real estate'],
  ['company', 'Company'],
];

function canonicalAssetType(value?: string | null) {
  const normalized = String(value || '').trim().toLowerCase();
  const aliases: Record<string, string> = {
    stock: 'stock',
    stocks: 'stock',
    accion: 'stock',
    acciones: 'stock',
    fund: 'fund',
    funds: 'fund',
    fondo: 'fund',
    fondos: 'fund',
    bond: 'bond',
    bonds: 'bond',
    cash: 'cash',
    realestate: 'real_estate',
    real_estate: 'real_estate',
    inmueble: 'real_estate',
    casa: 'real_estate',
    company: 'company',
    empresa: 'company',
  };
  return aliases[normalized] || normalized;
}

function normalizeAsset(asset: Asset): Asset {
  return {
    ...asset,
    asset_type: canonicalAssetType(asset.asset_type) || null,
  };
}

const valuationMethods = [
  ['market_direct', 'Market direct'],
  ['price_provider', 'Price provider'],
  ['market_minus_debt', 'Market minus debt'],
  ['company_net_assets', 'Company net assets'],
];

type TabKey = 'data' | 'antonio' | 'patricia' | 'aggregate' | 'admin';
type StatusTone = 'muted' | 'success' | 'error';
type Status = { message: string; tone: StatusTone };
type DraftErrorMap = Record<number, string[]>;
type ChartKind = 'line' | 'pie' | 'bar';
type ChartDefinition = {
  key: string;
  title: string;
  categories: string[];
  kind: ChartKind;
  includeTotal: boolean;
  defaultHiddenSeries?: string[];
};

function formatCurrency(value: number | null | undefined) {
  return new Intl.NumberFormat('es-ES', {
    style: 'currency',
    currency: 'EUR',
    maximumFractionDigits: 2,
  }).format(value || 0);
}

function formatNumber(value: number | null | undefined) {
  return new Intl.NumberFormat('es-ES', { maximumFractionDigits: 2 }).format(value || 0);
}

function formatPercent(value: number | null | undefined) {
  return `${formatNumber(value || 0)}%`;
}

function normalizeDate(value?: string | null) {
  if (!value) return '';
  return value.slice(0, 10);
}

function rowValue(row: { value?: number | null }) {
  return Number(row.value || 0);
}

function rowQuantity(row: { quantity?: number | null }) {
  return Number(row.quantity || 0);
}

function isStockRow(row: { asset_type?: string | null }) {
  return canonicalAssetType(row.asset_type) === 'stock';
}

function stockQuantityLabel(row: { asset_type?: string | null; quantity?: number | null }) {
  return isStockRow(row) ? formatNumber(row.quantity) : '';
}

function ownerMatches(row: DashboardDetailRow, owner: string) {
  return (ownerAliases[owner] || [owner]).includes(row.owner_name);
}

function defaultCategoryIsInvested(category?: string | null) {
  const normalized = String(category || '').trim().toLowerCase();
  return !['cash', 'caja', 'efectivo', 'casa'].includes(normalized);
}

function isCategoryInvested(category: string | null | undefined, investmentByCategory: Record<string, boolean>) {
  const key = category || 'Uncategorized';
  return investmentByCategory[key] ?? defaultCategoryIsInvested(key);
}

function investmentValue(rows: Array<{ category?: string | null; value?: number | null }>, investmentByCategory: Record<string, boolean>) {
  return rows
    .filter((row) => isCategoryInvested(row.category, investmentByCategory))
    .reduce((sum, row) => sum + rowValue(row), 0);
}

function categoryList(historyRows: Array<{ details: Array<{ category?: string | null }> }>) {
  const values = new Set<string>();
  historyRows.forEach((item) => item.details.forEach((row) => values.add(row.category || 'Uncategorized')));
  return [...values].sort((a, b) => a.localeCompare(b));
}

function categoryValue(rows: Array<{ category?: string | null; value?: number | null }>, category: string) {
  return rows
    .filter((row) => (row.category || 'Uncategorized') === category)
    .reduce((sum, row) => sum + rowValue(row), 0);
}

function firstNonZero(values: number[]) {
  return values.find((value) => Number(value || 0) !== 0) || 0;
}

function transformSeries(values: number[], mode: ChartMode) {
  if (mode === 'currency') return values;
  const first = firstNonZero(values);
  if (!first) return values.map(() => 0);
  if (mode === 'percent') return values.map((value) => (Number(value || 0) / first - 1) * 100);
  return values.map((value) => (Number(value || 0) / first) * 100);
}

function formatChartValue(value: number, mode: ChartMode) {
  if (mode === 'currency') return formatCurrency(value);
  if (mode === 'percent') return formatPercent(value);
  return `${formatNumber(value)} idx`;
}

function usesNetValuation(row: { valuation_method?: string | null }) {
  return row.valuation_method === 'company_net_assets' || row.valuation_method === 'market_minus_debt';
}

function hashLabel(label: string) {
  return [...label].reduce((hash, char) => ((hash << 5) - hash + char.charCodeAt(0)) | 0, 0);
}

function colorForLabel(label: string, palette: string[], domain: string[]) {
  const index = domain.indexOf(label);
  const fallbackIndex = Math.abs(hashLabel(label));
  return palette[(index >= 0 ? index : fallbackIndex) % palette.length];
}

function seriesLabel(key: string) {
  return key === TOTAL_SERIES_KEY ? 'Total' : key;
}

function seriesKeys(categories: string[], includeTotal: boolean) {
  return includeTotal ? [...categories, TOTAL_SERIES_KEY] : categories;
}

function nextDirection(current?: SortState, key?: string): SortDirection {
  return current && current.key === key && current.direction === 'asc' ? 'desc' : 'asc';
}

function compareValues(a: unknown, b: unknown, direction: SortDirection) {
  const left = typeof a === 'number' ? a : String(a ?? '').toLowerCase();
  const right = typeof b === 'number' ? b : String(b ?? '').toLowerCase();
  const result = typeof left === 'number' && typeof right === 'number'
    ? left - right
    : String(left).localeCompare(String(right), 'es', { numeric: true, sensitivity: 'base' });
  return direction === 'asc' ? result : -result;
}

function useLocalChartSettings() {
  const [settings, setSettings] = useState<Record<string, ChartSettings>>(() => {
    try {
      return JSON.parse(localStorage.getItem('patrimonioChartSettings') || '{}');
    } catch {
      return {};
    }
  });

  useEffect(() => {
    localStorage.setItem('patrimonioChartSettings', JSON.stringify(settings));
  }, [settings]);

  const getSettings = useCallback((key: string, categories: string[], includeTotal: boolean, defaultHiddenSeries: string[] = []) => {
    const allowed = seriesKeys(categories, includeTotal);
    const current = settings[key] || { ...defaultChartSettings, hiddenSeries: defaultHiddenSeries };
    return {
      ...defaultChartSettings,
      ...current,
      hiddenSeries: (current.hiddenSeries || []).filter((item) => allowed.includes(item)),
      seriesAxes: {
        ...current.seriesAxes,
        ...(includeTotal ? { [TOTAL_SERIES_KEY]: current.seriesAxes?.[TOTAL_SERIES_KEY] || 'secondary' } : {}),
      },
    };
  }, [settings]);

  const patchSettings = useCallback((key: string, patch: Partial<ChartSettings>) => {
    setSettings((current) => ({
      ...current,
      [key]: {
        ...defaultChartSettings,
        ...(current[key] || {}),
        ...patch,
        seriesAxes: {
          ...defaultChartSettings.seriesAxes,
          ...(current[key]?.seriesAxes || {}),
          ...(patch.seriesAxes || {}),
        },
      },
    }));
  }, []);

  return { getSettings, patchSettings };
}

function Modal({
  title,
  subtitle,
  onClose,
  children,
}: {
  title: string;
  subtitle?: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const closeRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      previous?.focus();
    };
  }, [onClose]);

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
        <div className="modal-header">
          <div>
            <h2 id="modalTitle">{title}</h2>
            {subtitle ? <p className="muted">{subtitle}</p> : null}
          </div>
          <button ref={closeRef} className="secondary icon-button" type="button" onClick={onClose} aria-label="Close">
            <X size={17} />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

function StatusLine({ status }: { status: Status }) {
  return (
    <div className={`status ${status.tone}`} role="status" aria-live="polite">
      {status.message}
    </div>
  );
}

function SortableTable<T>({
  id,
  columns,
  rows,
  emptyLabel = 'No data',
}: {
  id: string;
  columns: Array<{
    key: string;
    label: string;
    value: (row: T) => unknown;
    render: (row: T, index: number) => React.ReactNode;
  }>;
  rows: T[];
  emptyLabel?: string;
}) {
  const [sort, setSort] = useState<SortState | undefined>();
  const sortedRows = useMemo(() => {
    if (!sort) return rows;
    const column = columns.find((item) => item.key === sort.key);
    if (!column) return rows;
    return [...rows].sort((a, b) => compareValues(column.value(a), column.value(b), sort.direction));
  }, [columns, rows, sort]);

  return (
    <div className="table-wrap">
      <table id={id}>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} aria-sort={sort?.key === column.key ? (sort.direction === 'asc' ? 'ascending' : 'descending') : 'none'}>
                <button
                  className="sort-button"
                  type="button"
                  onClick={() => setSort({ key: column.key, direction: nextDirection(sort, column.key) })}
                >
                  {column.label}
                  <span className="sort-indicator" aria-hidden="true">{sort?.key === column.key ? (sort.direction === 'asc' ? '▲' : '▼') : '↕'}</span>
                  <span className="sr-only">{sort?.key === column.key ? (sort.direction === 'asc' ? 'sorted ascending' : 'sorted descending') : 'sort'}</span>
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedRows.length ? sortedRows.map((row, index) => (
            <tr key={index}>{columns.map((column) => <td key={column.key}>{column.render(row, index)}</td>)}</tr>
          )) : (
            <tr><td colSpan={columns.length} className="muted">{emptyLabel}</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="metric">
      <h3>{label}</h3>
      <div className="value">{value}</div>
      {sub ? <div className="metric-sub">{sub}</div> : null}
    </div>
  );
}

function validateAsset(asset: Partial<Asset>) {
  if (!asset.name?.trim()) return 'Asset name is required.';
  if (!asset.category?.trim()) return 'Category is required.';
  if ((asset.asset_type || '').toLowerCase() === 'stock' && asset.price_provider === 'yahoo' && !asset.price_symbol?.trim()) {
    return 'Yahoo stock assets need a ticker/ID.';
  }
  return '';
}

function buildDefaultShares(owners: Owner[]) {
  const antonio = owners.find((owner) => owner.name === 'Antonio');
  const patri = owners.find((owner) => owner.name === 'Patri') || owners.find((owner) => owner.name === 'Patricia');
  return [
    ...(antonio ? [{ owner_id: antonio.id, owner_name: antonio.name, share: 0.5 }] : []),
    ...(patri ? [{ owner_id: patri.id, owner_name: patri.name, share: 0.5 }] : []),
  ];
}

function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('data');
  const [authStatus, setAuthStatus] = useState<AuthStatus | null>(null);
  const [dates, setDates] = useState<string[]>([]);
  const [viewDate, setViewDate] = useState('');
  const [targetDate, setTargetDate] = useState('');
  const [sourceDate, setSourceDate] = useState('');
  const [owners, setOwners] = useState<Owner[]>([]);
  const [entityOwnerships, setEntityOwnerships] = useState<EntityOwnership[]>([]);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [investingAssets, setInvestingAssets] = useState<InvestingAsset[]>([]);
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [details, setDetails] = useState<DashboardDetailRow[]>([]);
  const [history, setHistory] = useState<DashboardHistoryPoint[]>([]);
  const [draftRows, setDraftRows] = useState<SnapshotRow[]>([]);
  const [draftDirty, setDraftDirty] = useState(false);
  const [removedStack, setRemovedStack] = useState<Array<{ row: SnapshotRow; index: number }>>([]);
  const [status, setStatus] = useState<Status>({ message: '', tone: 'muted' });
  const [ownerStatus, setOwnerStatus] = useState<Status>({ message: '', tone: 'muted' });
  const [assetStatus, setAssetStatus] = useState<Status>({ message: '', tone: 'muted' });
  const [ownershipRowIndex, setOwnershipRowIndex] = useState<number | null>(null);
  const [companyValuationRowIndex, setCompanyValuationRowIndex] = useState<number | null>(null);
  const [chartDefinition, setChartDefinition] = useState<ChartDefinition | null>(null);
  const [saveReviewOpen, setSaveReviewOpen] = useState(false);
  const [priceAudit, setPriceAudit] = useState<PriceAuditRow[] | null>(null);
  const [restoreResult, setRestoreResult] = useState<RestoreResult | null>(null);
  const [ownerForm, setOwnerForm] = useState<Partial<Owner>>({ name: '', type: 'person' });
  const [assetForm, setAssetForm] = useState<Partial<Asset>>({
    name: '',
    category: '',
    asset_type: '',
    valuation_method: 'market_direct',
    price_provider: 'manual',
    price_symbol: '',
  });
  const [ownerEdits, setOwnerEdits] = useState<Record<number, Partial<Owner>>>({});
  const [assetEdits, setAssetEdits] = useState<Record<number, Partial<Asset>>>({});
  const { getSettings, patchSettings } = useLocalChartSettings();

  const loadAuthStatus = useCallback(async () => {
    const next = await api.authStatus();
    setAuthStatus(next);
    return next;
  }, []);

  const loadEntities = useCallback(async () => {
    const [ownerRows, assetRows, investingRows, ownershipRows] = await Promise.all([
      api.owners(),
      api.assets(),
      api.investingAssets(),
      api.entityOwnerships(),
    ]);
    const normalizedAssets = assetRows.map(normalizeAsset);
    setOwners(ownerRows);
    setAssets(normalizedAssets);
    setInvestingAssets(investingRows);
    setEntityOwnerships(ownershipRows);
    setOwnerEdits(Object.fromEntries(ownerRows.map((owner) => [owner.id, owner])));
    setAssetEdits(Object.fromEntries(normalizedAssets.map((asset) => [asset.id, asset])));
  }, []);

  const investmentByCategory = useMemo(() => Object.fromEntries(
    investingAssets.map((item) => [item.category || 'Uncategorized', item.is_invested]),
  ), [investingAssets]);

  const loadDashboards = useCallback(async (date?: string) => {
    const [summaryRows, detailRows, historyRows] = await Promise.all([
      api.summary(date),
      api.details(date),
      api.history(),
    ]);
    setSummary(summaryRows);
    setDetails(detailRows);
    setHistory(historyRows);
  }, []);

  const loadDates = useCallback(async () => {
    const dateRows = await api.dates();
    setDates(dateRows);
    const latest = dateRows[dateRows.length - 1] || '';
    setViewDate((current) => current || latest);
    setTargetDate((current) => current || latest);
    setSourceDate((current) => current || latest);
    return latest;
  }, []);

  const loadSnapshot = useCallback(async (date?: string, markDirty = false) => {
    const rows = await api.snapshot(date);
    setDraftRows(rows.map((row) => ({ ...row, owner_id: row.owner_id ?? '' })));
    setSourceDate(date || '');
    setDraftDirty(markDirty);
    setRemovedStack([]);
  }, []);

  const refreshAll = useCallback(async () => {
    setStatus({ message: 'Refreshing data...', tone: 'muted' });
    await loadEntities();
    const latest = await loadDates();
    const date = viewDate || latest;
    await Promise.all([loadDashboards(date), loadSnapshot(date)]);
    setStatus({ message: 'Data refreshed.', tone: 'success' });
  }, [loadDashboards, loadDates, loadEntities, loadSnapshot, viewDate]);

  useEffect(() => {
    loadAuthStatus()
      .then((auth) => {
        if (auth.authenticated) return refreshAll();
        setStatus({ message: 'Login required.', tone: 'muted' });
        return undefined;
      })
      .catch((error) => setStatus({ message: error.message, tone: 'error' }));
  }, [loadAuthStatus, refreshAll]);

  useEffect(() => {
    if (authStatus?.enabled && authStatus.authenticated && !['owner', 'admin', 'editor'].includes(authStatus.role || '')) {
      setActiveTab('aggregate');
    }
  }, [authStatus]);

  useEffect(() => {
    if (!viewDate) return;
    loadDashboards(viewDate).catch((error) => setStatus({ message: error.message, tone: 'error' }));
  }, [loadDashboards, viewDate]);

  useEffect(() => {
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      if (!draftDirty) return;
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, [draftDirty]);

  const draftErrors = useMemo(() => validateDraftRows(draftRows), [draftRows]);
  const saveReview = useMemo(() => buildSaveReview(draftRows, draftErrors), [draftRows, draftErrors]);

  const confirmDraftDiscard = () => !draftDirty || window.confirm('Discard unsaved snapshot draft changes?');

  const changeViewDate = async (date: string) => {
    if (!confirmDraftDiscard()) return;
    setViewDate(date);
    setTargetDate(date);
    await loadSnapshot(date);
  };

  const loadSelectedTemplate = async () => {
    if (!viewDate) return;
    if (!confirmDraftDiscard()) return;
    await loadSnapshot(viewDate, false);
    setTargetDate(viewDate);
    setStatus({ message: `Editing ${viewDate} using ${viewDate} as template.`, tone: 'success' });
  };

  const loadPreviousTemplate = async () => {
    if (!confirmDraftDiscard()) return;
    const previous = [...dates].reverse().find((date) => date < (targetDate || viewDate || '')) || dates[dates.length - 1];
    if (!previous) {
      setStatus({ message: 'No previous date is available.', tone: 'error' });
      return;
    }
    await loadSnapshot(previous, true);
    setSourceDate(previous);
    setStatus({ message: `Editing ${targetDate || viewDate || previous} using ${previous} as template.`, tone: 'success' });
  };

  const updateDraftRow = (index: number, patch: Partial<SnapshotRow>) => {
    setDraftRows((rows) => rows.map((row, rowIndex) => {
      if (rowIndex !== index) return row;
      const next = { ...row, ...patch };
      if (patch.asset_id !== undefined) {
        const asset = assets.find((item) => Number(item.id) === Number(patch.asset_id));
        next.asset_name = asset?.name || '';
        next.category = asset?.category || '';
        next.asset_type = asset?.asset_type || '';
        next.valuation_method = asset?.valuation_method || '';
      }
      if (patch.owner_id !== undefined && patch.owner_id !== '') {
        next.ownership_shares = [];
      }
      if (patch.owner_id === '' && (!next.ownership_shares || !next.ownership_shares.length)) {
        next.ownership_shares = buildDefaultShares(owners);
      }
      return next;
    }));
    setDraftDirty(true);
  };

  const addDraftRow = () => {
    setDraftRows((rows) => [...rows, {
      asset_id: assets[0]?.id || '',
      owner_id: owners[0]?.id || '',
      category: assets[0]?.category || '',
      asset_type: assets[0]?.asset_type || '',
      valuation_method: assets[0]?.valuation_method || '',
      quantity: 0,
      value: 0,
      broker: '',
      ownership_shares: [],
    }]);
    setDraftDirty(true);
  };

  const removeDraftRow = (index: number) => {
    setDraftRows((rows) => {
      const next = [...rows];
      const [removed] = next.splice(index, 1);
      if (removed) setRemovedStack((stack) => [...stack, { row: removed, index }]);
      return next;
    });
    setDraftDirty(true);
  };

  const undoRemove = () => {
    setRemovedStack((stack) => {
      const nextStack = [...stack];
      const item = nextStack.pop();
      if (!item) return nextStack;
      setDraftRows((rows) => {
        const next = [...rows];
        next.splice(item.index, 0, item.row);
        return next;
      });
      setDraftDirty(true);
      return nextStack;
    });
  };

  const fetchStockPrices = async () => {
    const audits: PriceAuditRow[] = [];
    setStatus({ message: 'Fetching configured stock prices...', tone: 'muted' });
    for (const [index, row] of draftRows.entries()) {
      const asset = assets.find((item) => Number(item.id) === Number(row.asset_id));
      if (!asset || asset.valuation_method !== 'price_provider' || asset.price_provider !== 'yahoo' || !asset.price_symbol) continue;
      const quote: PriceQuote = await api.quote(asset.id);
      audits.push({
        rowIndex: index,
        assetName: asset.name,
        symbol: asset.price_symbol,
        provider: quote.provider,
        quantity: rowQuantity(row),
        oldValue: rowValue(row),
        unitPrice: quote.price,
        newValue: rowQuantity(row) * quote.price,
        currency: quote.currency,
        asOf: quote.as_of,
      });
    }
    if (!audits.length) {
      setStatus({ message: 'No configured stock rows found. Set asset valuation to Price provider and add a Yahoo ticker first.', tone: 'error' });
      return;
    }
    setPriceAudit(audits);
    setStatus({ message: `Reviewed ${audits.length} price update(s).`, tone: 'success' });
  };

  const applyPriceAudit = () => {
    if (!priceAudit) return;
    setDraftRows((rows) => rows.map((row, index) => {
      const audit = priceAudit.find((item) => item.rowIndex === index);
      return audit ? { ...row, value: Number(audit.newValue.toFixed(2)) } : row;
    }));
    setDraftDirty(true);
    setPriceAudit(null);
  };

  const saveSnapshot = async () => {
    if (!targetDate) {
      setStatus({ message: 'Choose a draft target date before saving.', tone: 'error' });
      return;
    }
    if (Object.keys(draftErrors).length) {
      setStatus({ message: 'Fix the row errors before saving.', tone: 'error' });
      return;
    }
    await api.saveSnapshot(targetDate, draftRows.filter((row) => rowValue(row) > 0), true);
    setDraftDirty(false);
    setRemovedStack([]);
    setSaveReviewOpen(false);
    setViewDate(targetDate);
    await Promise.all([loadDates(), loadDashboards(targetDate), loadSnapshot(targetDate)]);
    setStatus({ message: `Snapshot ${targetDate} saved.`, tone: 'success' });
  };

  const createOwner = async (event: React.FormEvent) => {
    event.preventDefault();
    if (draftDirty) {
      setOwnerStatus({ message: 'Save or discard the snapshot draft before changing the catalog.', tone: 'error' });
      return;
    }
    await api.createOwner(ownerForm);
    setOwnerForm({ name: '', type: 'person' });
    await loadEntities();
    setOwnerStatus({ message: 'Owner added.', tone: 'success' });
  };

  const createAsset = async (event: React.FormEvent) => {
    event.preventDefault();
    if (draftDirty) {
      setAssetStatus({ message: 'Save or discard the snapshot draft before changing the catalog.', tone: 'error' });
      return;
    }
    const error = validateAsset(assetForm);
    if (error) {
      setAssetStatus({ message: error, tone: 'error' });
      return;
    }
    await api.createAsset(assetForm);
    setAssetForm({ name: '', category: '', asset_type: '', valuation_method: 'market_direct', price_provider: 'manual', price_symbol: '' });
    await loadEntities();
    setAssetStatus({ message: 'Asset added.', tone: 'success' });
  };

  const updateOwner = async (ownerId: number) => {
    if (draftDirty) {
      setOwnerStatus({ message: 'Save or discard the snapshot draft before changing the catalog.', tone: 'error' });
      return;
    }
    await api.updateOwner(ownerId, ownerEdits[ownerId]);
    await loadEntities();
    setOwnerStatus({ message: 'Owner saved.', tone: 'success' });
  };

  const updateAsset = async (assetId: number) => {
    if (draftDirty) {
      setAssetStatus({ message: 'Save or discard the snapshot draft before changing the catalog.', tone: 'error' });
      return;
    }
    const payload = assetEdits[assetId];
    const error = validateAsset(payload);
    if (error) {
      setAssetStatus({ message: error, tone: 'error' });
      return;
    }
    await api.updateAsset(assetId, payload);
    await loadEntities();
    setAssetStatus({ message: 'Asset saved.', tone: 'success' });
  };

  const updateInvestingAsset = async (category: string, isInvested: boolean) => {
    if (draftDirty) {
      setAssetStatus({ message: 'Save or discard the snapshot draft before changing the catalog.', tone: 'error' });
      return;
    }
    await api.updateInvestingAsset(category, isInvested);
    await loadEntities();
    setAssetStatus({ message: 'Investing_Assets saved.', tone: 'success' });
  };

  const deleteOwner = async (ownerId: number) => {
    if (draftDirty || !window.confirm('Delete this owner?')) return;
    await api.deleteOwner(ownerId);
    await loadEntities();
    setOwnerStatus({ message: 'Owner deleted.', tone: 'success' });
  };

  const deleteAsset = async (assetId: number) => {
    if (draftDirty || !window.confirm('Delete this asset?')) return;
    await api.deleteAsset(assetId);
    await loadEntities();
    setAssetStatus({ message: 'Asset deleted.', tone: 'success' });
  };

  const logout = async () => {
    await api.logout();
    setAuthStatus(await api.authStatus());
  };

  if (authStatus && authStatus.enabled && !authStatus.authenticated) {
    const resetToken = new URLSearchParams(window.location.search).get('reset');
    if (resetToken) {
      return <PasswordResetScreen token={resetToken} onComplete={() => {
        window.history.replaceState({}, '', window.location.pathname);
        loadAuthStatus().catch((error) => setStatus({ message: error.message, tone: 'error' }));
      }} />;
    }
    if (authStatus.needs_bootstrap) {
      return <BootstrapScreen onBootstrap={async (payload) => {
        const next = await api.bootstrap(payload);
        setAuthStatus(next);
        await refreshAll();
      }} />;
    }
    return <LoginScreen onLogin={async (username, password) => {
      const next = await api.login(username, password);
      setAuthStatus(next);
      await refreshAll();
    }} />;
  }

  const canEdit = !authStatus?.enabled || ['owner', 'admin', 'editor'].includes(authStatus.role || '');
  const isAdmin = !authStatus?.enabled || ['owner', 'admin'].includes(authStatus.role || '');

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Patrimonio</h1>
          <p className="muted">Snapshot-based asset control</p>
        </div>
        <div className="toolbar">
          <label>
            Viewing date
            <select value={viewDate} onChange={(event) => changeViewDate(event.target.value)}>
              {dates.map((date) => <option key={date} value={date}>{date}</option>)}
            </select>
          </label>
          {isAdmin ? <a className="button secondary" href="/export" onClick={(event) => {
            if (!confirmDraftDiscard()) event.preventDefault();
          }}>
            <Download size={16} /> Export backup
          </a> : null}
          <button type="button" onClick={() => refreshAll().catch((error) => setStatus({ message: error.message, tone: 'error' }))}>
            <RefreshCw size={16} /> Refresh
          </button>
          {authStatus?.enabled ? (
            <button className="secondary" type="button" onClick={() => logout().catch((error) => setStatus({ message: error.message, tone: 'error' }))}>
              {authStatus.username || 'User'} - Logout
            </button>
          ) : null}
        </div>
      </header>

      <nav className="tabs" aria-label="Dashboard tabs">
        {[
          ['data', 'Data entry'],
          ['antonio', 'Antonio'],
          ['patricia', 'Patricia'],
          ['aggregate', 'Aggregate'],
          ['admin', 'Admin'],
        ].filter(([key]) => (key !== 'data' || canEdit) && (key !== 'admin' || isAdmin)).map(([key, label]) => (
          <button
            key={key}
            type="button"
            className={`tab-button ${activeTab === key ? 'active' : ''}`}
            onClick={() => setActiveTab(key as TabKey)}
          >
            {label}
          </button>
        ))}
      </nav>

      <StatusLine status={status} />

      {activeTab === 'data' && canEdit ? (
        <DataEntryTab
          assets={assets}
          owners={owners}
          entityOwnerships={entityOwnerships}
          investingAssets={investingAssets}
          draftRows={draftRows}
          draftErrors={draftErrors}
          draftDirty={draftDirty}
          removedCount={removedStack.length}
          targetDate={targetDate}
          sourceDate={sourceDate}
          viewDate={viewDate}
          status={status}
          ownerStatus={ownerStatus}
          assetStatus={assetStatus}
          ownerForm={ownerForm}
          assetForm={assetForm}
          ownerEdits={ownerEdits}
          assetEdits={assetEdits}
          saveReview={saveReview}
          setTargetDate={setTargetDate}
          setOwnerForm={setOwnerForm}
          setAssetForm={setAssetForm}
          setOwnerEdits={setOwnerEdits}
          setAssetEdits={setAssetEdits}
          onLoadSelected={loadSelectedTemplate}
          onLoadPrevious={loadPreviousTemplate}
          onAddRow={addDraftRow}
          onUpdateRow={updateDraftRow}
          onRemoveRow={removeDraftRow}
          onUndoRemove={undoRemove}
          onOpenSplit={setOwnershipRowIndex}
          onOpenCompanyValuation={setCompanyValuationRowIndex}
          onFetchPrices={() => fetchStockPrices().catch((error) => setStatus({ message: error.message, tone: 'error' }))}
          onOpenSaveReview={() => setSaveReviewOpen(true)}
          onCreateOwner={(event) => createOwner(event).catch((error) => setOwnerStatus({ message: error.message, tone: 'error' }))}
          onCreateAsset={(event) => createAsset(event).catch((error) => setAssetStatus({ message: error.message, tone: 'error' }))}
          onUpdateOwner={(id) => updateOwner(id).catch((error) => setOwnerStatus({ message: error.message, tone: 'error' }))}
          onUpdateAsset={(id) => updateAsset(id).catch((error) => setAssetStatus({ message: error.message, tone: 'error' }))}
          onUpdateInvestingAsset={(category, isInvested) => updateInvestingAsset(category, isInvested).catch((error) => setAssetStatus({ message: error.message, tone: 'error' }))}
          onDeleteOwner={(id) => deleteOwner(id).catch((error) => setOwnerStatus({ message: error.message, tone: 'error' }))}
          onDeleteAsset={(id) => deleteAsset(id).catch((error) => setAssetStatus({ message: error.message, tone: 'error' }))}
          onCreateEntityOwnership={async (payload) => {
            await api.createEntityOwnership(payload);
            await loadEntities();
            setOwnerStatus({ message: 'Company ownership saved.', tone: 'success' });
          }}
          onDeleteEntityOwnership={async (id) => {
            if (!window.confirm('Delete this company ownership relationship?')) return;
            await api.deleteEntityOwnership(id);
            await loadEntities();
            setOwnerStatus({ message: 'Company ownership deleted.', tone: 'success' });
          }}
        />
      ) : null}

      {activeTab === 'antonio' ? (
        <OwnerDashboard
          ownerName="Antonio"
          viewDate={viewDate}
          details={details}
          history={history}
          investmentByCategory={investmentByCategory}
          getSettings={getSettings}
          patchSettings={patchSettings}
          openChartSettings={setChartDefinition}
        />
      ) : null}

      {activeTab === 'patricia' ? (
        <OwnerDashboard
          ownerName="Patricia"
          viewDate={viewDate}
          details={details}
          history={history}
          investmentByCategory={investmentByCategory}
          getSettings={getSettings}
          patchSettings={patchSettings}
          openChartSettings={setChartDefinition}
        />
      ) : null}

      {activeTab === 'aggregate' ? (
        <AggregateDashboard
          viewDate={viewDate}
          owners={owners}
          summary={summary}
          details={details}
          history={history}
          investmentByCategory={investmentByCategory}
          getSettings={getSettings}
          patchSettings={patchSettings}
          openChartSettings={setChartDefinition}
        />
      ) : null}

      {activeTab === 'admin' && isAdmin ? (
        <AdminTab
          authStatus={authStatus}
          restoreResult={restoreResult}
          onRestore={async (backup) => {
            const result = await api.restore(backup);
            setRestoreResult(result);
            await refreshAll();
          }}
        />
      ) : null}

      {ownershipRowIndex !== null ? (
        <OwnershipModal
          row={draftRows[ownershipRowIndex]}
          owners={owners}
          onClose={() => setOwnershipRowIndex(null)}
          onSave={(shares) => {
            updateDraftRow(ownershipRowIndex, { owner_id: '', ownership_shares: shares });
            setOwnershipRowIndex(null);
          }}
        />
      ) : null}

      {companyValuationRowIndex !== null ? (
        <CompanyValuationModal
          row={draftRows[companyValuationRowIndex]}
          targetDate={targetDate}
          onClose={() => setCompanyValuationRowIndex(null)}
          onApply={(value) => {
            updateDraftRow(companyValuationRowIndex, { value, quantity: 1 });
            setCompanyValuationRowIndex(null);
          }}
        />
      ) : null}

      {chartDefinition ? (
        <ChartSettingsModal
          definition={chartDefinition}
          settings={getSettings(chartDefinition.key, chartDefinition.categories, chartDefinition.includeTotal, chartDefinition.defaultHiddenSeries || [])}
          patchSettings={patchSettings}
          onClose={() => setChartDefinition(null)}
        />
      ) : null}

      {saveReviewOpen ? (
        <SaveReviewModal
          targetDate={targetDate}
          sourceDate={sourceDate}
          review={saveReview}
          onClose={() => setSaveReviewOpen(false)}
          onConfirm={() => saveSnapshot().catch((error) => setStatus({ message: error.message, tone: 'error' }))}
        />
      ) : null}

      {priceAudit ? (
        <PriceAuditModal
          rows={priceAudit}
          onClose={() => setPriceAudit(null)}
          onApply={applyPriceAudit}
        />
      ) : null}
    </div>
  );
}

function validateDraftRows(rows: SnapshotRow[]): DraftErrorMap {
  const errors: DraftErrorMap = {};
  rows.forEach((row, index) => {
    const rowErrors: string[] = [];
    if (!row.asset_id) rowErrors.push('Name is required');
    if (!row.category) rowErrors.push('Category is required');
    if (rowQuantity(row) < 0) rowErrors.push('Quantity cannot be negative');
    if (rowValue(row) < 0) rowErrors.push('Value cannot be negative');
    if (!row.owner_id) {
      const total = (row.ownership_shares || []).reduce((sum, share) => sum + Number(share.share || 0), 0);
      if (total > 1.000001) rowErrors.push('Split cannot exceed 100%');
    }
    if (rowErrors.length) errors[index] = rowErrors;
  });
  return errors;
}

function buildSaveReview(rows: SnapshotRow[], errors: DraftErrorMap) {
  const activeRows = rows.filter((row) => rowValue(row) > 0);
  const ignoredRows = rows.filter((row) => rowValue(row) <= 0);
  const groups = new Map<string, number>();
  activeRows.forEach((row) => {
    const key = `${row.asset_id}|${row.owner_id || 'shared'}|${row.broker || ''}`;
    groups.set(key, (groups.get(key) || 0) + 1);
  });
  const mergedRows = [...groups.values()].filter((count) => count > 1).reduce((sum, count) => sum + count - 1, 0);
  return {
    kept: activeRows.length,
    ignored: ignoredRows.length,
    merged: mergedRows,
    errors: Object.values(errors).flat().length,
  };
}

function LoginScreen({ onLogin }: { onLogin: (username: string, password: string) => Promise<void> }) {
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('');
  const [status, setStatus] = useState<Status>({ message: '', tone: 'muted' });
  const [showReset, setShowReset] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setStatus({ message: 'Signing in...', tone: 'muted' });
    try {
      await onLogin(username, password);
    } catch (error) {
      setStatus({ message: error instanceof Error ? error.message : 'Login failed', tone: 'error' });
    }
  };

  if (showReset) {
    return <PasswordResetRequestScreen onBack={() => setShowReset(false)} />;
  }

  return (
    <main className="login-shell">
      <section className="panel login-panel">
        <h1>Patrimonio</h1>
        <p className="muted">Sign in to manage private asset data.</p>
        <form className="form-grid" onSubmit={submit}>
          <label>Username<input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" /></label>
          <label>Password<input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" /></label>
          <button type="submit">Login</button>
          <button className="secondary" type="button" onClick={() => setShowReset(true)}>Reset password</button>
        </form>
        <StatusLine status={status} />
      </section>
    </main>
  );
}

function PasswordResetRequestScreen({ onBack }: { onBack: () => void }) {
  const [email, setEmail] = useState('');
  const [resetUrl, setResetUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>({ message: '', tone: 'muted' });

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setStatus({ message: 'Requesting password reset...', tone: 'muted' });
    try {
      const result = await api.requestPasswordReset(email);
      setResetUrl(result.dev_reset_url || null);
      setStatus({ message: result.message, tone: 'success' });
    } catch (error) {
      setStatus({ message: error instanceof Error ? error.message : 'Could not request a password reset.', tone: 'error' });
    }
  };

  return (
    <main className="login-shell">
      <section className="panel login-panel">
        <h1>Reset password</h1>
        <form className="form-grid" onSubmit={submit}>
          <label>Email<input required type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" /></label>
          <button type="submit">Send reset link</button>
          <button className="secondary" type="button" onClick={onBack}>Back to login</button>
        </form>
        {resetUrl ? <a className="button secondary" href={resetUrl}>Open reset link</a> : null}
        <StatusLine status={status} />
      </section>
    </main>
  );
}

function PasswordResetScreen({ token, onComplete }: { token: string; onComplete: () => void }) {
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [status, setStatus] = useState<Status>({ message: '', tone: 'muted' });

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (password !== confirmation) {
      setStatus({ message: 'Passwords do not match.', tone: 'error' });
      return;
    }
    try {
      const result = await api.confirmPasswordReset(token, password);
      setStatus({ message: result.message, tone: 'success' });
    } catch (error) {
      setStatus({ message: error instanceof Error ? error.message : 'Could not reset the password.', tone: 'error' });
    }
  };

  return (
    <main className="login-shell">
      <section className="panel login-panel">
        <h1>Choose a password</h1>
        <form className="form-grid" onSubmit={submit}>
          <label>New password<input required minLength={8} type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="new-password" /></label>
          <label>Confirm password<input required minLength={8} type="password" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} autoComplete="new-password" /></label>
          <button type="submit">Update password</button>
          <button className="secondary" type="button" onClick={onComplete}>Back to login</button>
        </form>
        <StatusLine status={status} />
      </section>
    </main>
  );
}

function BootstrapScreen({ onBootstrap }: { onBootstrap: (payload: { email: string; display_name: string; password: string; person_owner_id?: number; setup_token?: string }) => Promise<void> }) {
  const [owners, setOwners] = useState<Owner[]>([]);
  const [email, setEmail] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');
  const [personOwnerId, setPersonOwnerId] = useState('');
  const [setupToken, setSetupToken] = useState('');
  const [status, setStatus] = useState<Status>({ message: 'Preparing account setup...', tone: 'muted' });

  useEffect(() => {
    api.bootstrapOptions()
      .then((rows) => {
        setOwners(rows);
        setStatus({ message: '', tone: 'muted' });
      })
      .catch((error) => setStatus({ message: error instanceof Error ? error.message : 'Could not prepare account setup.', tone: 'error' }));
  }, []);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setStatus({ message: 'Creating the first account...', tone: 'muted' });
    try {
      await onBootstrap({
        email,
        display_name: displayName,
        password,
        person_owner_id: personOwnerId ? Number(personOwnerId) : undefined,
        setup_token: setupToken || undefined,
      });
    } catch (error) {
      setStatus({ message: error instanceof Error ? error.message : 'Could not create the account.', tone: 'error' });
    }
  };

  return (
    <main className="login-shell">
      <section className="panel login-panel">
        <h1>Patrimonio</h1>
        <p className="muted">Create the first project administrator.</p>
        <form className="form-grid" onSubmit={submit}>
          <label>Name<input required value={displayName} onChange={(event) => setDisplayName(event.target.value)} autoComplete="name" /></label>
          <label>Email<input required type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" /></label>
          <label>Password<input required type="password" minLength={8} value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="new-password" /></label>
          <label>Link to person entity
            <select value={personOwnerId} onChange={(event) => setPersonOwnerId(event.target.value)}>
              <option value="">Not linked yet</option>
              {owners.map((owner) => <option key={owner.id} value={owner.id}>{owner.name}</option>)}
            </select>
          </label>
          <label>Setup token<input type="password" value={setupToken} onChange={(event) => setSetupToken(event.target.value)} autoComplete="off" /></label>
          <button type="submit">Create administrator</button>
        </form>
        <StatusLine status={status} />
      </section>
    </main>
  );
}

function EntityOwnershipPanel({
  owners,
  rows,
  disabled,
  onCreate,
  onDelete,
}: {
  owners: Owner[];
  rows: EntityOwnership[];
  disabled: boolean;
  onCreate: (payload: Omit<EntityOwnership, 'id' | 'owner_name' | 'owned_name'>) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
}) {
  const peopleOrCompanies = owners;
  const companies = owners.filter((owner) => owner.type === 'company');
  const [ownerId, setOwnerId] = useState('');
  const [ownedId, setOwnedId] = useState('');
  const [share, setShare] = useState('50');
  const [effectiveFrom, setEffectiveFrom] = useState(new Date().toISOString().slice(0, 10));
  const [status, setStatus] = useState<Status>({ message: '', tone: 'muted' });

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    try {
      await onCreate({
        owner_id: Number(ownerId),
        owned_id: Number(ownedId),
        share: Number(share) / 100,
        effective_from: effectiveFrom,
        effective_to: null,
      });
      setStatus({ message: 'Ownership relationship saved.', tone: 'success' });
    } catch (error) {
      setStatus({ message: error instanceof Error ? error.message : 'Could not save the relationship.', tone: 'error' });
    }
  };

  return (
    <section className="panel">
      <h2>Company ownership</h2>
      <form className="form-grid ownership-form" onSubmit={submit}>
        <label>Owner
          <select required value={ownerId} disabled={disabled} onChange={(event) => setOwnerId(event.target.value)}>
            <option value="">Select entity</option>
            {peopleOrCompanies.map((owner) => <option key={owner.id} value={owner.id}>{owner.name}</option>)}
          </select>
        </label>
        <label>Company owned
          <select required value={ownedId} disabled={disabled} onChange={(event) => setOwnedId(event.target.value)}>
            <option value="">Select company</option>
            {companies.map((company) => <option key={company.id} value={company.id}>{company.name}</option>)}
          </select>
        </label>
        <label>Ownership %<input required min="0.01" max="100" step="0.01" type="number" value={share} disabled={disabled} onChange={(event) => setShare(event.target.value)} /></label>
        <label>Effective from<input required type="date" value={effectiveFrom} disabled={disabled} onChange={(event) => setEffectiveFrom(event.target.value)} /></label>
        <button type="submit" disabled={disabled}>Add relationship</button>
      </form>
      <StatusLine status={status} />
      <SortableTable
        id="entityOwnershipsTable"
        rows={rows}
        columns={[
          { key: 'owner', label: 'Owner', value: (row) => row.owner_name, render: (row) => row.owner_name },
          { key: 'company', label: 'Company owned', value: (row) => row.owned_name, render: (row) => row.owned_name },
          { key: 'share', label: 'Ownership', value: (row) => row.share, render: (row) => formatPercent(row.share * 100) },
          { key: 'from', label: 'Effective from', value: (row) => normalizeDate(row.effective_from), render: (row) => normalizeDate(row.effective_from) },
          { key: 'delete', label: 'Delete', value: () => '', render: (row) => <button className="danger" type="button" disabled={disabled} onClick={() => onDelete(row.id).catch((error) => setStatus({ message: error.message, tone: 'error' }))}>Delete</button> },
        ]}
      />
    </section>
  );
}

function ProjectAccessPanel() {
  const [users, setUsers] = useState<ProjectUser[]>([]);
  const [userEdits, setUserEdits] = useState<Record<number, { role: string; is_active: boolean }>>({});
  const [email, setEmail] = useState('');
  const [role, setRole] = useState('editor');
  const [invite, setInvite] = useState<ProjectInvitationResult | null>(null);
  const [status, setStatus] = useState<Status>({ message: 'Loading project access...', tone: 'muted' });

  const loadUsers = useCallback(async () => {
    try {
      const rows = await api.projectUsers();
      setUsers(rows);
      setUserEdits(Object.fromEntries(rows.map((user) => [user.id, {
        role: user.role || 'viewer',
        is_active: user.is_active,
      }])));
      setStatus({ message: '', tone: 'muted' });
    } catch (error) {
      setStatus({ message: error instanceof Error ? error.message : 'Could not load project users.', tone: 'error' });
    }
  }, []);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    try {
      const result = await api.inviteProjectUser({ email, role });
      setInvite(result);
      setEmail('');
      setStatus({ message: result.invite_url ? 'Invitation created. Send the link to the invited user.' : 'Invitation email sent.', tone: 'success' });
    } catch (error) {
      setStatus({ message: error instanceof Error ? error.message : 'Could not create an invitation.', tone: 'error' });
    }
  };

  const saveUser = async (user: ProjectUser) => {
    try {
      await api.updateProjectUser(user.id, userEdits[user.id]);
      await loadUsers();
      setStatus({ message: 'Project access updated.', tone: 'success' });
    } catch (error) {
      setStatus({ message: error instanceof Error ? error.message : 'Could not update project access.', tone: 'error' });
    }
  };

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>Project access</h2>
          <p className="muted">{users.length} active project user{users.length === 1 ? '' : 's'}.</p>
        </div>
      </div>
      <form className="form-grid" onSubmit={submit}>
        <label>Email<input required type="email" value={email} onChange={(event) => setEmail(event.target.value)} /></label>
        <label>Role
          <select value={role} onChange={(event) => setRole(event.target.value)}>
            <option value="viewer">Viewer</option>
            <option value="editor">Editor</option>
            <option value="admin">Administrator</option>
          </select>
        </label>
        <button type="submit">Invite user</button>
      </form>
      {invite?.invite_url ? (
        <label className="invite-link">
          Invitation link for {invite.email}
          <input readOnly value={invite.invite_url} onFocus={(event) => event.currentTarget.select()} />
        </label>
      ) : null}
      <StatusLine status={status} />
      <SortableTable
        id="projectUsersTable"
        rows={users}
        columns={[
          { key: 'name', label: 'Name', value: (user) => user.display_name, render: (user) => user.display_name },
          { key: 'email', label: 'Email', value: (user) => user.email, render: (user) => user.email },
          { key: 'role', label: 'Role', value: (user) => user.role || '', render: (user) => user.role === 'owner' ? 'Owner' : <select value={userEdits[user.id]?.role || 'viewer'} onChange={(event) => setUserEdits((edits) => ({ ...edits, [user.id]: { ...edits[user.id], role: event.target.value } }))}><option value="viewer">Viewer</option><option value="editor">Editor</option><option value="admin">Administrator</option></select> },
          { key: 'active', label: 'Active', value: (user) => user.is_active ? 1 : 0, render: (user) => user.role === 'owner' ? 'Yes' : <input type="checkbox" checked={!!userEdits[user.id]?.is_active} onChange={(event) => setUserEdits((edits) => ({ ...edits, [user.id]: { ...edits[user.id], is_active: event.target.checked } }))} /> },
          { key: 'save', label: 'Save', value: () => '', render: (user) => user.role === 'owner' ? null : <button className="secondary" type="button" onClick={() => saveUser(user)}>Save</button> },
        ]}
      />
    </section>
  );
}

function AdminTab({ authStatus, restoreResult, onRestore }: { authStatus: AuthStatus | null; restoreResult: RestoreResult | null; onRestore: (backup: unknown) => Promise<void> }) {
  const [auditRows, setAuditRows] = useState<AuditLog[]>([]);
  const [priceRows, setPriceRows] = useState<PriceHistory[]>([]);
  const [auditStatus, setAuditStatus] = useState<Status>({ message: 'Loading audit log...', tone: 'muted' });
  const [restoreStatus, setRestoreStatus] = useState<Status>({ message: '', tone: 'muted' });
  const [restoreText, setRestoreText] = useState('');
  const [confirmation, setConfirmation] = useState('');

  const loadAudit = useCallback(async () => {
    setAuditStatus({ message: 'Loading audit log...', tone: 'muted' });
    try {
      const [audit, prices] = await Promise.all([api.auditLog(), api.priceHistory()]);
      setAuditRows(audit);
      setPriceRows(prices);
      setAuditStatus({ message: 'Admin data loaded.', tone: 'success' });
    } catch (error) {
      setAuditStatus({ message: error instanceof Error ? error.message : 'Could not load audit log.', tone: 'error' });
    }
  }, []);

  useEffect(() => {
    loadAudit();
  }, [loadAudit]);

  const chooseFile = async (file?: File) => {
    if (!file) return;
    const text = await file.text();
    setRestoreText(text);
    setRestoreStatus({ message: `${file.name} loaded for restore review.`, tone: 'muted' });
  };

  const restore = async () => {
    if (confirmation !== 'RESTORE') {
      setRestoreStatus({ message: 'Type RESTORE before replacing the database.', tone: 'error' });
      return;
    }
    if (!authStatus?.restore_enabled) {
      setRestoreStatus({ message: 'Restore is disabled in this environment.', tone: 'error' });
      return;
    }
    try {
      const backup = JSON.parse(restoreText);
      await onRestore(backup);
      setConfirmation('');
      setRestoreText('');
      setRestoreStatus({ message: 'Backup restored.', tone: 'success' });
      await loadAudit();
    } catch (error) {
      setRestoreStatus({ message: error instanceof Error ? error.message : 'Restore failed.', tone: 'error' });
    }
  };

  return (
    <main>
      {authStatus?.role === 'owner' || authStatus?.role === 'admin' ? <ProjectAccessPanel /> : null}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Audit log</h2>
            <p className="muted">Recent API changes and restore events.</p>
          </div>
          <button className="secondary" type="button" onClick={loadAudit}>Refresh audit</button>
        </div>
        <StatusLine status={auditStatus} />
        <SortableTable
          id="auditTable"
          rows={auditRows}
          columns={[
            { key: 'created_at', label: 'Time', value: (row) => row.created_at, render: (row) => normalizeDate(row.created_at) },
            { key: 'actor', label: 'Actor', value: (row) => row.actor || '', render: (row) => row.actor || '' },
            { key: 'action', label: 'Action', value: (row) => row.action, render: (row) => row.action },
            { key: 'entity_type', label: 'Entity', value: (row) => row.entity_type || '', render: (row) => row.entity_type || '' },
            { key: 'entity_id', label: 'ID', value: (row) => row.entity_id || '', render: (row) => row.entity_id || '' },
            { key: 'details', label: 'Details', value: (row) => row.details || '', render: (row) => <span className="audit-details">{row.details || ''}</span> },
          ]}
        />
      </section>

      <section className="panel">
        <h2>Price history</h2>
        <p className="muted">Stored prices from successful provider quote checks.</p>
        <SortableTable
          id="priceHistoryTable"
          rows={priceRows}
          columns={[
            { key: 'created_at', label: 'Fetched', value: (row) => row.created_at, render: (row) => normalizeDate(row.created_at) },
            { key: 'provider', label: 'Provider', value: (row) => row.provider, render: (row) => row.provider },
            { key: 'symbol', label: 'Ticker/ID', value: (row) => row.symbol, render: (row) => row.symbol },
            { key: 'price', label: 'Price', value: (row) => row.price, render: (row) => formatNumber(row.price) },
            { key: 'currency', label: 'Currency', value: (row) => row.currency || '', render: (row) => row.currency || '' },
            { key: 'as_of', label: 'Provider date', value: (row) => row.as_of || '', render: (row) => row.as_of ? normalizeDate(row.as_of) : '' },
          ]}
        />
      </section>

      <section className="panel">
        <h2>Restore backup</h2>
        <p className="muted">Restoring replaces owners, assets, positions, ownership splits, valuations, and investing configuration.</p>
        {!authStatus?.restore_enabled ? <StatusLine status={{ message: 'Restore is disabled. Set APP_RESTORE_ENABLED=true only when you intentionally need it.', tone: 'muted' }} /> : null}
        <div className="toolbar">
          <label>Backup JSON<input type="file" accept="application/json,.json" disabled={!authStatus?.restore_enabled} onChange={(event) => chooseFile(event.target.files?.[0]).catch((error) => setRestoreStatus({ message: error.message, tone: 'error' }))} /></label>
          <label>Confirmation<input value={confirmation} placeholder="Type RESTORE" disabled={!authStatus?.restore_enabled} onChange={(event) => setConfirmation(event.target.value)} /></label>
          <button className="danger" type="button" disabled={!authStatus?.restore_enabled || !restoreText || confirmation !== 'RESTORE'} onClick={restore}>Restore backup</button>
        </div>
        <StatusLine status={restoreStatus} />
        {restoreResult ? (
          <div className="review-grid">
            <MetricCard label="Owners" value={String(restoreResult.owners)} />
            <MetricCard label="Assets" value={String(restoreResult.assets)} />
            <MetricCard label="Positions" value={String(restoreResult.positions)} />
            <MetricCard label="Investing categories" value={String(restoreResult.investing_assets)} />
          </div>
        ) : null}
      </section>
    </main>
  );
}

function DataEntryTab({
  assets,
  owners,
  entityOwnerships,
  investingAssets,
  draftRows,
  draftErrors,
  draftDirty,
  removedCount,
  targetDate,
  sourceDate,
  viewDate,
  ownerStatus,
  assetStatus,
  ownerForm,
  assetForm,
  ownerEdits,
  assetEdits,
  saveReview,
  setTargetDate,
  setOwnerForm,
  setAssetForm,
  setOwnerEdits,
  setAssetEdits,
  onLoadSelected,
  onLoadPrevious,
  onAddRow,
  onUpdateRow,
  onRemoveRow,
  onUndoRemove,
  onOpenSplit,
  onOpenCompanyValuation,
  onFetchPrices,
  onOpenSaveReview,
  onCreateOwner,
  onCreateAsset,
  onUpdateOwner,
  onUpdateAsset,
  onUpdateInvestingAsset,
  onDeleteOwner,
  onDeleteAsset,
  onCreateEntityOwnership,
  onDeleteEntityOwnership,
}: {
  assets: Asset[];
  owners: Owner[];
  entityOwnerships: EntityOwnership[];
  investingAssets: InvestingAsset[];
  draftRows: SnapshotRow[];
  draftErrors: DraftErrorMap;
  draftDirty: boolean;
  removedCount: number;
  targetDate: string;
  sourceDate: string;
  viewDate: string;
  status: Status;
  ownerStatus: Status;
  assetStatus: Status;
  ownerForm: Partial<Owner>;
  assetForm: Partial<Asset>;
  ownerEdits: Record<number, Partial<Owner>>;
  assetEdits: Record<number, Partial<Asset>>;
  saveReview: ReturnType<typeof buildSaveReview>;
  setTargetDate: (date: string) => void;
  setOwnerForm: (form: Partial<Owner>) => void;
  setAssetForm: (form: Partial<Asset>) => void;
  setOwnerEdits: React.Dispatch<React.SetStateAction<Record<number, Partial<Owner>>>>;
  setAssetEdits: React.Dispatch<React.SetStateAction<Record<number, Partial<Asset>>>>;
  onLoadSelected: () => void;
  onLoadPrevious: () => void;
  onAddRow: () => void;
  onUpdateRow: (index: number, patch: Partial<SnapshotRow>) => void;
  onRemoveRow: (index: number) => void;
  onUndoRemove: () => void;
  onOpenSplit: (index: number) => void;
  onOpenCompanyValuation: (index: number) => void;
  onFetchPrices: () => void;
  onOpenSaveReview: () => void;
  onCreateOwner: (event: React.FormEvent) => void;
  onCreateAsset: (event: React.FormEvent) => void;
  onUpdateOwner: (id: number) => void;
  onUpdateAsset: (id: number) => void;
  onUpdateInvestingAsset: (category: string, isInvested: boolean) => void;
  onDeleteOwner: (id: number) => void;
  onDeleteAsset: (id: number) => void;
  onCreateEntityOwnership: (payload: Omit<EntityOwnership, 'id' | 'owner_name' | 'owned_name'>) => Promise<void>;
  onDeleteEntityOwnership: (id: number) => Promise<void>;
}) {
  const categories = useMemo(() => [...new Set(assets.map((asset) => asset.category || '').filter(Boolean))].sort(), [assets]);
  const investingRows = useMemo(() => {
    const configured = new Map(investingAssets.map((item) => [item.category, item.is_invested]));
    return categories.map((category) => ({
      category,
      is_invested: configured.get(category) ?? defaultCategoryIsInvested(category),
    }));
  }, [categories, investingAssets]);

  return (
    <main>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <h2>Snapshot data</h2>
            <p className="muted">Editing {targetDate || 'no target date'} using {sourceDate || viewDate || 'no source date'} as template.</p>
          </div>
          {draftDirty ? <div className="dirty-banner">Unsaved draft changes</div> : null}
        </div>
        <div className="toolbar">
          <label>
            Draft target date
            <input type="date" value={targetDate} onChange={(event) => setTargetDate(event.target.value)} />
          </label>
          <button className="secondary" type="button" onClick={onLoadSelected}>Load viewing date</button>
          <button className="secondary" type="button" onClick={onLoadPrevious}>Use previous as template</button>
          <button className="secondary" type="button" onClick={onAddRow}><Plus size={16} /> Add row</button>
          <button className="secondary" type="button" onClick={onFetchPrices}>Apply stock prices</button>
          <button className="secondary" type="button" onClick={onUndoRemove} disabled={!removedCount}><RotateCcw size={16} /> Undo remove</button>
          <button type="button" onClick={onOpenSaveReview} disabled={!!saveReview.errors}><Save size={16} /> Replace snapshot</button>
        </div>
        <div className="preview-grid">
          <div className="preview-item"><span className="label">Rows kept</span><strong>{saveReview.kept}</strong></div>
          <div className="preview-item"><span className="label">Ignored zero rows</span><strong>{saveReview.ignored}</strong></div>
          <div className="preview-item"><span className="label">Merged duplicates</span><strong>{saveReview.merged}</strong></div>
          <div className="preview-item"><span className="label">Row errors</span><strong>{saveReview.errors}</strong></div>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Row</th>
                <th>Category</th>
                <th>Name</th>
                <th>Holder</th>
                <th>Quantity</th>
                <th>Value</th>
                <th>Broker</th>
                <th>Errors</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {draftRows.map((row, index) => (
                <tr key={`${row.position_id || 'new'}-${index}`} className={draftErrors[index]?.length ? 'row-error' : ''}>
                  <td>{index + 1}</td>
                  <td>
                    <select value={row.category || ''} onChange={(event) => onUpdateRow(index, { category: event.target.value })}>
                      <option value="">Category</option>
                      {categories.map((category) => <option key={category} value={category}>{category}</option>)}
                    </select>
                  </td>
                  <td>
                    <select value={row.asset_id} onChange={(event) => onUpdateRow(index, { asset_id: Number(event.target.value) })}>
                      <option value="">Asset</option>
                      {assets.map((asset) => <option key={asset.id} value={asset.id}>{asset.name}</option>)}
                    </select>
                  </td>
                  <td>
                    <div className="owner-cell">
                      <select value={row.owner_id ?? ''} onChange={(event) => onUpdateRow(index, { owner_id: event.target.value ? Number(event.target.value) : '' })}>
                        <option value="">Common / shared holding</option>
                        {owners.map((owner) => <option key={owner.id} value={owner.id}>{owner.name}</option>)}
                      </select>
                      {!row.owner_id ? <button className="secondary split-button" type="button" onClick={() => onOpenSplit(index)}>Split</button> : null}
                      {!row.owner_id ? <span className="split-summary">{splitSummary(row, owners)}</span> : null}
                    </div>
                  </td>
                  <td>{isStockRow(row) ? <input type="number" step="0.01" value={row.quantity} onChange={(event) => onUpdateRow(index, { quantity: Number(event.target.value) })} /> : null}</td>
                  <td><input type="number" step="0.01" value={row.value} onChange={(event) => onUpdateRow(index, { value: Number(event.target.value) })} /></td>
                  <td><input value={row.broker || ''} onChange={(event) => onUpdateRow(index, { broker: event.target.value })} /></td>
                  <td className="error-text">{draftErrors[index]?.join('. ')}</td>
                  <td>
                    <div className="row-actions">
                      {usesNetValuation(row) ? <button className="secondary" type="button" onClick={() => onOpenCompanyValuation(index)}>Valuation</button> : null}
                      <button className="danger icon-button" type="button" onClick={() => onRemoveRow(index)} aria-label={`Remove row ${index + 1}`}><Trash2 size={16} /></button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="catalog-stack">
        <section className="panel">
          <h2>Assets</h2>
          <form className="form-grid" onSubmit={onCreateAsset}>
            <label>Name<input required value={assetForm.name || ''} onChange={(event) => setAssetForm({ ...assetForm, name: event.target.value })} /></label>
            <label>Category<input required value={assetForm.category || ''} onChange={(event) => setAssetForm({ ...assetForm, category: event.target.value })} /></label>
            <AssetTypeFields asset={assetForm} onChange={setAssetForm} />
            <button type="submit" disabled={draftDirty}>Add asset</button>
          </form>
          <StatusLine status={assetStatus} />
          <SortableTable
            id="assetsTable"
            rows={assets}
            columns={[
              { key: 'name', label: 'Name', value: (asset) => asset.name, render: (asset) => <input value={assetEdits[asset.id]?.name || ''} onChange={(event) => setAssetEdits((edits) => ({ ...edits, [asset.id]: { ...edits[asset.id], name: event.target.value } }))} /> },
              { key: 'category', label: 'Category', value: (asset) => asset.category || '', render: (asset) => <input value={assetEdits[asset.id]?.category || ''} onChange={(event) => setAssetEdits((edits) => ({ ...edits, [asset.id]: { ...edits[asset.id], category: event.target.value } }))} /> },
              { key: 'type', label: 'Type', value: (asset) => asset.asset_type || '', render: (asset) => <AssetTypeSelect asset={assetEdits[asset.id] || asset} onChange={(next) => setAssetEdits((edits) => ({ ...edits, [asset.id]: next }))} /> },
              { key: 'valuation', label: 'Valuation', value: (asset) => asset.valuation_method || '', render: (asset) => <ValuationSelect asset={assetEdits[asset.id] || asset} onChange={(next) => setAssetEdits((edits) => ({ ...edits, [asset.id]: next }))} /> },
              { key: 'stock', label: 'Stock config', value: (asset) => `${asset.price_provider || ''} ${asset.price_symbol || ''}`, render: (asset) => <StockConfigFields asset={assetEdits[asset.id] || asset} onChange={(next) => setAssetEdits((edits) => ({ ...edits, [asset.id]: next }))} /> },
              { key: 'save', label: 'Save', value: () => '', render: (asset) => <button className="secondary" type="button" disabled={draftDirty} onClick={() => onUpdateAsset(asset.id)}>Save</button> },
              { key: 'delete', label: 'Delete', value: () => '', render: (asset) => <button className="danger" type="button" disabled={draftDirty} onClick={() => onDeleteAsset(asset.id)}>Delete</button> },
            ]}
          />
        </section>

        <div className="catalog-grid">
          <section className="panel">
            <h2>People & companies</h2>
            <form className="form-grid" onSubmit={onCreateOwner}>
              <label>Name<input required value={ownerForm.name || ''} onChange={(event) => setOwnerForm({ ...ownerForm, name: event.target.value })} /></label>
              <label>Type
                <select value={ownerForm.type || 'person'} onChange={(event) => setOwnerForm({ ...ownerForm, type: event.target.value })}>
                  <option value="person">Person</option>
                  <option value="company">Company</option>
                </select>
              </label>
              <button type="submit" disabled={draftDirty}>Add owner</button>
            </form>
            <StatusLine status={ownerStatus} />
            <SortableTable
              id="ownersTable"
              rows={owners}
              columns={[
                { key: 'name', label: 'Name', value: (owner) => owner.name, render: (owner) => <input value={ownerEdits[owner.id]?.name || ''} onChange={(event) => setOwnerEdits((edits) => ({ ...edits, [owner.id]: { ...edits[owner.id], name: event.target.value } }))} /> },
                { key: 'type', label: 'Type', value: (owner) => owner.type || '', render: (owner) => <select value={ownerEdits[owner.id]?.type || 'person'} onChange={(event) => setOwnerEdits((edits) => ({ ...edits, [owner.id]: { ...edits[owner.id], type: event.target.value } }))}><option value="person">Person</option><option value="company">Company</option></select> },
                { key: 'family', label: 'Family', value: (owner) => owner.is_family_member ? 1 : 0, render: (owner) => <input type="checkbox" checked={!!ownerEdits[owner.id]?.is_family_member} onChange={(event) => setOwnerEdits((edits) => ({ ...edits, [owner.id]: { ...edits[owner.id], is_family_member: event.target.checked } }))} /> },
                { key: 'save', label: 'Save', value: () => '', render: (owner) => <button className="secondary" type="button" disabled={draftDirty} onClick={() => onUpdateOwner(owner.id)}>Save</button> },
                { key: 'delete', label: 'Delete', value: () => '', render: (owner) => <button className="danger" type="button" disabled={draftDirty} onClick={() => onDeleteOwner(owner.id)}>Delete</button> },
              ]}
            />
          </section>

          <EntityOwnershipPanel
            owners={owners}
            rows={entityOwnerships}
            disabled={draftDirty}
            onCreate={onCreateEntityOwnership}
            onDelete={onDeleteEntityOwnership}
          />

          <section className="panel">
            <h2>Investing_Assets</h2>
            <p className="muted">Category-level investing classification used by the Investing KPI and default chart view.</p>
            <SortableTable
              id="investingAssetsTable"
              rows={investingRows}
              columns={[
                { key: 'category', label: 'Category', value: (row) => row.category, render: (row) => row.category },
                { key: 'is_invested', label: 'Is invested', value: (row) => row.is_invested ? 1 : 0, render: (row) => (
                  <select value={row.is_invested ? 'yes' : 'no'} disabled={draftDirty} onChange={(event) => onUpdateInvestingAsset(row.category, event.target.value === 'yes')}>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                ) },
              ]}
            />
          </section>
        </div>
      </div>
    </main>
  );
}

function updateAssetConfig(asset: Partial<Asset>, patch: Partial<Asset>) {
  const next = { ...asset, ...patch };
    if (patch.asset_type !== undefined) {
      next.asset_type = canonicalAssetType(patch.asset_type);
    }
    if (canonicalAssetType(patch.asset_type) === 'stock') {
      next.valuation_method = 'price_provider';
      if (!next.price_provider || next.price_provider === 'manual') next.price_provider = 'yahoo';
    }
  if (patch.asset_type && canonicalAssetType(patch.asset_type) !== 'stock') {
    next.price_provider = 'manual';
    next.price_symbol = '';
  }
  return next;
}

function AssetTypeFields({ asset, onChange }: { asset: Partial<Asset>; onChange: (asset: Partial<Asset>) => void }) {
  const isStock = canonicalAssetType(asset.asset_type) === 'stock';

  return (
    <>
      <AssetTypeSelect asset={asset} onChange={onChange} />
      <ValuationSelect asset={asset} onChange={onChange} />
      {isStock ? <StockConfigFields asset={asset} onChange={onChange} /> : null}
    </>
  );
}

function AssetTypeSelect({ asset, onChange }: { asset: Partial<Asset>; onChange: (asset: Partial<Asset>) => void }) {
  const update = (patch: Partial<Asset>) => {
    onChange(updateAssetConfig(asset, patch));
  };

  return (
    <select value={canonicalAssetType(asset.asset_type)} onChange={(event) => update({ asset_type: event.target.value })}>
      {assetTypes.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
    </select>
  );
}

function ValuationSelect({ asset, onChange }: { asset: Partial<Asset>; onChange: (asset: Partial<Asset>) => void }) {
  return (
    <select value={asset.valuation_method || 'market_direct'} onChange={(event) => onChange({ ...asset, valuation_method: event.target.value })}>
      {valuationMethods.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
    </select>
  );
}

function StockConfigFields({ asset, onChange }: { asset: Partial<Asset>; onChange: (asset: Partial<Asset>) => void }) {
  const isStock = canonicalAssetType(asset.asset_type) === 'stock';
  if (!isStock) return <span className="muted">Only for stocks</span>;
  return (
    <div className="stock-config">
      <label>
        Provider
        <select value={asset.price_provider || 'manual'} onChange={(event) => update({ price_provider: event.target.value })}>
          <option value="manual">Manual</option>
          <option value="yahoo">Yahoo</option>
        </select>
      </label>
      <label>
        Ticker / ID
        <input value={asset.price_symbol || ''} onChange={(event) => update({ price_symbol: event.target.value.trim().toUpperCase() })} placeholder="SAN.MC" />
      </label>
    </div>
  );

  function update(patch: Partial<Asset>) {
    onChange({ ...asset, ...patch });
  }
}

function splitSummary(row: SnapshotRow, owners: Owner[]) {
  const shares = row.ownership_shares || [];
  if (!shares.length) return 'Default family split';
  return shares.map((share) => {
    const owner = owners.find((item) => item.id === share.owner_id);
    return `${owner?.name || share.owner_name || 'Owner'} ${formatPercent(share.share * 100)}`;
  }).join(', ');
}

function OwnershipModal({ row, owners, onClose, onSave }: { row: SnapshotRow; owners: Owner[]; onClose: () => void; onSave: (shares: SnapshotRow['ownership_shares']) => void }) {
  const [shares, setShares] = useState(() => {
    const existing = row.ownership_shares?.length ? row.ownership_shares : buildDefaultShares(owners);
    return owners.map((owner) => ({
      owner_id: owner.id,
      owner_name: owner.name,
      share: existing.find((item) => item.owner_id === owner.id)?.share || 0,
    }));
  });
  const total = shares.reduce((sum, share) => sum + Number(share.share || 0), 0);

  return (
      <Modal title="Position allocation" subtitle={row.asset_name || 'Shared position'} onClose={onClose}>
      <div className="split-list">
        {shares.map((share, index) => (
          <label key={share.owner_id} className="ownership-row">
            {share.owner_name}
            <input
              type="number"
              min="0"
              max="100"
              step="0.01"
              value={formatNumber(share.share * 100).replace(',', '.')}
              onChange={(event) => setShares((current) => current.map((item, rowIndex) => rowIndex === index ? { ...item, share: Number(event.target.value || 0) / 100 } : item))}
            />
          </label>
        ))}
      </div>
      <StatusLine status={{ message: `Total split: ${formatPercent(total * 100)}`, tone: total > 1 ? 'error' : 'muted' }} />
      <div className="modal-actions">
        <button className="secondary" type="button" onClick={onClose}>Cancel</button>
        <button type="button" disabled={total > 1} onClick={() => onSave(shares.filter((share) => share.share > 0))}>Save split</button>
      </div>
    </Modal>
  );
}

function CompanyValuationModal({
  row,
  targetDate,
  onClose,
  onApply,
}: {
  row: SnapshotRow;
  targetDate: string;
  onClose: () => void;
  onApply: (value: number) => void;
}) {
  const [items, setItems] = useState<CompanyValuationItem[]>([]);
  const [status, setStatus] = useState<Status>({ message: 'Loading valuation items...', tone: 'muted' });
  const assetId = Number(row.asset_id || 0);
  const isMarketMinusDebt = row.valuation_method === 'market_minus_debt';
  const valuationTitle = isMarketMinusDebt ? 'Market minus debt' : 'Net asset valuation';
  const assetLabel = isMarketMinusDebt ? 'Market value' : 'Assets';
  const liabilityLabel = isMarketMinusDebt ? 'Debt' : 'Liabilities';

  useEffect(() => {
    if (!assetId || !targetDate) {
      setStatus({ message: 'Choose an asset and target date first.', tone: 'error' });
      return;
    }
    api.companyValuation(assetId, targetDate)
      .then((valuation) => {
        setItems(valuation.items.length ? valuation.items : isMarketMinusDebt ? [
          { item_type: 'asset', name: 'Market value', amount: 0 },
          { item_type: 'liability', name: 'Mortgage/debt', amount: 0 },
        ] : [
          { item_type: 'asset', name: '', amount: 0 },
          { item_type: 'liability', name: '', amount: 0 },
        ]);
        setStatus({ message: 'Valuation loaded.', tone: 'success' });
      })
      .catch((error) => setStatus({ message: error.message, tone: 'error' }));
  }, [assetId, targetDate, isMarketMinusDebt]);

  const assetItems = items.filter((item) => item.item_type === 'asset');
  const liabilityItems = items.filter((item) => item.item_type === 'liability');
  const assetsTotal = assetItems.reduce((sum, item) => sum + Number(item.amount || 0), 0);
  const liabilitiesTotal = liabilityItems.reduce((sum, item) => sum + Number(item.amount || 0), 0);
  const netValue = assetsTotal - liabilitiesTotal;

  const updateItem = (globalIndex: number, patch: Partial<CompanyValuationItem>) => {
    setItems((current) => current.map((item, index) => index === globalIndex ? { ...item, ...patch } : item));
  };

  const removeItem = (globalIndex: number) => {
    setItems((current) => current.filter((_, index) => index !== globalIndex));
  };

  const save = async () => {
    const cleaned = items
      .map((item) => ({ ...item, name: item.name.trim(), amount: Number(item.amount || 0) }))
      .filter((item) => item.name || item.amount);
    const saved = await api.saveCompanyValuation(assetId, targetDate, cleaned);
    onApply(saved.net_value);
  };

  return (
    <Modal title={valuationTitle} subtitle={`${row.asset_name || 'Asset'} on ${targetDate}`} onClose={onClose}>
      <StatusLine status={status} />
      <div className="review-grid">
        <MetricCard label={assetLabel} value={formatCurrency(assetsTotal)} />
        <MetricCard label={liabilityLabel} value={formatCurrency(liabilitiesTotal)} />
        <MetricCard label="Net value" value={formatCurrency(netValue)} />
      </div>
      <div className="split">
        <ValuationItemsTable
          title={assetLabel}
          items={assetItems}
          allItems={items}
          itemType="asset"
          onAdd={() => setItems((current) => [...current, { item_type: 'asset', name: '', amount: 0 }])}
          onUpdate={updateItem}
          onRemove={removeItem}
        />
        <ValuationItemsTable
          title={liabilityLabel}
          items={liabilityItems}
          allItems={items}
          itemType="liability"
          onAdd={() => setItems((current) => [...current, { item_type: 'liability', name: '', amount: 0 }])}
          onUpdate={updateItem}
          onRemove={removeItem}
        />
      </div>
      <div className="modal-actions">
        <button className="secondary" type="button" onClick={onClose}>Cancel</button>
        <button type="button" disabled={!assetId || !targetDate} onClick={() => save().catch((error) => setStatus({ message: error.message, tone: 'error' }))}>Save and use net value</button>
      </div>
    </Modal>
  );
}

function ValuationItemsTable({
  title,
  items,
  allItems,
  itemType,
  onAdd,
  onUpdate,
  onRemove,
}: {
  title: string;
  items: CompanyValuationItem[];
  allItems: CompanyValuationItem[];
  itemType: 'asset' | 'liability';
  onAdd: () => void;
  onUpdate: (index: number, patch: Partial<CompanyValuationItem>) => void;
  onRemove: (index: number) => void;
}) {
  return (
    <section className="valuation-box">
      <div className="chart-header">
        <h3>{title}</h3>
        <button className="secondary" type="button" onClick={onAdd}><Plus size={15} /> Add</button>
      </div>
      <table>
        <thead><tr><th>Name</th><th>Amount</th><th></th></tr></thead>
        <tbody>
          {items.map((item) => {
            const globalIndex = allItems.findIndex((candidate) => candidate === item);
            return (
              <tr key={globalIndex}>
                <td><input value={item.name} onChange={(event) => onUpdate(globalIndex, { name: event.target.value, item_type: itemType })} /></td>
                <td><input type="number" step="0.01" value={item.amount} onChange={(event) => onUpdate(globalIndex, { amount: Number(event.target.value), item_type: itemType })} /></td>
                <td><button className="danger icon-button" type="button" onClick={() => onRemove(globalIndex)} aria-label={`Remove ${title} item`}><Trash2 size={15} /></button></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

function SaveReviewModal({ targetDate, sourceDate, review, onClose, onConfirm }: { targetDate: string; sourceDate: string; review: ReturnType<typeof buildSaveReview>; onClose: () => void; onConfirm: () => void }) {
  return (
    <Modal title="Replace snapshot" subtitle={`Target ${targetDate}. Template source ${sourceDate || 'not set'}.`} onClose={onClose}>
      <div className="review-grid">
        <MetricCard label="Rows kept" value={String(review.kept)} />
        <MetricCard label="Zero rows ignored" value={String(review.ignored)} />
        <MetricCard label="Duplicate rows merged" value={String(review.merged)} />
        <MetricCard label="Errors" value={String(review.errors)} />
      </div>
      <p className="warning-text">Saving replaces the full snapshot for this date. Rows not included for this date can be deleted by the backend.</p>
      <div className="modal-actions">
        <button className="secondary" type="button" onClick={onClose}>Cancel</button>
        <button type="button" disabled={review.errors > 0} onClick={onConfirm}>Confirm replace</button>
      </div>
    </Modal>
  );
}

function PriceAuditModal({ rows, onClose, onApply }: { rows: PriceAuditRow[]; onClose: () => void; onApply: () => void }) {
  return (
    <Modal title="Stock price review" subtitle="Review provider prices before changing the snapshot." onClose={onClose}>
      <SortableTable
        id="priceAuditTable"
        rows={rows}
        columns={[
          { key: 'asset', label: 'Asset', value: (row) => row.assetName, render: (row) => row.assetName },
          { key: 'provider', label: 'Provider', value: (row) => row.provider, render: (row) => `${row.provider} / ${row.symbol}` },
          { key: 'quote', label: 'Quote', value: (row) => row.unitPrice, render: (row) => `${formatNumber(row.unitPrice)} ${row.currency || ''}` },
          { key: 'formula', label: 'Formula', value: (row) => row.newValue, render: (row) => `${formatNumber(row.quantity)} x ${formatNumber(row.unitPrice)} = ${formatCurrency(row.newValue)}` },
          { key: 'old', label: 'Previous', value: (row) => row.oldValue, render: (row) => formatCurrency(row.oldValue) },
          { key: 'date', label: 'Quote date', value: (row) => row.asOf || '', render: (row) => row.asOf || '' },
        ]}
      />
      <div className="modal-actions">
        <button className="secondary" type="button" onClick={onClose}>Cancel</button>
        <button type="button" onClick={onApply}>Apply prices</button>
      </div>
    </Modal>
  );
}

function OwnerDashboard({
  ownerName,
  viewDate,
  details,
  history,
  investmentByCategory,
  getSettings,
  patchSettings,
  openChartSettings,
}: {
  ownerName: string;
  viewDate: string;
  details: DashboardDetailRow[];
  history: DashboardHistoryPoint[];
  investmentByCategory: Record<string, boolean>;
  getSettings: (key: string, categories: string[], includeTotal: boolean, defaultHiddenSeries?: string[]) => ChartSettings;
  patchSettings: (key: string, patch: Partial<ChartSettings>) => void;
  openChartSettings: (definition: ChartDefinition) => void;
}) {
  const rows = details.filter((row) => ownerMatches(row, ownerName));
  const ownerHistory = history.map((item) => {
    const historyDetails = item.details.filter((row) => ownerMatches(row, ownerName));
    return {
      date: item.date,
      details: historyDetails,
      total: historyDetails.reduce((sum, row) => sum + rowValue(row), 0),
      investing: investmentValue(historyDetails, investmentByCategory),
    };
  });
  const categories = categoryList(ownerHistory);
  const metrics = {
    total: rows.reduce((sum, row) => sum + rowValue(row), 0),
    investing: investmentValue(rows, investmentByCategory),
  };
  const previous = previousMetric(ownerHistory, viewDate);
  const movers = largestMovers(ownerHistory, viewDate);

  return (
    <main>
      <div className="grid">
        <MetricCard label="Total assets" value={formatCurrency(metrics.total)} sub={deltaText(metrics.total, previous?.total)} />
        <MetricCard label="Investing" value={formatCurrency(metrics.investing)} sub={deltaText(metrics.investing, previous?.investing)} />
        <MetricCard label="Positions" value={String(rows.length)} />
      </div>
      <DashboardInsights movers={movers} />
      <ChartRows
        prefix={ownerName.toLowerCase()}
        rows={rows}
        historyRows={ownerHistory}
        categories={categories}
        investmentByCategory={investmentByCategory}
        getSettings={getSettings}
        patchSettings={patchSettings}
        openChartSettings={openChartSettings}
      />
      <section className="panel">
        <h2>Positions as of {viewDate}</h2>
        <PositionsTable rows={rows} />
      </section>
      <HistoryTables historyRows={ownerHistory} categories={categories} />
    </main>
  );
}

function AggregateDashboard({
  viewDate,
  owners,
  summary,
  details,
  history,
  investmentByCategory,
  getSettings,
  patchSettings,
  openChartSettings,
}: {
  viewDate: string;
  owners: Owner[];
  summary: DashboardSummary | null;
  details: DashboardDetailRow[];
  history: DashboardHistoryPoint[];
  investmentByCategory: Record<string, boolean>;
  getSettings: (key: string, categories: string[], includeTotal: boolean, defaultHiddenSeries?: string[]) => ChartSettings;
  patchSettings: (key: string, patch: Partial<ChartSettings>) => void;
  openChartSettings: (definition: ChartDefinition) => void;
}) {
  const categories = categoryList(history);
  const investing = investmentValue(details, investmentByCategory);
  const historyRows = history.map((item) => ({
    date: item.date,
    details: item.details,
    total: item.details.reduce((sum, row) => sum + rowValue(row), 0),
    investing: investmentValue(item.details, investmentByCategory),
  }));
  const previous = previousMetric(historyRows, viewDate);
  const rows = groupedAggregateRows(details);
  const movers = largestMovers(historyRows, viewDate);
  const byOwner = summary?.by_owner || [];

  return (
    <main>
      <div className="grid">
        <MetricCard label="Total assets" value={formatCurrency(summary?.total_value || 0)} sub={deltaText(summary?.total_value || 0, previous?.total)} />
        <MetricCard label="Investing" value={formatCurrency(investing)} sub={deltaText(investing, previous?.investing)} />
        <MetricCard label="Positions" value={String(details.length)} />
        <MetricCard label="Owners" value={String(owners.length)} />
      </div>
      <DashboardInsights movers={movers} byOwner={byOwner} />
      <ChartRows
        prefix="aggregate"
        rows={details}
        historyRows={historyRows}
        categories={categories}
        investmentByCategory={investmentByCategory}
        getSettings={getSettings}
        patchSettings={patchSettings}
        openChartSettings={openChartSettings}
      />
      <div className="split">
        <BarChartPanel
          title="By category"
          chartKey="aggregateCategoryBar"
          rows={summary?.by_category || []}
          labelKey="category"
          getSettings={getSettings}
          openChartSettings={openChartSettings}
        />
        <BarChartPanel
          title="By broker/bank"
          chartKey="aggregateBrokerBar"
          rows={summary?.by_broker || []}
          labelKey="broker"
          getSettings={getSettings}
          openChartSettings={openChartSettings}
        />
      </div>
      <section className="panel">
        <h2>Aggregated positions</h2>
        <SortableTable
          id="aggregatePositions"
          rows={rows}
          columns={[
            { key: 'asset', label: 'Asset', value: (row) => row.asset, render: (row) => row.asset },
            { key: 'category', label: 'Category', value: (row) => row.category, render: (row) => row.category },
            { key: 'broker', label: 'Broker', value: (row) => row.broker, render: (row) => row.broker },
            { key: 'quantity', label: 'Quantity', value: (row) => isStockRow(row) ? row.quantity : 0, render: (row) => stockQuantityLabel(row) },
            { key: 'value', label: 'Value', value: (row) => row.value, render: (row) => formatCurrency(row.value) },
            { key: 'pct', label: 'Value %', value: (row) => row.valuePct, render: (row) => formatPercent(row.valuePct) },
          ]}
        />
      </section>
      <HistoryTables historyRows={historyRows} categories={categories} />
    </main>
  );
}

function ChartRows({
  prefix,
  rows,
  historyRows,
  categories,
  investmentByCategory,
  getSettings,
  patchSettings,
  openChartSettings,
}: {
  prefix: string;
  rows: DashboardDetailRow[];
  historyRows: Array<{ date: string; details: DashboardDetailRow[]; total: number; investing: number }>;
  categories: string[];
  investmentByCategory: Record<string, boolean>;
  getSettings: (key: string, categories: string[], includeTotal: boolean, defaultHiddenSeries?: string[]) => ChartSettings;
  patchSettings: (key: string, patch: Partial<ChartSettings>) => void;
  openChartSettings: (definition: ChartDefinition) => void;
}) {
  const investingDefaultHidden = categories.filter((category) => !isCategoryInvested(category, investmentByCategory));
  return (
    <>
      <div className="split">
        <LineChartPanel
          title="Total assets by category"
          chartKey={`${prefix}TotalLine`}
          categories={categories}
          historyRows={historyRows}
          colorDomain={categories}
          getSettings={getSettings}
          openChartSettings={openChartSettings}
        />
        <PieChartPanel
          title="Latest NAV by category"
          chartKey={`${prefix}TotalPie`}
          rows={rows}
          categories={categories}
          colorDomain={categories}
          getSettings={getSettings}
          openChartSettings={openChartSettings}
        />
      </div>
      <div className="split">
        <LineChartPanel
          title="Investing by category"
          chartKey={`${prefix}InvestingLineV2`}
          categories={categories}
          historyRows={historyRows}
          colorDomain={categories}
          defaultHiddenSeries={investingDefaultHidden}
          getSettings={getSettings}
          openChartSettings={openChartSettings}
        />
        <PieChartPanel
          title="Latest investing NAV by category"
          chartKey={`${prefix}InvestingPieV2`}
          rows={rows}
          categories={categories}
          colorDomain={categories}
          defaultHiddenSeries={investingDefaultHidden}
          getSettings={getSettings}
          openChartSettings={openChartSettings}
        />
      </div>
    </>
  );
}

function LineChartPanel({
  title,
  chartKey,
  categories,
  historyRows,
  colorDomain,
  defaultHiddenSeries = [],
  getSettings,
  openChartSettings,
}: {
  title: string;
  chartKey: string;
  categories: string[];
  historyRows: Array<{ date: string; details: DashboardDetailRow[] }>;
  colorDomain: string[];
  defaultHiddenSeries?: string[];
  getSettings: (key: string, categories: string[], includeTotal: boolean, defaultHiddenSeries?: string[]) => ChartSettings;
  openChartSettings: (definition: ChartDefinition) => void;
}) {
  const settings = getSettings(chartKey, categories, true, defaultHiddenSeries);
  const visibleCategories = categories.filter((category) => !settings.hiddenSeries.includes(category));
  const includeTotal = !settings.hiddenSeries.includes(TOTAL_SERIES_KEY);
  const series = buildLineSeries(historyRows, visibleCategories, settings, includeTotal);
  return (
    <section className="panel chart-panel">
      <div className="chart-header">
        <h2>{title}</h2>
        <button className="secondary icon-button" type="button" onClick={() => openChartSettings({ key: chartKey, title, categories, kind: 'line', includeTotal: true, defaultHiddenSeries })} aria-label={`Chart settings for ${title}`}>
          <Settings size={17} />
        </button>
      </div>
      <LineChart series={series} labels={historyRows.map((row) => row.date)} palette={palettes[settings.palette]} colorDomain={[...colorDomain, 'Total']} mode={settings.mode} />
    </section>
  );
}

function PieChartPanel({
  title,
  chartKey,
  rows,
  categories,
  colorDomain,
  defaultHiddenSeries = [],
  getSettings,
  openChartSettings,
}: {
  title: string;
  chartKey: string;
  rows: DashboardDetailRow[];
  categories: string[];
  colorDomain: string[];
  defaultHiddenSeries?: string[];
  getSettings: (key: string, categories: string[], includeTotal: boolean, defaultHiddenSeries?: string[]) => ChartSettings;
  openChartSettings: (definition: ChartDefinition) => void;
}) {
  const settings = getSettings(chartKey, categories, false, defaultHiddenSeries);
  const visibleCategories = categories.filter((category) => !settings.hiddenSeries.includes(category));
  const slices = visibleCategories
    .map((category) => ({ label: category, value: categoryValue(rows, category) }))
    .filter((item) => item.value > 0);
  return (
    <section className="panel chart-panel">
      <div className="chart-header">
        <h2>{title}</h2>
        <button className="secondary icon-button" type="button" onClick={() => openChartSettings({ key: chartKey, title, categories, kind: 'pie', includeTotal: false, defaultHiddenSeries })} aria-label={`Chart settings for ${title}`}>
          <Settings size={17} />
        </button>
      </div>
      <PieChart slices={slices} palette={palettes[settings.palette]} colorDomain={colorDomain} />
    </section>
  );
}

function BarChartPanel<T extends Record<string, unknown>>({
  title,
  chartKey,
  rows,
  labelKey,
  getSettings,
  openChartSettings,
}: {
  title: string;
  chartKey: string;
  rows: T[];
  labelKey: keyof T;
  getSettings: (key: string, categories: string[], includeTotal: boolean, defaultHiddenSeries?: string[]) => ChartSettings;
  openChartSettings: (definition: ChartDefinition) => void;
}) {
  const labels = rows.map((row) => String(row[labelKey] || ''));
  const settings = getSettings(chartKey, labels, false);
  const palette = palettes[settings.palette];
  const visibleRows = rows.filter((row) => !settings.hiddenSeries.includes(String(row[labelKey] || '')));
  return (
    <section className="panel chart-panel">
      <div className="chart-header">
        <h2>{title}</h2>
        <button className="secondary icon-button" type="button" onClick={() => openChartSettings({ key: chartKey, title, categories: labels, kind: 'bar', includeTotal: false })} aria-label={`Chart settings for ${title}`}>
          <Settings size={17} />
        </button>
      </div>
      <BarChart rows={visibleRows.map((row) => ({ label: String(row[labelKey] || ''), value: Number(row.value || 0) }))} palette={palette} colorDomain={labels} />
    </section>
  );
}

function buildLineSeries(
  historyRows: Array<{ details: DashboardDetailRow[] }>,
  categories: string[],
  settings: ChartSettings,
  includeTotal: boolean,
) {
  const series = categories.map((category) => {
    const rawValues = historyRows.map((item) => categoryValue(item.details, category));
    return {
      label: category,
      axis: settings.seriesAxes[category] || 'primary',
      rawValues,
      values: transformSeries(rawValues, settings.mode),
    };
  });
  if (includeTotal) {
    const rawValues = historyRows.map((item) => categories
      .reduce((sum, category) => sum + categoryValue(item.details, category), 0));
    series.push({
      label: 'Total',
      axis: settings.seriesAxes[TOTAL_SERIES_KEY] || 'secondary',
      rawValues,
      values: transformSeries(rawValues, settings.mode),
    });
  }
  return series;
}

function axisScale(series: Array<{ values: number[] }>) {
  if (!series.length) return { min: 0, max: 1, range: 1 };
  const values = series.flatMap((item) => item.values.map((value) => Number(value || 0)));
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 1);
  return { min, max, range: max - min || 1 };
}

function LineChart({
  series,
  labels,
  palette,
  colorDomain,
  mode,
}: {
  series: Array<{ label: string; axis: string; values: number[]; rawValues: number[] }>;
  labels: string[];
  palette: string[];
  colorDomain: string[];
  mode: ChartMode;
}) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const width = 760;
  const height = 250;
  const padding = { top: 24, right: 96, bottom: 42, left: 92 };
  const activeSeries = series.filter((item) => item.rawValues.some((value) => value > 0));
  const primary = activeSeries.filter((item) => item.axis !== 'secondary');
  const secondary = activeSeries.filter((item) => item.axis === 'secondary');
  const primaryScale = axisScale(primary);
  const secondaryScale = axisScale(secondary);
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const point = (value: number, index: number, scale: ReturnType<typeof axisScale>) => {
    const x = padding.left + plotWidth * (labels.length <= 1 ? 0 : index / (labels.length - 1));
    const y = padding.top + plotHeight - ((value - scale.min) / scale.range) * plotHeight;
    return [x, y];
  };
  const hoverX = hoverIndex === null ? null : padding.left + plotWidth * (labels.length <= 1 ? 0 : hoverIndex / (labels.length - 1));
  const handleMove = (event: React.MouseEvent<SVGSVGElement>) => {
    if (!labels.length) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const localX = ((event.clientX - rect.left) / rect.width) * width;
    const ratio = (localX - padding.left) / plotWidth;
    const index = Math.max(0, Math.min(labels.length - 1, Math.round(ratio * (labels.length - 1))));
    setHoverIndex(index);
  };

  if (!activeSeries.length) return <div className="empty-chart">No category data</div>;

  return (
    <div className="line-chart-frame">
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Historical line chart" onMouseMove={handleMove} onMouseLeave={() => setHoverIndex(null)}>
      {[0, 1, 2, 3, 4].map((tick) => {
        const ratio = tick / 4;
        const y = padding.top + plotHeight * ratio;
        const leftValue = primaryScale.max - primaryScale.range * ratio;
        const rightValue = secondaryScale.max - secondaryScale.range * ratio;
        return (
          <g key={tick}>
            <line x1={padding.left} x2={width - padding.right} y1={y} y2={y} className="grid-line" />
            <text x={padding.left - 10} y={y + 4} textAnchor="end" className="axis-label">{formatChartValue(leftValue, mode)}</text>
            {secondary.length ? <text x={width - padding.right + 10} y={y + 4} className="axis-label">{formatChartValue(rightValue, mode)}</text> : null}
          </g>
        );
      })}
      <line x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} className="axis-line" />
      <line x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} className="axis-line" />
      {secondary.length ? <line x1={width - padding.right} x2={width - padding.right} y1={padding.top} y2={height - padding.bottom} className="axis-line" /> : null}
      <text x={padding.left} y={14} className="axis-title">Left {mode === 'currency' ? 'EUR' : mode === 'percent' ? '% vs start' : 'Index 100'}</text>
      {secondary.length ? <text x={width - padding.right} y={14} textAnchor="end" className="axis-title">Right {mode === 'currency' ? 'EUR' : mode === 'percent' ? '% vs start' : 'Index 100'}</text> : null}
      {labels.length ? <text x={padding.left} y={height - 12} className="axis-label">{labels[0]}</text> : null}
      {labels.length > 1 ? <text x={width - padding.right} y={height - 12} textAnchor="end" className="axis-label">{labels[labels.length - 1]}</text> : null}
        {activeSeries.map((item) => {
        const scale = item.axis === 'secondary' ? secondaryScale : primaryScale;
        const path = item.values.map((value, index) => {
          const [x, y] = point(value, index, scale);
          return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
        }).join(' ');
        return (
          <g key={item.label}>
            <path d={path} fill="none" stroke={colorForLabel(item.label, palette, colorDomain)} strokeWidth={item.label === 'Total' ? 3 : 2} />
            {hoverIndex !== null ? (() => {
              const [x, y] = point(item.values[hoverIndex] || 0, hoverIndex, scale);
              return <circle cx={x} cy={y} r="3.5" fill={colorForLabel(item.label, palette, colorDomain)} stroke="var(--bg-soft)" strokeWidth="2" />;
            })() : null}
          </g>
        );
      })}
      {hoverX !== null ? <line x1={hoverX} x2={hoverX} y1={padding.top} y2={height - padding.bottom} className="hover-line" /> : null}
      <rect x={padding.left} y={padding.top} width={plotWidth} height={plotHeight} fill="transparent" />
      </svg>
      {hoverIndex !== null ? (
        <div className="chart-tooltip" style={{ left: `${Math.min(86, Math.max(14, ((hoverX || padding.left) / width) * 100))}%` }}>
          <strong>{labels[hoverIndex]}</strong>
          {activeSeries.map((item) => (
            <span key={item.label}>
              <i style={{ background: colorForLabel(item.label, palette, colorDomain) }} />
              {item.label}: {formatCurrency(item.rawValues[hoverIndex] || 0)}
            </span>
          ))}
        </div>
      ) : null}
      <div className="line-legend">
        {activeSeries.map((item) => (
          <div className="legend-item" key={item.label}>
            <span className="legend-swatch" style={{ background: colorForLabel(item.label, palette, colorDomain) }} />
            <span>{item.label}{item.axis === 'secondary' ? ' · right axis' : ''}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function PieChart({ slices, palette, colorDomain }: { slices: Array<{ label: string; value: number }>; palette: string[]; colorDomain: string[] }) {
  const total = slices.reduce((sum, row) => sum + row.value, 0);
  let cumulative = 0;
  if (!total) return <div className="empty-chart">No latest NAV</div>;

  return (
    <div className="pie-chart-layout">
      <svg className="chart-svg" viewBox="0 0 280 240" role="img" aria-label="Latest allocation pie chart">
        {slices.map((slice, index) => {
          const start = cumulative / total;
          cumulative += slice.value;
          const end = cumulative / total;
          const path = describePieSlice(140, 120, 86, start, end);
          const mid = (start + end) / 2;
          const angle = mid * Math.PI * 2 - Math.PI / 2;
          const labelX = 140 + Math.cos(angle) * 52;
          const labelY = 120 + Math.sin(angle) * 52;
          const outerX = 140 + Math.cos(angle) * 112;
          const outerY = 120 + Math.sin(angle) * 100 + (index % 2 ? 5 : -5);
          const lineStartX = 140 + Math.cos(angle) * 88;
          const lineStartY = 120 + Math.sin(angle) * 88;
          const lineEndX = 140 + Math.cos(angle) * 101;
          const lineEndY = 120 + Math.sin(angle) * 101;
          const share = slice.value / total * 100;
          const color = colorForLabel(slice.label, palette, colorDomain);
          return (
            <g key={slice.label}>
              <path d={path} fill={color} />
              {share > 6 ? (
                <text x={labelX} y={labelY} textAnchor="middle" className="pie-value">
                  {formatPercent(share)}
                </text>
              ) : (
                <>
                  <path d={`M ${lineStartX} ${lineStartY} L ${lineEndX} ${lineEndY} L ${outerX} ${outerY}`} className="pie-callout" />
                  <text x={outerX} y={outerY + 3} textAnchor={outerX >= 140 ? 'start' : 'end'} className="pie-callout-label">
                    {formatPercent(share)}
                  </text>
                </>
              )}
            </g>
          );
        })}
      </svg>
      <div className="pie-legend">
        {slices.map((slice) => (
          <div className="legend-item" key={slice.label}>
            <span className="legend-swatch" style={{ background: colorForLabel(slice.label, palette, colorDomain) }} />
            <span>{slice.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function describePieSlice(cx: number, cy: number, r: number, start: number, end: number) {
  const startAngle = start * Math.PI * 2 - Math.PI / 2;
  const endAngle = end * Math.PI * 2 - Math.PI / 2;
  const x1 = cx + Math.cos(startAngle) * r;
  const y1 = cy + Math.sin(startAngle) * r;
  const x2 = cx + Math.cos(endAngle) * r;
  const y2 = cy + Math.sin(endAngle) * r;
  const largeArc = end - start > 0.5 ? 1 : 0;
  return `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2} Z`;
}

function BarChart({ rows, palette, colorDomain }: { rows: Array<{ label: string; value: number }>; palette: string[]; colorDomain: string[] }) {
  const topRows = rows.slice(0, 8);
  const max = Math.max(...topRows.map((row) => row.value), 1);
  if (!topRows.length) return <div className="empty-chart">No breakdown data</div>;
  return (
    <svg className="chart-svg" viewBox="0 0 760 250" role="img" aria-label="Breakdown bar chart">
      {topRows.map((row, index) => {
        const y = 24 + index * 27;
        const width = 520 * (row.value / max);
        return (
          <g key={row.label}>
            <text x="10" y={y + 15} className="axis-label">{row.label}</text>
            <rect x="150" y={y} width={width} height="18" fill={colorForLabel(row.label, palette, colorDomain)} rx="3" />
            <text x={160 + width} y={y + 14} className="axis-label">{formatCurrency(row.value)}</text>
          </g>
        );
      })}
    </svg>
  );
}

function ChartSettingsModal({
  definition,
  settings,
  patchSettings,
  onClose,
}: {
  definition: ChartDefinition;
  settings: ChartSettings;
  patchSettings: (key: string, patch: Partial<ChartSettings>) => void;
  onClose: () => void;
}) {
  const keys = seriesKeys(definition.categories, definition.includeTotal);
  const isLine = definition.kind === 'line';
  const updateHidden = (key: string, checked: boolean) => {
    const hidden = checked
      ? settings.hiddenSeries.filter((item) => item !== key)
      : [...new Set([...settings.hiddenSeries, key])];
    patchSettings(definition.key, { hiddenSeries: hidden });
  };

  return (
    <Modal title="Chart settings" subtitle={definition.title} onClose={onClose}>
      <div className="toolbar">
        <label>Palette
          <select value={settings.palette} onChange={(event) => patchSettings(definition.key, { palette: event.target.value as ChartPalette })}>
            <option value="calm">Calm</option>
            <option value="vivid">Vivid</option>
            <option value="mono">Mono</option>
          </select>
        </label>
        {isLine ? (
          <label>Units
            <select value={settings.mode} onChange={(event) => patchSettings(definition.key, { mode: event.target.value as ChartMode })}>
              <option value="currency">EUR</option>
              <option value="percent">% vs start</option>
              <option value="index">Index 100</option>
            </select>
          </label>
        ) : null}
      </div>
      <div className="toolbar">
        <button className="secondary" type="button" onClick={() => patchSettings(definition.key, { hiddenSeries: [] })}>Show all</button>
        <button className="secondary" type="button" onClick={() => patchSettings(definition.key, { hiddenSeries: keys })}>Hide all</button>
      </div>
      <div className="chart-settings-list">
        {keys.map((key) => (
          <div className="chart-setting-row" key={key}>
            <label>
              <input type="checkbox" checked={!settings.hiddenSeries.includes(key)} onChange={(event) => updateHidden(key, event.target.checked)} />
              {seriesLabel(key)}
            </label>
            {isLine ? (
              <select
                value={settings.seriesAxes[key] || 'primary'}
                onChange={(event) => patchSettings(definition.key, { seriesAxes: { [key]: event.target.value as 'primary' | 'secondary' } })}
              >
                <option value="primary">Left axis</option>
                <option value="secondary">Right axis</option>
              </select>
            ) : <span className="muted">Visible</span>}
          </div>
        ))}
      </div>
    </Modal>
  );
}

function PositionsTable({ rows }: { rows: DashboardDetailRow[] }) {
  return (
    <SortableTable
      id="positionsTable"
      rows={rows}
      columns={[
        { key: 'asset', label: 'Asset', value: (row) => row.asset_name, render: (row) => row.asset_name },
        { key: 'owner', label: 'Owner', value: (row) => row.owner_name, render: (row) => row.owner_name },
        { key: 'category', label: 'Category', value: (row) => row.category || '', render: (row) => row.category || '' },
        { key: 'quantity', label: 'Quantity', value: (row) => isStockRow(row) ? row.quantity : 0, render: (row) => stockQuantityLabel(row) },
        { key: 'value', label: 'Value', value: (row) => row.value, render: (row) => formatCurrency(row.value) },
        { key: 'broker', label: 'Broker', value: (row) => row.broker || '', render: (row) => row.broker || '' },
      ]}
    />
  );
}

function HistoryTables({ historyRows, categories }: { historyRows: Array<{ date: string; details: DashboardDetailRow[]; total: number }>; categories: string[] }) {
  const start = historyRows.find((item) => item.total !== 0) || historyRows[0];
  return (
    <>
      <section className="panel">
        <h2>History absolute</h2>
        <SortableTable
          id="historyAbsolute"
          rows={historyRows}
          columns={[
            { key: 'date', label: 'Date', value: (row) => row.date, render: (row) => row.date },
            { key: 'total', label: 'Total', value: (row) => row.total, render: (row) => formatCurrency(row.total) },
            ...categories.map((category) => ({
              key: category,
              label: category,
              value: (row: typeof historyRows[number]) => categoryValue(row.details, category),
              render: (row: typeof historyRows[number]) => formatCurrency(categoryValue(row.details, category)),
            })),
          ]}
        />
      </section>
      <section className="panel">
        <h2>History vs start</h2>
        <SortableTable
          id="historyPercent"
          rows={historyRows}
          columns={[
            { key: 'date', label: 'Date', value: (row) => row.date, render: (row) => row.date },
            { key: 'total', label: 'Total', value: (row) => start?.total ? (row.total / start.total - 1) * 100 : 0, render: (row) => formatPercent(start?.total ? (row.total / start.total - 1) * 100 : 0) },
            ...categories.map((category) => {
              const base = start ? categoryValue(start.details, category) : 0;
              return {
                key: category,
                label: category,
                value: (row: typeof historyRows[number]) => base ? (categoryValue(row.details, category) / base - 1) * 100 : 0,
                render: (row: typeof historyRows[number]) => formatPercent(base ? (categoryValue(row.details, category) / base - 1) * 100 : 0),
              };
            }),
          ]}
        />
      </section>
    </>
  );
}

function groupedAggregateRows(rows: DashboardDetailRow[]) {
  const total = rows.reduce((sum, row) => sum + rowValue(row), 0);
  const map = new Map<string, { asset: string; category: string; broker: string; asset_type?: string | null; quantity: number; value: number }>();
  rows.forEach((row) => {
    const key = `${row.asset_name}|${row.category || ''}|${row.broker || ''}`;
    if (!map.has(key)) {
      map.set(key, { asset: row.asset_name, category: row.category || '', broker: row.broker || '', asset_type: row.asset_type, quantity: 0, value: 0 });
    }
    const item = map.get(key);
    if (!item) return;
    if (isStockRow(row)) item.quantity += rowQuantity(row);
    item.value += rowValue(row);
  });
  return [...map.values()].sort((a, b) => b.value - a.value).map((item) => ({
    ...item,
    valuePct: total ? item.value / total * 100 : 0,
  }));
}

function previousMetric(historyRows: Array<{ date: string; total: number; investing: number }>, viewDate: string) {
  const index = historyRows.findIndex((row) => row.date === viewDate);
  if (index <= 0) return null;
  return historyRows[index - 1];
}

function deltaText(current: number, previous?: number) {
  if (previous === undefined || previous === null) return 'No previous date';
  const delta = current - previous;
  const pct = previous ? delta / previous * 100 : 0;
  return `${delta >= 0 ? '+' : ''}${formatCurrency(delta)} (${delta >= 0 ? '+' : ''}${formatPercent(pct)})`;
}

function largestMovers(historyRows: Array<{ date: string; details: DashboardDetailRow[] }>, viewDate: string) {
  const index = historyRows.findIndex((row) => row.date === viewDate);
  if (index <= 0) return [];
  const current = historyRows[index];
  const previous = historyRows[index - 1];
  const categories = [...new Set([...categoryList([current]), ...categoryList([previous])])];
  return categories
    .map((category) => ({
      category,
      value: categoryValue(current.details, category) - categoryValue(previous.details, category),
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 5)
    .sort((a, b) => b.value - a.value);
}

function DashboardInsights({ movers, byOwner = [] }: { movers: Array<{ category: string; value: number }>; byOwner?: Array<{ owner_name: string; value: number }> }) {
  return (
    <div className="split compact-split">
      <section className="panel">
        <h2>Largest movers</h2>
        {movers.length ? movers.map((row) => (
          <div className="insight-row" key={row.category}>
            <span>{row.category}</span>
            <strong>{row.value >= 0 ? '+' : ''}{formatCurrency(row.value)}</strong>
          </div>
        )) : <p className="muted">No previous date to compare.</p>}
      </section>
      {byOwner.length ? (
        <section className="panel">
          <h2>By owner</h2>
          {byOwner.map((row) => (
            <div className="insight-row" key={row.owner_name}>
              <span>{row.owner_name}</span>
              <strong>{formatCurrency(row.value)}</strong>
            </div>
          ))}
        </section>
      ) : null}
    </div>
  );
}

export { App };
