export type Owner = {
  id: number;
  name: string;
  type?: string | null;
  is_family_member?: boolean;
};

export type EntityOwnership = {
  id: number;
  owner_id: number;
  owner_name: string;
  owned_id: number;
  owned_name: string;
  share: number;
  effective_from: string;
  effective_to?: string | null;
};

export type Asset = {
  id: number;
  name: string;
  category?: string | null;
  asset_type?: string | null;
  valuation_method?: string | null;
  price_provider?: string | null;
  price_symbol?: string | null;
  is_investment?: boolean | null;
  is_shared?: boolean | null;
  created_at?: string;
};

export type OwnershipShare = {
  owner_id: number;
  owner_name?: string | null;
  share: number;
};

export type SnapshotRow = {
  position_id?: number | null;
  asset_id: number | '';
  owner_id?: number | '' | null;
  asset_name?: string;
  owner_name?: string | null;
  category?: string | null;
  asset_type?: string | null;
  valuation_method?: string | null;
  is_investment?: boolean | null;
  broker?: string | null;
  quantity: number;
  value: number;
  ownership_shares: OwnershipShare[];
};

export type DashboardDetailRow = {
  position_id: number;
  asset_id: number;
  owner_id?: number | null;
  asset_name: string;
  owner_name: string;
  category?: string | null;
  asset_type?: string | null;
  valuation_method?: string | null;
  is_investment?: boolean | null;
  broker?: string | null;
  quantity: number;
  value: number;
};

export type Breakdown = {
  value: number;
};

export type AssetBreakdown = Breakdown & {
  asset_name: string;
  category?: string | null;
};

export type CategoryBreakdown = Breakdown & {
  category: string;
};

export type BrokerBreakdown = Breakdown & {
  broker: string;
};

export type OwnerBreakdown = Breakdown & {
  owner_name: string;
};

export type DashboardSummary = {
  as_of_date?: string | null;
  total_value: number;
  position_count: number;
  by_asset: AssetBreakdown[];
  by_category: CategoryBreakdown[];
  by_broker: BrokerBreakdown[];
  by_owner: OwnerBreakdown[];
};

export type DashboardHistoryPoint = {
  date: string;
  summary: DashboardSummary;
  details: DashboardDetailRow[];
};

export type PriceQuote = {
  provider: string;
  symbol: string;
  price: number;
  currency?: string | null;
  as_of?: string | null;
};

export type PriceHistory = PriceQuote & {
  id: number;
  asset_id: number;
  created_at: string;
};

export type InvestingAsset = {
  category: string;
  is_invested: boolean;
};

export type PriceAuditRow = {
  rowIndex: number;
  assetName: string;
  symbol: string;
  provider: string;
  quantity: number;
  oldValue: number;
  unitPrice: number;
  newValue: number;
  currency?: string | null;
  asOf?: string | null;
};

export type CompanyValuationItem = {
  id?: number;
  asset_id?: number;
  as_of_date?: string;
  item_type: 'asset' | 'liability';
  name: string;
  amount: number;
};

export type CompanyValuationSnapshot = {
  asset_id: number;
  as_of_date: string;
  items: CompanyValuationItem[];
  assets_total: number;
  liabilities_total: number;
  net_value: number;
};

export type AuthStatus = {
  enabled: boolean;
  authenticated: boolean;
  username?: string | null;
  csrf_token?: string | null;
  restore_enabled: boolean;
  needs_bootstrap?: boolean;
  role?: string | null;
  project_name?: string | null;
};

export type ProjectUser = {
  id: number;
  email: string;
  display_name: string;
  person_owner_id?: number | null;
  is_active: boolean;
  role?: string | null;
};

export type ProjectInvitationResult = {
  id: number;
  email: string;
  role: string;
  expires_at: string;
  invite_url?: string | null;
};

export type AuditLog = {
  id: number;
  created_at: string;
  actor?: string | null;
  action: string;
  entity_type?: string | null;
  entity_id?: string | null;
  details?: string | null;
};

export type RestoreResult = {
  owners: number;
  assets: number;
  positions: number;
  investing_assets: number;
};

export type SortDirection = 'asc' | 'desc';

export type SortState = {
  key: string;
  direction: SortDirection;
};

export type ChartMode = 'currency' | 'percent' | 'index';
export type ChartAxis = 'primary' | 'secondary';
export type ChartPalette = 'calm' | 'vivid' | 'mono';

export type ChartSettings = {
  palette: ChartPalette;
  mode: ChartMode;
  hiddenSeries: string[];
  seriesAxes: Record<string, ChartAxis>;
};
