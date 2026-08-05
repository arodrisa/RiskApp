import {
  Asset,
  AuditLog,
  AuthStatus,
  EntityOwnership,
  CompanyValuationItem,
  CompanyValuationSnapshot,
  DashboardDetailRow,
  DashboardHistoryPoint,
  DashboardSummary,
  InvestingAsset,
  Owner,
  PriceHistory,
  PriceQuote,
  ProjectInvitationResult,
  ProjectUser,
  RestoreResult,
  SnapshotRow,
} from './types';

let csrfToken: string | null = null;

function rememberAuth(status: AuthStatus) {
  csrfToken = status.csrf_token || null;
  return status;
}

async function request<T>(url: string, options: RequestInit = {}): Promise<T> {
  const method = String(options.method || 'GET').toUpperCase();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> || {}),
  };
  if (csrfToken && !['GET', 'HEAD', 'OPTIONS'].includes(method)) {
    headers['X-CSRF-Token'] = csrfToken;
  }

  const response = await fetch(url, {
    credentials: 'same-origin',
    headers,
    ...options,
  });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      // Keep the HTTP status message.
    }
    throw new Error(message);
  }

  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  authStatus: async () => rememberAuth(await request<AuthStatus>('/auth/status')),
  login: async (username: string, password: string) => rememberAuth(await request<AuthStatus>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  })),
  logout: async () => rememberAuth(await request<AuthStatus>('/auth/logout', { method: 'POST' })),
  bootstrapOptions: () => request<Owner[]>('/auth/bootstrap-options'),
  bootstrap: async (payload: { email: string; display_name: string; password: string; person_owner_id?: number; setup_token?: string }) => rememberAuth(await request<AuthStatus>('/auth/bootstrap', {
    method: 'POST',
    body: JSON.stringify(payload),
  })),
  requestPasswordReset: (email: string) => request<{ message: string; dev_reset_url?: string | null }>('/auth/password-reset/request', {
    method: 'POST',
    body: JSON.stringify({ email }),
  }),
  confirmPasswordReset: (token: string, password: string) => request<{ message: string }>('/auth/password-reset/confirm', {
    method: 'POST',
    body: JSON.stringify({ token, password }),
  }),
  auditLog: () => request<AuditLog[]>('/audit-log/'),
  restore: (backup: unknown) => request<RestoreResult>('/restore', {
    method: 'POST',
    body: JSON.stringify({ confirm_restore: true, backup }),
  }),
  owners: () => request<Owner[]>('/owners/'),
  createOwner: (payload: Partial<Owner>) => request<Owner>('/owners/', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  updateOwner: (id: number, payload: Partial<Owner>) => request<Owner>(`/owners/${id}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  }),
  deleteOwner: (id: number) => request<void>(`/owners/${id}`, { method: 'DELETE' }),
  entityOwnerships: () => request<EntityOwnership[]>('/entity-ownerships/'),
  createEntityOwnership: (payload: Omit<EntityOwnership, 'id' | 'owner_name' | 'owned_name'>) => request<EntityOwnership>('/entity-ownerships/', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  deleteEntityOwnership: (id: number) => request<void>(`/entity-ownerships/${id}`, { method: 'DELETE' }),
  projectUsers: () => request<ProjectUser[]>('/project-users/'),
  updateProjectUser: (id: number, payload: { role?: string; is_active?: boolean }) => request<ProjectUser>(`/project-users/${id}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  }),
  inviteProjectUser: (payload: { email: string; role: string }) => request<ProjectInvitationResult>('/project-invitations/', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),

  assets: () => request<Asset[]>('/assets/'),
  createAsset: (payload: Partial<Asset>) => request<Asset>('/assets/', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  updateAsset: (id: number, payload: Partial<Asset>) => request<Asset>(`/assets/${id}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  }),
  deleteAsset: (id: number) => request<void>(`/assets/${id}`, { method: 'DELETE' }),
  investingAssets: () => request<InvestingAsset[]>('/investing-assets/'),
  updateInvestingAsset: (category: string, isInvested: boolean) => request<InvestingAsset>(`/investing-assets/${encodeURIComponent(category)}`, {
    method: 'PUT',
    body: JSON.stringify({ is_invested: isInvested }),
  }),
  companyValuation: (assetId: number, asOfDate: string) => request<CompanyValuationSnapshot>(`/assets/${assetId}/company-valuation?as_of_date=${encodeURIComponent(asOfDate)}`),
  saveCompanyValuation: (assetId: number, asOfDate: string, items: CompanyValuationItem[]) => request<CompanyValuationSnapshot>(`/assets/${assetId}/company-valuation`, {
    method: 'PUT',
    body: JSON.stringify({ as_of_date: asOfDate, items }),
  }),

  dates: () => request<string[]>('/dashboard/dates'),
  summary: (date?: string) => request<DashboardSummary>(`/dashboard/summary${date ? `?as_of_date=${encodeURIComponent(date)}` : ''}`),
  details: (date?: string) => request<DashboardDetailRow[]>(`/dashboard/details${date ? `?as_of_date=${encodeURIComponent(date)}` : ''}`),
  history: () => request<DashboardHistoryPoint[]>('/dashboard/history'),
  snapshot: (date?: string) => request<SnapshotRow[]>(`/positions/snapshot${date ? `?as_of_date=${encodeURIComponent(date)}` : ''}`),
  saveSnapshot: (asOfDate: string, positions: SnapshotRow[], replaceSnapshot = true) => request<SnapshotRow[]>('/positions/bulk', {
    method: 'POST',
    body: JSON.stringify({
      as_of_date: asOfDate,
      replace_snapshot: replaceSnapshot,
      positions: positions.map((row) => ({
        position_id: row.position_id || undefined,
        asset_id: row.asset_id || undefined,
        owner_id: row.owner_id === '' ? null : row.owner_id,
        as_of_date: asOfDate,
        quantity: Number(row.quantity || 0),
        value: Number(row.value || 0),
        broker: row.broker || null,
        source: 'manual',
        ownership_shares: row.owner_id ? [] : row.ownership_shares || [],
      })),
    }),
  }),
  quote: (assetId: number) => request<PriceQuote>(`/prices/quote?asset_id=${assetId}`),
  priceHistory: () => request<PriceHistory[]>('/prices/history'),
};
