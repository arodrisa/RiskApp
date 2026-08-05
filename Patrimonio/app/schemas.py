from pydantic import BaseModel, Field, validator
from typing import Any, Dict, List, Optional
from datetime import datetime


class OwnerBase(BaseModel):
    name: str
    type: Optional[str] = 'person'
    is_family_member: bool = False


class OwnerCreate(OwnerBase):
    pass


class OwnerUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    is_family_member: Optional[bool] = None


class Owner(OwnerBase):
    id: int

    class Config:
        orm_mode = True


class AssetValuationBase(BaseModel):
    as_of_date: datetime
    value: float
    source: Optional[str] = 'import'


class AssetValuation(AssetValuationBase):
    id: int
    asset_id: int

    class Config:
        orm_mode = True


class AssetBase(BaseModel):
    name: str
    category: Optional[str] = None
    asset_type: Optional[str] = None
    valuation_method: Optional[str] = 'market_direct'
    price_provider: Optional[str] = 'manual'
    price_symbol: Optional[str] = None
    is_investment: Optional[bool] = None
    is_shared: Optional[bool] = False


class AssetCreate(AssetBase):
    pass


class AssetUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    asset_type: Optional[str] = None
    valuation_method: Optional[str] = None
    price_provider: Optional[str] = None
    price_symbol: Optional[str] = None
    is_investment: Optional[bool] = None
    is_shared: Optional[bool] = None


class Asset(AssetBase):
    id: int
    valuations: List[AssetValuation] = []
    created_at: datetime

    class Config:
        orm_mode = True


class DuplicateAssetItem(BaseModel):
    id: int
    name: str
    category: Optional[str] = None
    asset_type: Optional[str] = None


class DuplicateAssetGroup(BaseModel):
    name: str
    count: int
    assets: List[DuplicateAssetItem]


class PriceQuote(BaseModel):
    provider: str
    symbol: str
    price: float
    currency: Optional[str] = None
    as_of: Optional[str] = None


class PriceHistory(BaseModel):
    id: int
    asset_id: int
    provider: Optional[str] = None
    symbol: Optional[str] = None
    price: float
    currency: Optional[str] = None
    as_of: Optional[str] = None
    created_at: datetime

    class Config:
        orm_mode = True


class LoginRequest(BaseModel):
    username: str
    password: str


class BootstrapRequest(BaseModel):
    email: str
    display_name: str
    password: str
    person_owner_id: Optional[int] = None
    setup_token: Optional[str] = None


class UserSummary(BaseModel):
    id: int
    email: str
    display_name: str
    person_owner_id: Optional[int] = None
    is_active: bool
    role: Optional[str] = None

    class Config:
        orm_mode = True


class ProjectUserUpdate(BaseModel):
    role: Optional[str] = None
    is_active: Optional[bool] = None

    @validator('role')
    def validate_updated_project_role(cls, value):
        if value is not None and value not in {'admin', 'editor', 'viewer'}:
            raise ValueError('Role must be admin, editor, or viewer')
        return value


class ProjectInvitationCreate(BaseModel):
    email: str
    role: str = 'viewer'

    @validator('role')
    def validate_project_role(cls, value):
        if value not in {'admin', 'editor', 'viewer'}:
            raise ValueError('Role must be admin, editor, or viewer')
        return value


class ProjectInvitationResult(BaseModel):
    id: int
    email: str
    role: str
    expires_at: datetime
    invite_url: Optional[str] = None


class InvitationAcceptRequest(BaseModel):
    token: str
    display_name: str
    password: str
    person_owner_id: Optional[int] = None


class PasswordResetRequest(BaseModel):
    email: str


class PasswordResetRequestResult(BaseModel):
    message: str
    dev_reset_url: Optional[str] = None


class PasswordResetConfirm(BaseModel):
    token: str
    password: str


class AuthStatus(BaseModel):
    enabled: bool
    authenticated: bool
    username: Optional[str] = None
    csrf_token: Optional[str] = None
    restore_enabled: bool = False
    needs_bootstrap: bool = False
    role: Optional[str] = None
    project_name: Optional[str] = None


class InvestingAsset(BaseModel):
    category: str
    is_invested: bool

    class Config:
        orm_mode = True


class InvestingAssetUpdate(BaseModel):
    is_invested: bool


class CompanyValuationItemBase(BaseModel):
    item_type: str
    name: str
    amount: float


class CompanyValuationItemCreate(CompanyValuationItemBase):
    pass


class CompanyValuationItem(CompanyValuationItemBase):
    id: int
    asset_id: int
    as_of_date: datetime

    class Config:
        orm_mode = True


class CompanyValuationUpdate(BaseModel):
    as_of_date: datetime
    items: List[CompanyValuationItemCreate] = Field(default_factory=list)

    @validator('as_of_date', pre=True)
    def parse_company_date_only(cls, value):
        if isinstance(value, str) and len(value) == 10:
            return f'{value}T00:00:00'
        return value


class CompanyValuationSnapshot(BaseModel):
    asset_id: int
    as_of_date: datetime
    items: List[CompanyValuationItem] = Field(default_factory=list)
    assets_total: float
    liabilities_total: float
    net_value: float


class EntityOwnershipBase(BaseModel):
    owner_id: int
    owned_id: int
    share: float = Field(..., gt=0, le=1)
    effective_from: datetime
    effective_to: Optional[datetime] = None

    @validator('effective_from', 'effective_to', pre=True)
    def parse_entity_ownership_dates(cls, value):
        if isinstance(value, str) and len(value) == 10:
            return f'{value}T00:00:00'
        return value


class EntityOwnershipCreate(EntityOwnershipBase):
    pass


class EntityOwnershipUpdate(BaseModel):
    owner_id: Optional[int] = None
    owned_id: Optional[int] = None
    share: Optional[float] = Field(default=None, gt=0, le=1)
    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None

    _parse_entity_ownership_dates = validator('effective_from', 'effective_to', pre=True, allow_reuse=True)(
        EntityOwnershipBase.parse_entity_ownership_dates,
    )


class EntityOwnership(EntityOwnershipBase):
    id: int
    owner_name: str
    owned_name: str

    class Config:
        orm_mode = True


class OwnershipShare(BaseModel):
    owner_id: int
    owner_name: Optional[str] = None
    share: float


class PositionOwnershipUpdate(BaseModel):
    shares: List[OwnershipShare] = Field(default_factory=list)


class PositionOwnershipState(BaseModel):
    position_id: int
    shares: List[OwnershipShare] = Field(default_factory=list)


class PositionBase(BaseModel):
    as_of_date: datetime
    quantity: Optional[float] = 0.0
    value: Optional[float] = 0.0
    owner_id: Optional[int] = None
    broker: Optional[str] = None
    source: Optional[str] = 'import'

    @validator('as_of_date', pre=True)
    def parse_date_only_snapshot(cls, value):
        if isinstance(value, str) and len(value) == 10:
            return f'{value}T00:00:00'
        return value


class PositionCreate(PositionBase):
    asset_id: int


class BulkPositionSaveItem(PositionBase):
    asset_id: Optional[int] = None
    asset_name: Optional[str] = None
    position_id: Optional[int] = None
    category: Optional[str] = None
    ownership_shares: List[OwnershipShare] = Field(default_factory=list)


class BulkPositionSave(BaseModel):
    positions: List[BulkPositionSaveItem] = Field(default_factory=list)
    as_of_date: Optional[datetime] = None
    replace_snapshot: bool = False

    @validator('as_of_date', pre=True)
    def parse_bulk_date_only_snapshot(cls, value):
        if isinstance(value, str) and len(value) == 10:
            return f'{value}T00:00:00'
        return value


class PositionUpdate(BaseModel):
    as_of_date: Optional[datetime] = None
    quantity: Optional[float] = None
    value: Optional[float] = None
    owner_id: Optional[int] = None
    broker: Optional[str] = None
    source: Optional[str] = None


class Position(PositionBase):
    id: int
    asset_id: int

    class Config:
        orm_mode = True


class PositionSnapshotRow(BaseModel):
    position_id: int
    asset_id: int
    owner_id: Optional[int] = None
    asset_name: str
    owner_name: Optional[str] = None
    category: Optional[str] = None
    asset_type: Optional[str] = None
    valuation_method: Optional[str] = None
    is_investment: Optional[bool] = None
    broker: Optional[str] = None
    quantity: float
    value: float
    ownership_shares: List[OwnershipShare] = Field(default_factory=list)


class AssetBreakdownBase(BaseModel):
    asset_name: str
    category: Optional[str] = None
    value: float


class CategoryBreakdownBase(BaseModel):
    category: str
    value: float


class BrokerBreakdownBase(BaseModel):
    broker: str
    value: float


class OwnerBreakdownBase(BaseModel):
    owner_name: str
    value: float


class DashboardDetailRow(BaseModel):
    position_id: int
    asset_id: int
    owner_id: Optional[int] = None
    asset_name: str
    owner_name: str
    category: Optional[str] = None
    asset_type: Optional[str] = None
    valuation_method: Optional[str] = None
    is_investment: Optional[bool] = None
    broker: Optional[str] = None
    quantity: float
    value: float


class DashboardSummary(BaseModel):
    as_of_date: Optional[datetime] = None
    total_value: float
    position_count: int
    by_asset: List[AssetBreakdownBase] = []
    by_category: List[CategoryBreakdownBase] = []
    by_broker: List[BrokerBreakdownBase] = []
    by_owner: List[OwnerBreakdownBase] = []


class DashboardHistoryPoint(BaseModel):
    date: str
    summary: DashboardSummary
    details: List[DashboardDetailRow] = []


class RestoreBackupRequest(BaseModel):
    confirm_restore: bool = False
    backup: Dict[str, Any]


class RestoreBackupResult(BaseModel):
    owners: int
    assets: int
    positions: int
    investing_assets: int


class AuditLog(BaseModel):
    id: int
    created_at: datetime
    actor: Optional[str] = None
    action: str
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    details: Optional[str] = None

    class Config:
        orm_mode = True
