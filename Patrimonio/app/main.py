from datetime import datetime
import mimetypes
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / '.env')

from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from app import auth, models, schemas, crud, prices
from app.database import engine, get_db, ensure_beta_schema, ensure_application_schema

auth.validate_production_settings()
if not auth.is_production_environment():
    models.Base.metadata.create_all(bind=engine)
    ensure_beta_schema(engine)
    ensure_application_schema(engine)
with Session(bind=engine) as startup_db:
    if not auth.is_production_environment():
        crud.migrate_legacy_asset_ownership_to_positions(startup_db)
    crud.initialize_project_data(startup_db)

app = FastAPI(title='Patrimonio')

STATIC_DIR = Path(__file__).resolve().parent / 'static'
FRONTEND_DIST_DIR = PROJECT_DIR / 'frontend' / 'dist'

mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')

if FRONTEND_DIST_DIR.exists():
    app.mount('/ui', StaticFiles(directory=FRONTEND_DIST_DIR), name='react-ui')


@app.middleware('http')
async def csrf_middleware(request: Request, call_next):
    if request.method in {'POST', 'PUT', 'PATCH', 'DELETE'} and request.url.path not in {
        '/auth/login',
        '/auth/bootstrap',
        '/auth/accept-invitation',
        '/auth/password-reset/request',
        '/auth/password-reset/confirm',
    }:
        try:
            auth.require_csrf(request)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={'detail': exc.detail})
    return await call_next(request)


def dashboard_file_response():
    react_index = FRONTEND_DIST_DIR / 'index.html'
    if react_index.exists():
        return FileResponse(react_index)
    return FileResponse(STATIC_DIR / 'dashboard.html')

@app.get('/', include_in_schema=False)
def read_dashboard_page():
    return dashboard_file_response()

@app.get('/dashboard', include_in_schema=False)
def read_dashboard_page_alias():
    return dashboard_file_response()


@app.get('/health')
def read_health():
    return {'status': 'ok'}


@app.get('/auth/status', response_model=schemas.AuthStatus)
def read_auth_status(request: Request, db: Session = Depends(get_db)):
    return auth.status(request, db)


@app.post('/auth/login', response_model=schemas.AuthStatus)
def login(payload: schemas.LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    result = auth.login(request, response, payload.username, payload.password, db)
    return {'enabled': auth.auth_enabled(), **result}


@app.get('/auth/bootstrap-options', response_model=List[schemas.Owner])
def read_bootstrap_options(db: Session = Depends(get_db)):
    if crud.get_default_project(db) and db.query(models.User).count():
        raise HTTPException(status_code=404, detail='Bootstrap is no longer available')
    return [owner for owner in crud.list_owners(db) if owner.type == 'person']


@app.post('/auth/bootstrap', response_model=schemas.AuthStatus)
def bootstrap(payload: schemas.BootstrapRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    result = auth.bootstrap(request, response, payload, db)
    crud.record_audit(db, 'bootstrap_user', 'user', details={'email': payload.email}, actor=result['username'])
    return {'enabled': auth.auth_enabled(), 'needs_bootstrap': False, **result}


@app.post('/auth/accept-invitation', response_model=schemas.AuthStatus)
def accept_invitation(payload: schemas.InvitationAcceptRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    result = auth.accept_invitation(request, response, payload, db)
    crud.record_audit(db, 'accept_invitation', 'user', details={'email': result['username']}, actor=result['username'])
    return {'enabled': auth.auth_enabled(), 'needs_bootstrap': False, **result}


@app.post('/auth/password-reset/request', response_model=schemas.PasswordResetRequestResult)
def request_password_reset(payload: schemas.PasswordResetRequest, db: Session = Depends(get_db)):
    return auth.request_password_reset(payload, db)


@app.post('/auth/password-reset/confirm', response_model=schemas.PasswordResetRequestResult)
def confirm_password_reset(payload: schemas.PasswordResetConfirm, db: Session = Depends(get_db)):
    return auth.confirm_password_reset(payload, db)


@app.post('/auth/logout', response_model=schemas.AuthStatus)
def logout(response: Response):
    result = auth.logout(response)
    return {'enabled': auth.auth_enabled(), **result}


@app.get('/project-users/', response_model=List[schemas.UserSummary])
def read_project_users(request: Request, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_admin)):
    context = auth._session_context(request, db)
    memberships = db.query(models.ProjectMembership).filter(
        models.ProjectMembership.project_id == context['project_id'],
    ).order_by(models.User.display_name).join(models.User).all()
    return [
        {
            'id': membership.user.id,
            'email': membership.user.email,
            'display_name': membership.user.display_name,
            'person_owner_id': membership.user.person_owner_id,
            'is_active': membership.user.is_active,
            'role': membership.role,
        }
        for membership in memberships
    ]


@app.put('/project-users/{user_id}', response_model=schemas.UserSummary)
def update_project_user(user_id: int, payload: schemas.ProjectUserUpdate, request: Request, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_admin)):
    membership = auth.update_project_user(request, user_id, payload, db)
    crud.record_audit(db, 'update_project_user', 'user', user_id, payload.dict(exclude_unset=True), actor=actor)
    return {
        'id': membership.user.id,
        'email': membership.user.email,
        'display_name': membership.user.display_name,
        'person_owner_id': membership.user.person_owner_id,
        'is_active': membership.user.is_active,
        'role': membership.role,
    }


@app.post('/project-invitations/', response_model=schemas.ProjectInvitationResult)
def create_project_invitation(payload: schemas.ProjectInvitationCreate, request: Request, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_admin)):
    invitation, invite_url = auth.create_invitation(request, payload, db)
    crud.record_audit(db, 'create_project_invitation', 'project_invitation', invitation.id, {'email': invitation.email, 'role': invitation.role}, actor=actor)
    return {
        'id': invitation.id,
        'email': invitation.email,
        'role': invitation.role,
        'expires_at': invitation.expires_at,
        'invite_url': invite_url,
    }


@app.get('/audit-log/', response_model=List[schemas.AuditLog])
def read_audit_logs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_admin)):
    return crud.list_audit_logs(db, skip=skip, limit=limit)


@app.get('/export')
def export_backup(db: Session = Depends(get_db), actor: str = Depends(auth.require_project_admin)):
    return JSONResponse(
        content=jsonable_encoder(crud.export_data(db)),
        headers={'Content-Disposition': 'attachment; filename="patrimonio-export.json"'},
    )


@app.post('/restore', response_model=schemas.RestoreBackupResult)
def restore_backup(payload: schemas.RestoreBackupRequest, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_admin)):
    if not auth.restore_enabled():
        raise HTTPException(status_code=403, detail='Restore is disabled. Set APP_RESTORE_ENABLED=true to enable it.')
    if not payload.confirm_restore:
        raise HTTPException(status_code=400, detail='Set confirm_restore=true to replace the database from a backup')
    backup = payload.backup
    if backup.get('version') != 1:
        raise HTTPException(status_code=400, detail='Unsupported backup version')
    result = crud.restore_data(db, backup)
    crud.record_audit(db, 'restore_backup', 'backup', details=result, actor=actor)
    return result


@app.get('/prices/quote', response_model=schemas.PriceQuote)
def read_price_quote(asset_id: Optional[int] = None, provider: Optional[str] = None, symbol: Optional[str] = None, db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    asset = None
    if asset_id is not None:
        asset = crud.get_asset_by_id(db, asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail='Asset not found')
        provider = asset.price_provider
        symbol = asset.price_symbol

    try:
        quote = prices.get_quote(provider, symbol)
        if asset is not None:
            crud.record_price_quote(db, asset, quote)
        return quote
    except prices.PriceLookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get('/prices/history', response_model=List[schemas.PriceHistory])
def read_price_history(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_admin)):
    return crud.list_price_history(db, skip=skip, limit=limit)


@app.get('/investing-assets/', response_model=List[schemas.InvestingAsset])
def read_investing_assets(db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    return crud.list_investing_assets(db)


@app.put('/investing-assets/{category:path}', response_model=schemas.InvestingAsset)
def update_investing_asset(category: str, payload: schemas.InvestingAssetUpdate, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    result = crud.upsert_investing_asset(db, category, payload.is_invested)
    crud.record_audit(db, 'update_investing_asset', 'investing_asset', category, {'is_invested': payload.is_invested}, actor=actor)
    return result


@app.post('/owners/', response_model=schemas.Owner)
def create_owner(owner: schemas.OwnerCreate, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    db_owner = crud.get_owner_by_name(db, owner.name)
    if db_owner:
        raise HTTPException(status_code=400, detail='Owner already registered')
    result = crud.create_owner(
        db,
        name=owner.name,
        type=owner.type,
        is_family_member=owner.is_family_member,
    )
    crud.record_audit(db, 'create_owner', 'owner', result.id, owner.dict(), actor=actor)
    return result

@app.get('/owners/', response_model=List[schemas.Owner])
def read_owners(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    return crud.list_owners(db, skip=skip, limit=limit)


@app.put('/owners/{owner_id}', response_model=schemas.Owner)
def update_owner(owner_id: int, owner_update: schemas.OwnerUpdate, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    owner = crud.get_owner_by_id(db, owner_id)
    if owner is None:
        raise HTTPException(status_code=404, detail='Owner not found')

    data = owner_update.dict(exclude_unset=True)
    if 'name' in data and data.get('name') != owner.name:
        existing_owner = crud.get_owner_by_name(db, data.get('name'))
        if existing_owner is not None:
            raise HTTPException(status_code=400, detail='Owner already registered')

    result = crud.update_owner(db, owner, data)
    crud.record_audit(db, 'update_owner', 'owner', result.id, data, actor=actor)
    return result


@app.delete('/owners/{owner_id}', status_code=204)
def delete_owner(owner_id: int, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    owner = crud.get_owner_by_id(db, owner_id)
    if owner is None:
        raise HTTPException(status_code=404, detail='Owner not found')
    if owner.positions or owner.position_ownerships or owner.companies_owned or owner.owners_of_company:
        raise HTTPException(status_code=400, detail='Owner has positions or ownership splits')

    deleted_id = owner.id
    crud.delete_owner(db, owner)
    crud.record_audit(db, 'delete_owner', 'owner', deleted_id, actor=actor)
    return None


@app.get('/entity-ownerships/', response_model=List[schemas.EntityOwnership])
def read_entity_ownerships(db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    return crud.list_entity_ownerships(db)


def _validate_entity_ownership(payload: schemas.EntityOwnershipBase, db: Session):
    owner = crud.get_owner_by_id(db, payload.owner_id)
    owned = crud.get_owner_by_id(db, payload.owned_id)
    if owner is None or owned is None:
        raise HTTPException(status_code=404, detail='Owner or company not found')
    if owned.type != 'company':
        raise HTTPException(status_code=400, detail='The owned entity must be a company')
    if payload.effective_to and payload.effective_to < payload.effective_from:
        raise HTTPException(status_code=400, detail='The end date cannot precede the start date')


@app.post('/entity-ownerships/', response_model=schemas.EntityOwnership)
def create_entity_ownership(payload: schemas.EntityOwnershipCreate, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    _validate_entity_ownership(payload, db)
    try:
        result = crud.create_entity_ownership(db, payload.dict())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    crud.record_audit(db, 'create_entity_ownership', 'entity_ownership', result.id, payload.dict(), actor=actor)
    return {
        'id': result.id,
        'owner_id': result.owner_id,
        'owner_name': result.owner.name,
        'owned_id': result.owned_id,
        'owned_name': result.owned.name,
        'share': result.share,
        'effective_from': result.effective_from,
        'effective_to': result.effective_to,
    }


@app.put('/entity-ownerships/{ownership_id}', response_model=schemas.EntityOwnership)
def update_entity_ownership(ownership_id: int, payload: schemas.EntityOwnershipUpdate, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    relation = db.query(models.EntityOwnership).filter(models.EntityOwnership.id == ownership_id).first()
    if relation is None:
        raise HTTPException(status_code=404, detail='Entity ownership not found')
    data = payload.dict(exclude_unset=True)
    proposed = schemas.EntityOwnershipBase(
        owner_id=data.get('owner_id', relation.owner_id),
        owned_id=data.get('owned_id', relation.owned_id),
        share=data.get('share', relation.share),
        effective_from=data.get('effective_from', relation.effective_from),
        effective_to=data.get('effective_to', relation.effective_to),
    )
    _validate_entity_ownership(proposed, db)
    try:
        result = crud.update_entity_ownership(db, relation, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    crud.record_audit(db, 'update_entity_ownership', 'entity_ownership', result.id, data, actor=actor)
    return {
        'id': result.id,
        'owner_id': result.owner_id,
        'owner_name': result.owner.name,
        'owned_id': result.owned_id,
        'owned_name': result.owned.name,
        'share': result.share,
        'effective_from': result.effective_from,
        'effective_to': result.effective_to,
    }


@app.delete('/entity-ownerships/{ownership_id}', status_code=204)
def delete_entity_ownership(ownership_id: int, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    relation = db.query(models.EntityOwnership).filter(models.EntityOwnership.id == ownership_id).first()
    if relation is None:
        raise HTTPException(status_code=404, detail='Entity ownership not found')
    crud.delete_entity_ownership(db, relation)
    crud.record_audit(db, 'delete_entity_ownership', 'entity_ownership', ownership_id, actor=actor)
    return None


@app.post('/assets/', response_model=schemas.Asset)
def create_asset(asset: schemas.AssetCreate, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    existing = crud.get_asset_by_name(db, asset.name)
    if existing is not None:
        raise HTTPException(status_code=400, detail='Asset already exists')
    result = crud.create_asset(db, asset.dict())
    crud.record_audit(db, 'create_asset', 'asset', result.id, asset.dict(), actor=actor)
    return result

@app.get('/assets/', response_model=List[schemas.Asset])
def read_assets(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    return crud.list_assets(db, skip=skip, limit=limit)


@app.get('/assets/duplicates', response_model=List[schemas.DuplicateAssetGroup])
def read_duplicate_assets(db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    return crud.list_duplicate_assets(db)


@app.put('/assets/{asset_id}', response_model=schemas.Asset)
def update_asset(asset_id: int, asset_update: schemas.AssetUpdate, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    asset = crud.get_asset_by_id(db, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail='Asset not found')

    data = asset_update.dict(exclude_unset=True)
    result = crud.update_asset(db, asset, data)
    crud.record_audit(db, 'update_asset', 'asset', result.id, data, actor=actor)
    return result


@app.delete('/assets/{asset_id}', status_code=204)
def delete_asset(asset_id: int, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    asset = crud.get_asset_by_id(db, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail='Asset not found')

    deleted_id = asset.id
    crud.delete_asset(db, asset)
    crud.record_audit(db, 'delete_asset', 'asset', deleted_id, actor=actor)
    return None


@app.get('/assets/{asset_id}/company-valuation', response_model=schemas.CompanyValuationSnapshot)
def read_company_valuation(asset_id: int, as_of_date: str, db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    asset = crud.get_asset_by_id(db, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail='Asset not found')
    parsed_date = datetime.fromisoformat(as_of_date)
    return crud.get_company_valuation(db, asset, parsed_date)


@app.put('/assets/{asset_id}/company-valuation', response_model=schemas.CompanyValuationSnapshot)
def update_company_valuation(asset_id: int, payload: schemas.CompanyValuationUpdate, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    asset = crud.get_asset_by_id(db, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail='Asset not found')
    result = crud.replace_company_valuation(
        db,
        asset,
        payload.as_of_date,
        [item.dict() for item in payload.items],
    )
    crud.record_audit(db, 'replace_company_valuation', 'asset', asset.id, {'as_of_date': payload.as_of_date, 'items': len(payload.items)}, actor=actor)
    return result


def _validate_ownership_shares(ownership_update: schemas.PositionOwnershipUpdate, db: Session):
    seen_owner_ids = set()
    total_share = 0.0
    shares = []
    for item in ownership_update.shares:
        if item.owner_id in seen_owner_ids:
            raise HTTPException(status_code=400, detail='Duplicate owner in ownership split')
        seen_owner_ids.add(item.owner_id)

        owner = crud.get_owner_by_id(db, item.owner_id)
        if owner is None:
            raise HTTPException(status_code=404, detail=f'Owner not found: {item.owner_id}')
        if item.share < 0 or item.share > 1:
            raise HTTPException(status_code=400, detail='Ownership share must be between 0 and 1')

        total_share += item.share
        shares.append({'owner_id': item.owner_id, 'share': item.share})

    if total_share > 1.000001:
        raise HTTPException(status_code=400, detail='Ownership shares cannot exceed 100%')

    return shares


@app.get('/positions/ownership', response_model=List[schemas.PositionOwnershipState])
def read_all_position_ownership(db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    return crud.list_position_ownership(db)


@app.get('/positions/{position_id}/ownership', response_model=List[schemas.OwnershipShare])
def read_position_ownership(position_id: int, db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    position = crud.get_position_by_id(db, position_id)
    if position is None:
        raise HTTPException(status_code=404, detail='Position not found')

    return crud.get_position_ownership(db, position)


@app.put('/positions/{position_id}/ownership', response_model=List[schemas.OwnershipShare])
def update_position_ownership(position_id: int, ownership_update: schemas.PositionOwnershipUpdate, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    position = crud.get_position_by_id(db, position_id)
    if position is None:
        raise HTTPException(status_code=404, detail='Position not found')

    shares = _validate_ownership_shares(ownership_update, db)
    result = crud.replace_position_ownership(db, position, shares)
    crud.record_audit(db, 'replace_position_ownership', 'position', position.id, {'shares': shares}, actor=actor)
    return result


@app.get('/positions/', response_model=List[schemas.Position])
def read_positions(as_of_date: Optional[str] = None, asset_id: Optional[int] = None, skip: int = 0, limit: int = 100, db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    parsed_date = datetime.fromisoformat(as_of_date) if as_of_date else None
    return crud.get_positions(db, as_of_date=parsed_date, asset_id=asset_id, skip=skip, limit=limit)


@app.get('/positions/snapshot', response_model=List[schemas.PositionSnapshotRow])
def read_position_snapshot(as_of_date: Optional[str] = None, db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    parsed_date = datetime.fromisoformat(as_of_date) if as_of_date else None
    return crud.get_position_snapshot_rows(db, as_of_date=parsed_date)


@app.post('/positions/', response_model=schemas.Position)
def create_position(position: schemas.PositionCreate, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    asset = crud.get_asset_by_id(db, position.asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail='Asset not found')

    owner = None
    if position.owner_id is not None:
        owner = crud.get_owner_by_id(db, position.owner_id)
        if owner is None:
            raise HTTPException(status_code=404, detail='Owner not found')

    result = crud.create_position(
        db,
        asset,
        as_of_date=position.as_of_date,
        quantity=position.quantity or 0.0,
        value=position.value or 0.0,
        owner=owner,
        broker=position.broker,
        source=position.source or 'manual',
    )
    crud.record_audit(db, 'create_position', 'position', result.id, position.dict(), actor=actor)
    return result


@app.put('/positions/{position_id}', response_model=schemas.Position)
def update_position(position_id: int, position_update: schemas.PositionUpdate, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    position = crud.get_position_by_id(db, position_id)
    if position is None:
        raise HTTPException(status_code=404, detail='Position not found')

    data = position_update.dict(exclude_unset=True)
    if 'owner_id' in data and data.get('owner_id') is not None and crud.get_owner_by_id(db, data.get('owner_id')) is None:
        raise HTTPException(status_code=404, detail='Owner not found')

    result = crud.update_position(db, position, data)
    crud.record_audit(db, 'update_position', 'position', result.id, data, actor=actor)
    return result


@app.delete('/positions/{position_id}', status_code=204)
def delete_position(position_id: int, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    position = crud.get_position_by_id(db, position_id)
    if position is None:
        raise HTTPException(status_code=404, detail='Position not found')

    deleted_id = position.id
    crud.delete_position(db, position)
    crud.record_audit(db, 'delete_position', 'position', deleted_id, actor=actor)
    return None


@app.post('/positions/bulk', response_model=List[schemas.Position])
def save_positions_bulk(payload: schemas.BulkPositionSave, db: Session = Depends(get_db), actor: str = Depends(auth.require_project_editor)):
    grouped_positions = {}
    saved_positions = []
    snapshot_date = payload.as_of_date or (payload.positions[0].as_of_date if payload.positions else None)

    for item in payload.positions:
        asset = crud.get_asset_by_id(db, item.asset_id) if item.asset_id is not None else None
        asset_name = (item.asset_name or '').strip()
        if asset is None and asset_name:
            asset = crud.create_asset(db, {
                'name': asset_name,
                'category': item.category,
                'valuation_method': 'market_direct',
            }, commit=False)
        if asset is None:
            raise HTTPException(status_code=404, detail=f'Asset not found: {item.asset_id}')

        if item.category is not None and item.category != asset.category:
            crud.update_asset(db, asset, {'category': item.category}, commit=False)

        owner = None
        if item.owner_id is not None:
            owner = crud.get_owner_by_id(db, item.owner_id)
            if owner is None:
                raise HTTPException(status_code=404, detail=f'Owner not found: {item.owner_id}')

        existing_position = crud.get_position_by_id(db, item.position_id) if item.position_id else None
        if (item.value or 0.0) <= 0:
            if existing_position is not None:
                crud.delete_position(db, existing_position, commit=False)
            continue

        key = (item.as_of_date, asset.id, item.owner_id, item.broker or None)
        if key not in grouped_positions:
            ownership_shares = _validate_ownership_shares(
                schemas.PositionOwnershipUpdate(shares=item.ownership_shares),
                db,
            ) if item.ownership_shares else []
            grouped_positions[key] = {
                'asset': asset,
                'owner': owner,
                'as_of_date': item.as_of_date,
                'asset_id': asset.id,
                'owner_id': item.owner_id,
                'broker': item.broker or None,
                'quantity': 0.0,
                'value': 0.0,
                'source': item.source or 'manual',
                'position_ids': [],
                'ownership_shares': ownership_shares,
            }

        grouped_positions[key]['quantity'] += item.quantity or 0.0
        grouped_positions[key]['value'] += item.value or 0.0
        grouped_positions[key]['source'] = item.source or grouped_positions[key]['source']
        if item.ownership_shares:
            grouped_positions[key]['ownership_shares'] = _validate_ownership_shares(
                schemas.PositionOwnershipUpdate(shares=item.ownership_shares),
                db,
            )
        if item.position_id:
            grouped_positions[key]['position_ids'].append(item.position_id)

    for group in grouped_positions.values():
        matching_positions = crud.get_positions_by_snapshot_key(
            db,
            group['asset_id'],
            group['as_of_date'],
            owner_id=group['owner_id'],
            broker=group['broker'],
        )
        source_positions = [
            crud.get_position_by_id(db, position_id)
            for position_id in group['position_ids']
        ]
        source_positions = [position for position in source_positions if position is not None]

        target_position = source_positions[0] if source_positions else (matching_positions[0] if matching_positions else None)
        if target_position is not None:
            saved = crud.update_position(db, target_position, {
                'asset_id': group['asset_id'],
                'owner_id': group['owner_id'],
                'as_of_date': group['as_of_date'],
                'quantity': group['quantity'],
                'value': group['value'],
                'broker': group['broker'],
                'source': group['source'],
            }, commit=False)
        else:
            saved = crud.create_position(
                db,
                group['asset'],
                as_of_date=group['as_of_date'],
                quantity=group['quantity'],
                value=group['value'],
                owner=group['owner'],
                broker=group['broker'],
                source=group['source'],
                commit=False,
            )

        saved_positions.append(saved)
        if group['owner_id'] is None:
            shares = group['ownership_shares'] or crud.default_family_ownership_shares(db)
            crud.replace_position_ownership(db, saved, shares, commit=False)
        else:
            crud.replace_position_ownership(db, saved, [], commit=False)

        cleanup_positions = matching_positions + source_positions
        seen_cleanup_ids = set()
        for cleanup_position in cleanup_positions:
            if cleanup_position.id == saved.id or cleanup_position.id in seen_cleanup_ids:
                continue
            seen_cleanup_ids.add(cleanup_position.id)
            crud.delete_position(db, cleanup_position, commit=False)

    if payload.replace_snapshot and snapshot_date is not None:
        keep_position_ids = [position.id for position in saved_positions]
        crud.delete_positions_for_date_except(db, snapshot_date, keep_position_ids, commit=False)

    db.commit()

    crud.record_audit(db, 'save_positions_bulk', 'position_snapshot', snapshot_date, {
        'rows_in': len(payload.positions),
        'rows_saved': len(saved_positions),
        'replace_snapshot': payload.replace_snapshot,
    }, actor=actor)
    return saved_positions


@app.get('/dashboard/summary', response_model=schemas.DashboardSummary)
def read_dashboard_summary(as_of_date: Optional[str] = None, db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    parsed_date = datetime.fromisoformat(as_of_date) if as_of_date else None
    return crud.get_dashboard_summary(db, as_of_date=parsed_date)

@app.get('/dashboard/details', response_model=List[schemas.DashboardDetailRow])
def read_dashboard_details(as_of_date: Optional[str] = None, db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    parsed_date = datetime.fromisoformat(as_of_date) if as_of_date else None
    return crud.get_dashboard_details(db, as_of_date=parsed_date)

@app.get('/dashboard/dates')
def read_dashboard_dates(db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    return [value.date().isoformat() if hasattr(value, 'date') else str(value) for value in crud.get_available_dates(db)]


@app.get('/dashboard/history', response_model=List[schemas.DashboardHistoryPoint])
def read_dashboard_history(db: Session = Depends(get_db), actor: str = Depends(auth.require_auth)):
    return crud.get_dashboard_history(db)
