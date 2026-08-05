from datetime import datetime
import json
from sqlalchemy.orm import Session
from app import models as models_module

FAMILY_OWNER_NAMES = {'Antonio', 'Patri', 'Patricia'}


def get_default_project(db: Session):
    project = db.query(models_module.Project).order_by(models_module.Project.id).first()
    if project is None:
        project = models_module.Project(name='Family Patrimonio', base_currency='EUR')
        db.add(project)
        db.flush()
    return project


def initialize_project_data(db: Session):
    """Attach legacy catalog rows to the single initial project without changing holdings."""
    project = get_default_project(db)
    db.query(models_module.Owner).filter(models_module.Owner.project_id.is_(None)).update(
        {models_module.Owner.project_id: project.id}, synchronize_session=False,
    )
    db.query(models_module.Asset).filter(models_module.Asset.project_id.is_(None)).update(
        {models_module.Asset.project_id: project.id}, synchronize_session=False,
    )
    for owner in db.query(models_module.Owner).filter(models_module.Owner.name.in_(FAMILY_OWNER_NAMES)).all():
        owner.is_family_member = True
    db.commit()
    db.refresh(project)
    return project


def is_family_owner(owner: models_module.Owner = None):
    return owner is not None and (bool(owner.is_family_member) or owner.name in FAMILY_OWNER_NAMES)


def record_audit(db: Session, action: str, entity_type: str = None, entity_id=None, details: dict = None, actor: str = 'api'):
    entry = models_module.AuditLog(
        actor=actor,
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        details=json.dumps(details or {}, default=str, sort_keys=True),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def list_audit_logs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models_module.AuditLog).order_by(models_module.AuditLog.created_at.desc(), models_module.AuditLog.id.desc()).offset(skip).limit(limit).all()


def default_asset_is_investment(asset_data: dict):
    asset_type = str(asset_data.get('asset_type') or '').strip().lower()
    category = str(asset_data.get('category') or '').strip().lower()
    name = str(asset_data.get('name') or '').strip().lower()
    if asset_type == 'cash' or category in {'cash', 'caja', 'efectivo'}:
        return False
    if category == 'casa' or name == 'casa':
        return False
    return True


def default_category_is_invested(category: str):
    normalized = str(category or '').strip().lower()
    return normalized not in {'cash', 'caja', 'efectivo', 'casa'}


def asset_is_investment(asset):
    if asset is None:
        return False
    if asset.is_investment is not None:
        return bool(asset.is_investment)
    return default_asset_is_investment({
        'name': asset.name,
        'category': asset.category,
        'asset_type': asset.asset_type,
    })


def list_investing_assets(db: Session):
    categories = {
        row[0] or 'Uncategorized'
        for row in db.query(models_module.Asset.category).distinct().all()
    }
    existing = {
        item.category: item
        for item in db.query(models_module.InvestingAsset).all()
    }
    for category in categories:
        if category not in existing:
            existing[category] = models_module.InvestingAsset(
                category=category,
                is_invested=default_category_is_invested(category),
            )
    return [
        {
            'category': category,
            'is_invested': bool(existing[category].is_invested),
        }
        for category in sorted(existing)
    ]


def upsert_investing_asset(db: Session, category: str, is_invested: bool):
    normalized_category = category or 'Uncategorized'
    item = db.query(models_module.InvestingAsset).filter(
        models_module.InvestingAsset.category == normalized_category,
    ).first()
    if item is None:
        item = models_module.InvestingAsset(category=normalized_category, is_invested=bool(is_invested))
    else:
        item.is_invested = bool(is_invested)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def record_price_quote(db: Session, asset: models_module.Asset, quote: dict):
    item = models_module.PriceHistory(
        asset_id=asset.id,
        provider=quote.get('provider'),
        symbol=quote.get('symbol'),
        price=float(quote.get('price') or 0.0),
        currency=quote.get('currency'),
        as_of=quote.get('as_of'),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def list_price_history(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models_module.PriceHistory).order_by(
        models_module.PriceHistory.created_at.desc(),
        models_module.PriceHistory.id.desc(),
    ).offset(skip).limit(limit).all()


def export_data(db: Session):
    owners = db.query(models_module.Owner).order_by(models_module.Owner.id).all()
    assets = db.query(models_module.Asset).order_by(models_module.Asset.id).all()
    asset_valuations = db.query(models_module.AssetValuation).order_by(
        models_module.AssetValuation.as_of_date,
        models_module.AssetValuation.id,
    ).all()
    company_valuation_items = db.query(models_module.CompanyValuationItem).order_by(
        models_module.CompanyValuationItem.as_of_date,
        models_module.CompanyValuationItem.asset_id,
        models_module.CompanyValuationItem.id,
    ).all()
    price_history = db.query(models_module.PriceHistory).order_by(
        models_module.PriceHistory.created_at,
        models_module.PriceHistory.id,
    ).all()
    positions = db.query(models_module.Position).order_by(
        models_module.Position.as_of_date,
        models_module.Position.id,
    ).all()
    position_ownerships = db.query(models_module.PositionOwnership).order_by(
        models_module.PositionOwnership.position_id,
        models_module.PositionOwnership.owner_id,
    ).all()

    return {
        'version': 1,
        'owners': [
            {
                'id': owner.id,
                'name': owner.name,
                'type': owner.type,
            }
            for owner in owners
        ],
        'assets': [
            {
                'id': asset.id,
                'name': asset.name,
                'category': asset.category,
                'asset_type': asset.asset_type,
                'valuation_method': asset.valuation_method,
                'price_provider': asset.price_provider,
                'price_symbol': asset.price_symbol,
                'is_investment': asset_is_investment(asset),
                'is_shared': asset.is_shared,
                'created_at': asset.created_at,
            }
            for asset in assets
        ],
        'investing_assets': [
            {
                'category': item['category'],
                'is_invested': item['is_invested'],
            }
            for item in list_investing_assets(db)
        ],
        'positions': [
            {
                'id': position.id,
                'asset_id': position.asset_id,
                'owner_id': position.owner_id,
                'as_of_date': position.as_of_date,
                'quantity': position.quantity,
                'value': position.value,
                'broker': position.broker,
                'source': position.source,
                'created_at': position.created_at,
            }
            for position in positions
        ],
        'asset_valuations': [
            {
                'id': valuation.id,
                'asset_id': valuation.asset_id,
                'as_of_date': valuation.as_of_date,
                'value': valuation.value,
                'source': valuation.source,
                'created_at': valuation.created_at,
            }
            for valuation in asset_valuations
        ],
        'company_valuation_items': [
            {
                'id': item.id,
                'asset_id': item.asset_id,
                'as_of_date': item.as_of_date,
                'item_type': item.item_type,
                'name': item.name,
                'amount': item.amount,
                'created_at': item.created_at,
            }
            for item in company_valuation_items
        ],
        'price_history': [
            {
                'id': item.id,
                'asset_id': item.asset_id,
                'provider': item.provider,
                'symbol': item.symbol,
                'price': item.price,
                'currency': item.currency,
                'as_of': item.as_of,
                'created_at': item.created_at,
            }
            for item in price_history
        ],
        'position_ownerships': [
            {
                'position_id': ownership.position_id,
                'owner_id': ownership.owner_id,
                'share': ownership.share,
            }
            for ownership in position_ownerships
        ],
    }


def _parse_datetime(value):
    if value is None or isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    return datetime.fromisoformat(text)


def restore_data(db: Session, payload: dict):
    owner_id_map = {}
    asset_id_map = {}
    position_id_map = {}

    db.query(models_module.PositionOwnership).delete()
    db.query(models_module.CompanyValuationItem).delete()
    db.query(models_module.PriceHistory).delete()
    db.query(models_module.AssetValuation).delete()
    db.query(models_module.Position).delete()
    db.query(models_module.InvestingAsset).delete()
    db.query(models_module.Ownership).delete()
    db.query(models_module.Asset).delete()
    db.query(models_module.Owner).delete()
    db.flush()

    for row in payload.get('owners') or []:
        owner = models_module.Owner(
            name=row.get('name'),
            type=row.get('type') or 'person',
        )
        db.add(owner)
        db.flush()
        owner_id_map[row.get('id')] = owner.id

    for row in payload.get('assets') or []:
        asset = models_module.Asset(
            name=row.get('name'),
            category=row.get('category'),
            asset_type=row.get('asset_type'),
            valuation_method=row.get('valuation_method') or 'market_direct',
            price_provider=row.get('price_provider') or 'manual',
            price_symbol=row.get('price_symbol'),
            is_investment=row.get('is_investment'),
            is_shared=bool(row.get('is_shared') or False),
            created_at=_parse_datetime(row.get('created_at')) or datetime.utcnow(),
        )
        db.add(asset)
        db.flush()
        asset_id_map[row.get('id')] = asset.id

    for row in payload.get('investing_assets') or []:
        category = row.get('category')
        if not category:
            continue
        db.add(models_module.InvestingAsset(
            category=category,
            is_invested=bool(row.get('is_invested')),
        ))

    for row in payload.get('positions') or []:
        asset_id = asset_id_map.get(row.get('asset_id'))
        if asset_id is None:
            continue
        old_owner_id = row.get('owner_id')
        position = models_module.Position(
            asset_id=asset_id,
            owner_id=owner_id_map.get(old_owner_id) if old_owner_id is not None else None,
            as_of_date=_parse_datetime(row.get('as_of_date')),
            quantity=float(row.get('quantity') or 0.0),
            value=float(row.get('value') or 0.0),
            broker=row.get('broker'),
            source=row.get('source') or 'restore',
            created_at=_parse_datetime(row.get('created_at')) or datetime.utcnow(),
        )
        db.add(position)
        db.flush()
        position_id_map[row.get('id')] = position.id

    for row in payload.get('asset_valuations') or []:
        asset_id = asset_id_map.get(row.get('asset_id'))
        if asset_id is None:
            continue
        db.add(models_module.AssetValuation(
            asset_id=asset_id,
            as_of_date=_parse_datetime(row.get('as_of_date')),
            value=float(row.get('value') or 0.0),
            source=row.get('source') or 'restore',
            created_at=_parse_datetime(row.get('created_at')) or datetime.utcnow(),
        ))

    for row in payload.get('company_valuation_items') or []:
        asset_id = asset_id_map.get(row.get('asset_id'))
        if asset_id is None:
            continue
        db.add(models_module.CompanyValuationItem(
            asset_id=asset_id,
            as_of_date=_parse_datetime(row.get('as_of_date')),
            item_type=row.get('item_type'),
            name=row.get('name') or 'Restored item',
            amount=float(row.get('amount') or 0.0),
            created_at=_parse_datetime(row.get('created_at')) or datetime.utcnow(),
        ))

    for row in payload.get('price_history') or []:
        asset_id = asset_id_map.get(row.get('asset_id'))
        if asset_id is None:
            continue
        db.add(models_module.PriceHistory(
            asset_id=asset_id,
            provider=row.get('provider'),
            symbol=row.get('symbol'),
            price=float(row.get('price') or 0.0),
            currency=row.get('currency'),
            as_of=row.get('as_of'),
            created_at=_parse_datetime(row.get('created_at')) or datetime.utcnow(),
        ))

    for row in payload.get('position_ownerships') or []:
        position_id = position_id_map.get(row.get('position_id'))
        owner_id = owner_id_map.get(row.get('owner_id'))
        if position_id is None or owner_id is None:
            continue
        db.add(models_module.PositionOwnership(
            position_id=position_id,
            owner_id=owner_id,
            share=float(row.get('share') or 0.0),
        ))

    db.commit()
    return {
        'owners': len(owner_id_map),
        'assets': len(asset_id_map),
        'positions': len(position_id_map),
        'investing_assets': len(payload.get('investing_assets') or []),
    }


def create_owner(db: Session, name: str, type: str = 'person', is_family_member: bool = False, project=None):
    project = project or get_default_project(db)
    owner = models_module.Owner(
        name=name,
        type=type,
        is_family_member=bool(is_family_member),
        project_id=project.id,
    )
    db.add(owner)
    db.commit()
    db.refresh(owner)
    return owner


def get_owner_by_name(db: Session, name: str):
    return db.query(models_module.Owner).filter(models_module.Owner.name == name).first()


def get_owner_by_id(db: Session, owner_id: int):
    return db.query(models_module.Owner).filter(models_module.Owner.id == owner_id).first()


def update_owner(db: Session, owner: models_module.Owner, owner_data: dict):
    if 'name' in owner_data and owner_data.get('name') is not None:
        owner.name = owner_data.get('name')
    if 'type' in owner_data and owner_data.get('type') is not None:
        owner.type = owner_data.get('type')
    if 'is_family_member' in owner_data and owner_data.get('is_family_member') is not None:
        owner.is_family_member = bool(owner_data.get('is_family_member'))

    db.add(owner)
    db.commit()
    db.refresh(owner)
    return owner


def delete_owner(db: Session, owner: models_module.Owner):
    db.delete(owner)
    db.commit()


def list_owners(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models_module.Owner).filter(
        models_module.Owner.archived_at.is_(None),
    ).order_by(models_module.Owner.name).offset(skip).limit(limit).all()


def list_entity_ownerships(db: Session):
    rows = db.query(models_module.EntityOwnership).order_by(
        models_module.EntityOwnership.owned_id,
        models_module.EntityOwnership.effective_from.desc(),
        models_module.EntityOwnership.id,
    ).all()
    return [
        {
            'id': row.id,
            'owner_id': row.owner_id,
            'owner_name': row.owner.name,
            'owned_id': row.owned_id,
            'owned_name': row.owned.name,
            'share': row.share,
            'effective_from': row.effective_from,
            'effective_to': row.effective_to,
        }
        for row in rows
    ]


def _entity_ownership_creates_cycle(db: Session, owner_id: int, owned_id: int, ignore_id: int = None):
    edges = db.query(models_module.EntityOwnership.owner_id, models_module.EntityOwnership.owned_id)
    if ignore_id is not None:
        edges = edges.filter(models_module.EntityOwnership.id != ignore_id)
    graph = {}
    for current_owner_id, current_owned_id in edges.all():
        graph.setdefault(current_owner_id, set()).add(current_owned_id)

    pending = [owned_id]
    visited = set()
    while pending:
        current = pending.pop()
        if current == owner_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(graph.get(current, ()))
    return False


def create_entity_ownership(db: Session, data: dict):
    owner_id = int(data['owner_id'])
    owned_id = int(data['owned_id'])
    if owner_id == owned_id:
        raise ValueError('An entity cannot own itself')
    if _entity_ownership_creates_cycle(db, owner_id, owned_id):
        raise ValueError('This ownership relationship would create a company ownership cycle')
    relation = models_module.EntityOwnership(
        owner_id=owner_id,
        owned_id=owned_id,
        share=float(data['share']),
        effective_from=data['effective_from'],
        effective_to=data.get('effective_to'),
    )
    db.add(relation)
    db.commit()
    db.refresh(relation)
    return relation


def update_entity_ownership(db: Session, relation: models_module.EntityOwnership, data: dict):
    owner_id = int(data.get('owner_id', relation.owner_id))
    owned_id = int(data.get('owned_id', relation.owned_id))
    if owner_id == owned_id:
        raise ValueError('An entity cannot own itself')
    if _entity_ownership_creates_cycle(db, owner_id, owned_id, ignore_id=relation.id):
        raise ValueError('This ownership relationship would create a company ownership cycle')
    relation.owner_id = owner_id
    relation.owned_id = owned_id
    if 'share' in data:
        relation.share = float(data['share'])
    if 'effective_from' in data:
        relation.effective_from = data['effective_from']
    if 'effective_to' in data:
        relation.effective_to = data['effective_to']
    db.add(relation)
    db.commit()
    db.refresh(relation)
    return relation


def delete_entity_ownership(db: Session, relation: models_module.EntityOwnership):
    db.delete(relation)
    db.commit()


def create_asset(db: Session, asset_data: dict, commit: bool = True):
    # idempotent: update if asset with same name exists
    asset = db.query(models_module.Asset).filter(models_module.Asset.name == asset_data.get('name')).first()
    if asset:
        asset.category = asset_data.get('category')
        asset.asset_type = asset_data.get('asset_type')
        asset.valuation_method = asset_data.get('valuation_method', asset.valuation_method)
        asset.price_provider = asset_data.get('price_provider', asset.price_provider)
        asset.price_symbol = asset_data.get('price_symbol', asset.price_symbol)
        if 'is_investment' in asset_data and asset_data.get('is_investment') is not None:
            asset.is_investment = bool(asset_data.get('is_investment'))
        elif asset.is_investment is None:
            asset.is_investment = default_asset_is_investment(asset_data)
        if 'is_shared' in asset_data:
            asset.is_shared = bool(asset_data.get('is_shared'))
        db.add(asset)
        if commit:
            db.commit()
            db.refresh(asset)
        else:
            db.flush()
        return asset

    asset = models_module.Asset(
        name=asset_data.get('name'),
        category=asset_data.get('category'),
        asset_type=asset_data.get('asset_type'),
        valuation_method=asset_data.get('valuation_method', 'market_direct'),
        price_provider=asset_data.get('price_provider', 'manual'),
        price_symbol=asset_data.get('price_symbol'),
        is_investment=asset_data.get('is_investment') if asset_data.get('is_investment') is not None else default_asset_is_investment(asset_data),
        is_shared=bool(asset_data.get('is_shared', False)),
        project_id=get_default_project(db).id,
    )
    db.add(asset)
    if commit:
        db.commit()
        db.refresh(asset)
    else:
        db.flush()
    return asset


def get_asset_by_id(db: Session, asset_id: int):
    return db.query(models_module.Asset).filter(models_module.Asset.id == asset_id).first()


def get_asset_by_name(db: Session, name: str):
    return db.query(models_module.Asset).filter(models_module.Asset.name == name).first()


def list_duplicate_assets(db: Session):
    buckets = {}
    for asset in db.query(models_module.Asset).order_by(models_module.Asset.name, models_module.Asset.id).all():
        key = str(asset.name or '').strip().lower()
        if not key:
            continue
        buckets.setdefault(key, []).append(asset)
    return [
        {
            'name': items[0].name,
            'count': len(items),
            'assets': [
                {
                    'id': asset.id,
                    'name': asset.name,
                    'category': asset.category,
                    'asset_type': asset.asset_type,
                }
                for asset in items
            ],
        }
        for items in buckets.values()
        if len(items) > 1
    ]


def update_asset(db: Session, asset: models_module.Asset, asset_data: dict, commit: bool = True):
    if 'name' in asset_data and asset_data.get('name') is not None:
        asset.name = asset_data.get('name')
    if 'category' in asset_data:
        asset.category = asset_data.get('category')
    if 'asset_type' in asset_data:
        asset.asset_type = asset_data.get('asset_type')
    if 'valuation_method' in asset_data and asset_data.get('valuation_method') is not None:
        asset.valuation_method = asset_data.get('valuation_method')
    if 'price_provider' in asset_data and asset_data.get('price_provider') is not None:
        asset.price_provider = asset_data.get('price_provider')
    if 'price_symbol' in asset_data:
        asset.price_symbol = asset_data.get('price_symbol')
    if 'is_investment' in asset_data:
        asset.is_investment = asset_data.get('is_investment')
    if 'is_shared' in asset_data:
        asset.is_shared = bool(asset_data.get('is_shared'))

    db.add(asset)
    if commit:
        db.commit()
        db.refresh(asset)
    else:
        db.flush()
    return asset


def delete_asset(db: Session, asset: models_module.Asset):
    db.delete(asset)
    db.commit()


def create_asset_valuation(db: Session, asset: models_module.Asset, as_of_date, value: float, source: str = 'import'):
    existing = db.query(models_module.AssetValuation).filter(
        models_module.AssetValuation.asset_id == asset.id,
        models_module.AssetValuation.as_of_date == as_of_date,
    ).first()
    if existing:
        existing.value = value
        existing.source = source
        db.add(existing)
        db.commit()
        db.refresh(existing)
        return existing

    valuation = models_module.AssetValuation(asset_id=asset.id, as_of_date=as_of_date, value=value, source=source)
    db.add(valuation)
    db.commit()
    db.refresh(valuation)
    return valuation


def get_company_valuation(db: Session, asset: models_module.Asset, as_of_date):
    items = db.query(models_module.CompanyValuationItem).filter(
        models_module.CompanyValuationItem.asset_id == asset.id,
        models_module.CompanyValuationItem.as_of_date == as_of_date,
    ).order_by(models_module.CompanyValuationItem.item_type, models_module.CompanyValuationItem.id).all()
    assets_total = sum(float(item.amount or 0.0) for item in items if item.item_type == 'asset')
    liabilities_total = sum(float(item.amount or 0.0) for item in items if item.item_type == 'liability')
    return {
        'asset_id': asset.id,
        'as_of_date': as_of_date,
        'items': items,
        'assets_total': round(assets_total, 2),
        'liabilities_total': round(liabilities_total, 2),
        'net_value': round(assets_total - liabilities_total, 2),
    }


def replace_company_valuation(db: Session, asset: models_module.Asset, as_of_date, items: list):
    db.query(models_module.CompanyValuationItem).filter(
        models_module.CompanyValuationItem.asset_id == asset.id,
        models_module.CompanyValuationItem.as_of_date == as_of_date,
    ).delete()
    db.flush()
    for item in items:
        name = (item.get('name') or '').strip()
        item_type = item.get('item_type')
        if item_type not in {'asset', 'liability'} or not name:
            continue
        db.add(models_module.CompanyValuationItem(
            asset_id=asset.id,
            as_of_date=as_of_date,
            item_type=item_type,
            name=name,
            amount=float(item.get('amount') or 0.0),
        ))
    db.commit()
    return get_company_valuation(db, asset, as_of_date)


def create_position(
    db: Session,
    asset: models_module.Asset,
    as_of_date,
    quantity: float,
    value: float,
    owner: models_module.Owner = None,
    broker: str = None,
    source: str = 'import',
    commit: bool = True,
):
    owner_id = owner.id if owner is not None else None
    existing = db.query(models_module.Position).filter(
        models_module.Position.asset_id == asset.id,
        models_module.Position.as_of_date == as_of_date,
        models_module.Position.owner_id == owner_id,
        models_module.Position.broker == broker,
    ).first()
    if existing:
        existing.quantity = quantity
        existing.value = value
        existing.owner_id = owner_id
        existing.broker = broker
        existing.source = source
        db.add(existing)
        if commit:
            db.commit()
            db.refresh(existing)
        else:
            db.flush()
        return existing

    position = models_module.Position(
        asset_id=asset.id,
        owner_id=owner_id,
        as_of_date=as_of_date,
        quantity=quantity,
        value=value,
        broker=broker,
        source=source,
    )
    db.add(position)
    if commit:
        db.commit()
        db.refresh(position)
    else:
        db.flush()
    return position


def get_position_by_id(db: Session, position_id: int):
    return db.query(models_module.Position).filter(models_module.Position.id == position_id).first()


def get_position_ownership(db: Session, position: models_module.Position):
    return [
        {
            'owner_id': ownership.owner_id,
            'owner_name': ownership.owner.name if ownership.owner else None,
            'share': float(ownership.share or 0.0),
        }
        for ownership in sorted(position.ownerships, key=lambda item: item.owner.name if item.owner else '')
    ]


def list_position_ownership(db: Session):
    return [
        {
            'position_id': position.id,
            'shares': get_position_ownership(db, position),
        }
        for position in db.query(models_module.Position).all()
    ]


def replace_position_ownership(db: Session, position: models_module.Position, shares: list, commit: bool = True):
    db.query(models_module.PositionOwnership).filter(models_module.PositionOwnership.position_id == position.id).delete()
    db.flush()

    for share in shares:
        owner = get_owner_by_id(db, share['owner_id'])
        if owner is None:
            continue
        value = float(share.get('share') or 0.0)
        if value <= 0:
            continue
        db.add(models_module.PositionOwnership(position_id=position.id, owner_id=owner.id, share=value))

    if commit:
        db.commit()
        db.refresh(position)
    else:
        db.flush()
        db.expire(position, ['ownerships'])
    return get_position_ownership(db, position)


def default_family_ownership_shares(db: Session):
    shares = []
    antonio = get_owner_by_name(db, 'Antonio')
    patri = get_owner_by_name(db, 'Patri') or get_owner_by_name(db, 'Patricia')
    if antonio is not None:
        shares.append({'owner_id': antonio.id, 'share': 0.5})
    if patri is not None:
        shares.append({'owner_id': patri.id, 'share': 0.5})
    return shares


def _family_attribution_for_entity(db: Session, entity: models_module.Owner, as_of_date, visited=None):
    """Resolve a legal entity upward to the family's dated beneficial owners."""
    if entity is None:
        return {}
    if is_family_owner(entity):
        return {entity.id: 1.0}

    visited = set(visited or ())
    if entity.id in visited:
        return {}
    visited.add(entity.id)

    query = db.query(models_module.EntityOwnership).filter(
        models_module.EntityOwnership.owned_id == entity.id,
        models_module.EntityOwnership.effective_from <= as_of_date,
    )
    relationships = query.filter(
        (models_module.EntityOwnership.effective_to.is_(None)) |
        (models_module.EntityOwnership.effective_to >= as_of_date),
    ).all()
    resolved = {}
    for relationship in relationships:
        for owner_id, owner_share in _family_attribution_for_entity(
            db,
            relationship.owner,
            as_of_date,
            visited,
        ).items():
            resolved[owner_id] = resolved.get(owner_id, 0.0) + float(relationship.share or 0.0) * owner_share
    return resolved


def position_family_attribution_shares(db: Session, position: models_module.Position, as_of_date):
    """Return family beneficial-owner shares for one raw position."""
    resolved = {}

    def add_entity_share(entity, share):
        for owner_id, owner_share in _family_attribution_for_entity(db, entity, as_of_date).items():
            resolved[owner_id] = resolved.get(owner_id, 0.0) + float(share or 0.0) * owner_share

    if position.owner is not None:
        add_entity_share(position.owner, 1.0)
        return resolved

    shares = get_position_ownership(db, position) or default_family_ownership_shares(db)
    for share_row in shares:
        add_entity_share(get_owner_by_id(db, share_row['owner_id']), share_row.get('share'))
    return resolved


def migrate_legacy_asset_ownership_to_positions(db: Session):
    migrated = 0
    positions = db.query(models_module.Position).filter(models_module.Position.owner_id == None).all()
    for position in positions:
        if position.ownerships or position.asset is None or not position.asset.ownerships:
            continue
        shares = [
            {'owner_id': ownership.owner_id, 'share': float(ownership.share or 0.0)}
            for ownership in position.asset.ownerships
        ]
        replace_position_ownership(db, position, shares)
        migrated += 1
    db.query(models_module.Ownership).delete()
    db.commit()
    return migrated


def get_positions_by_snapshot_key(db: Session, asset_id: int, as_of_date, owner_id: int = None, broker: str = None):
    return db.query(models_module.Position).filter(
        models_module.Position.asset_id == asset_id,
        models_module.Position.as_of_date == as_of_date,
        models_module.Position.owner_id == owner_id,
        models_module.Position.broker == broker,
    ).order_by(models_module.Position.id).all()


def update_position(db: Session, position: models_module.Position, position_data: dict, commit: bool = True):
    if 'asset_id' in position_data and position_data.get('asset_id') is not None:
        position.asset_id = position_data.get('asset_id')
    if 'owner_id' in position_data:
        position.owner_id = position_data.get('owner_id')
    if 'as_of_date' in position_data and position_data.get('as_of_date') is not None:
        position.as_of_date = position_data.get('as_of_date')
    if 'quantity' in position_data and position_data.get('quantity') is not None:
        position.quantity = position_data.get('quantity')
    if 'value' in position_data and position_data.get('value') is not None:
        position.value = position_data.get('value')
    if 'broker' in position_data:
        position.broker = position_data.get('broker')
    if 'source' in position_data and position_data.get('source') is not None:
        position.source = position_data.get('source')

    db.add(position)
    if commit:
        db.commit()
        db.refresh(position)
    else:
        db.flush()
    return position


def delete_position(db: Session, position: models_module.Position, commit: bool = True):
    db.delete(position)
    if commit:
        db.commit()
    else:
        db.flush()


def delete_positions_for_date_except(db: Session, as_of_date, keep_position_ids=None, commit: bool = True):
    keep_position_ids = set(keep_position_ids or [])
    positions = db.query(models_module.Position).filter(
        models_module.Position.as_of_date == as_of_date,
    ).all()
    deleted = 0
    for position in positions:
        if position.id in keep_position_ids:
            continue
        db.delete(position)
        deleted += 1
    if commit:
        db.commit()
    else:
        db.flush()
    return deleted


def _resolve_positions_for_date(db: Session, as_of_date=None, asset_id: int = None):
    query = db.query(models_module.Position)
    if asset_id is not None:
        query = query.filter(models_module.Position.asset_id == asset_id)

    if as_of_date is None:
        return query.order_by(models_module.Position.as_of_date.desc(), models_module.Position.asset_id).all()

    return query.filter(models_module.Position.as_of_date == as_of_date).order_by(
        models_module.Position.asset_id,
        models_module.Position.as_of_date,
    ).all()


def get_available_dates(db: Session):
    return [
        row[0]
        for row in db.query(models_module.Position.as_of_date)
        .distinct()
        .order_by(models_module.Position.as_of_date)
        .all()
    ]


def get_positions(db: Session, as_of_date=None, asset_id: int = None, skip: int = 0, limit: int = 100):
    positions = _resolve_positions_for_date(db, as_of_date=as_of_date, asset_id=asset_id)
    return positions[skip:skip + limit]


def get_position_snapshot_rows(db: Session, as_of_date=None):
    effective_date = as_of_date
    if effective_date is None:
        latest_position = db.query(models_module.Position).order_by(models_module.Position.as_of_date.desc()).first()
        effective_date = latest_position.as_of_date if latest_position is not None else None

    positions = _resolve_positions_for_date(db, as_of_date=effective_date)
    rows = []
    for position in positions:
        asset = position.asset
        if asset is None:
            continue
        rows.append({
            'position_id': position.id,
            'asset_id': asset.id,
            'owner_id': position.owner_id,
            'asset_name': asset.name,
            'owner_name': position.owner.name if position.owner else None,
            'category': asset.category,
            'asset_type': asset.asset_type,
            'valuation_method': asset.valuation_method,
            'is_investment': asset_is_investment(asset),
            'broker': position.broker,
            'quantity': round(float(position.quantity or 0.0), 2),
            'value': round(float(position.value or 0.0), 2),
            'ownership_shares': get_position_ownership(db, position),
        })

    return sorted(rows, key=lambda item: (item['asset_name'], item['owner_name'] or '', item['broker'] or ''))


def get_dashboard_details(db: Session, as_of_date=None):
    effective_date = as_of_date
    if effective_date is None:
        latest_position = db.query(models_module.Position).order_by(models_module.Position.as_of_date.desc()).first()
        effective_date = latest_position.as_of_date if latest_position is not None else None

    positions = _resolve_positions_for_date(db, as_of_date=effective_date)
    rows = []

    for position in positions:
        asset = position.asset
        if asset is None:
            continue

        shares = position_family_attribution_shares(db, position, effective_date)
        if shares:
            for owner_id, share in shares.items():
                owner = get_owner_by_id(db, owner_id)
                if owner is None:
                    continue
                rows.append({
                    'position_id': position.id,
                    'asset_id': asset.id,
                    'owner_id': owner_id,
                    'asset_name': asset.name,
                    'owner_name': owner.name,
                    'category': asset.category,
                    'asset_type': asset.asset_type,
                    'valuation_method': asset.valuation_method,
                    'is_investment': asset_is_investment(asset),
                    'broker': position.broker,
                    'quantity': round(float(position.quantity or 0.0) * share, 2),
                    'value': round(float(position.value or 0.0) * share, 2),
                })
        elif position.owner is None and not (position.ownerships or default_family_ownership_shares(db)):
            rows.append({
                'position_id': position.id,
                'asset_id': asset.id,
                'owner_id': position.owner_id,
                'asset_name': asset.name,
                'owner_name': 'Unassigned',
                'category': asset.category,
                'asset_type': asset.asset_type,
                'valuation_method': asset.valuation_method,
                'is_investment': asset_is_investment(asset),
                'broker': position.broker,
                'quantity': round(float(position.quantity or 0.0), 2),
                'value': round(float(position.value or 0.0), 2),
            })

    return sorted(rows, key=lambda item: (item['owner_name'], item['asset_name']))


def get_dashboard_summary(db: Session, as_of_date=None):
    latest_position = db.query(models_module.Position).order_by(models_module.Position.as_of_date.desc()).first()
    if latest_position is None:
        return {
            'as_of_date': None,
            'total_value': 0.0,
            'position_count': 0,
            'by_asset': [],
            'by_category': [],
            'by_broker': [],
            'by_owner': [],
        }

    if as_of_date is None:
        effective_date = latest_position.as_of_date
    else:
        effective_date = as_of_date

    positions = _resolve_positions_for_date(db, as_of_date=effective_date)
    details = get_dashboard_details(db, as_of_date=effective_date)
    asset_totals = {}
    category_totals = {}
    broker_totals = {}
    owner_totals = {owner.name: 0.0 for owner in list_owners(db)}
    total_value = 0.0

    for detail in details:
        value = float(detail['value'] or 0.0)
        asset_name = detail['asset_name']
        category_name = detail['category'] or 'Uncategorized'
        broker_name = detail['broker'] or '(blank)'
        asset_totals[asset_name] = asset_totals.get(asset_name, 0.0) + value
        category_totals[category_name] = category_totals.get(category_name, 0.0) + value
        broker_totals[broker_name] = broker_totals.get(broker_name, 0.0) + value
        owner_totals[detail['owner_name']] = owner_totals.get(detail['owner_name'], 0.0) + value
        total_value += value

    by_asset = [
        {
            'asset_name': name,
            'category': next((asset.category for asset in [p.asset for p in positions if p.asset and p.asset.name == name]), None),
            'value': round(value, 2),
        }
        for name, value in sorted(asset_totals.items(), key=lambda item: item[1], reverse=True)
    ]
    by_owner = [
        {
            'owner_name': name,
            'value': round(value, 2),
        }
        for name, value in sorted(owner_totals.items(), key=lambda item: item[1], reverse=True)
    ]
    by_category = [
        {
            'category': name,
            'value': round(value, 2),
        }
        for name, value in sorted(category_totals.items(), key=lambda item: item[1], reverse=True)
    ]
    by_broker = [
        {
            'broker': name,
            'value': round(value, 2),
        }
        for name, value in sorted(broker_totals.items(), key=lambda item: item[1], reverse=True)
    ]

    return {
        'as_of_date': effective_date,
        'total_value': round(total_value, 2),
        'position_count': len(positions),
        'by_asset': by_asset,
        'by_category': by_category,
        'by_broker': by_broker,
        'by_owner': by_owner,
    }


def get_dashboard_history(db: Session):
    history = []
    for as_of_date in get_available_dates(db):
        history.append({
            'date': as_of_date.date().isoformat() if hasattr(as_of_date, 'date') else str(as_of_date),
            'summary': get_dashboard_summary(db, as_of_date=as_of_date),
            'details': get_dashboard_details(db, as_of_date=as_of_date),
        })
    return history


def set_ownership(db: Session, asset: models_module.Asset, owner: models_module.Owner, share: float = 1.0):
    own = db.query(models_module.Ownership).filter(models_module.Ownership.asset_id == asset.id, models_module.Ownership.owner_id == owner.id).first()
    if own:
        own.share = share
        db.add(own)
        db.commit()
        db.refresh(own)
        return own
    own = models_module.Ownership(owner_id=owner.id, asset_id=asset.id, share=share)
    db.add(own)
    db.commit()
    db.refresh(own)
    return own


def get_asset_ownership(db: Session, asset: models_module.Asset):
    return [
        {
            'owner_id': ownership.owner_id,
            'owner_name': ownership.owner.name if ownership.owner else None,
            'share': float(ownership.share or 0.0),
        }
        for ownership in sorted(asset.ownerships, key=lambda item: item.owner.name if item.owner else '')
    ]


def list_asset_ownership(db: Session):
    return [
        {
            'asset_id': asset.id,
            'shares': get_asset_ownership(db, asset),
        }
        for asset in list_assets(db, limit=10000)
    ]


def ensure_default_family_ownership(db: Session, asset: models_module.Asset):
    if asset.ownerships:
        return get_asset_ownership(db, asset)

    antonio = get_owner_by_name(db, 'Antonio')
    patri = get_owner_by_name(db, 'Patri') or get_owner_by_name(db, 'Patricia')
    if antonio is None or patri is None:
        return get_asset_ownership(db, asset)

    set_ownership(db, asset, antonio, share=0.5)
    set_ownership(db, asset, patri, share=0.5)
    db.refresh(asset)
    return get_asset_ownership(db, asset)


def replace_asset_ownership(db: Session, asset: models_module.Asset, shares: list):
    db.query(models_module.Ownership).filter(models_module.Ownership.asset_id == asset.id).delete()
    db.flush()

    for share in shares:
        owner = get_owner_by_id(db, share['owner_id'])
        if owner is None:
            continue
        value = float(share.get('share') or 0.0)
        if value <= 0:
            continue
        db.add(models_module.Ownership(owner_id=owner.id, asset_id=asset.id, share=value))

    db.commit()
    db.refresh(asset)
    return get_asset_ownership(db, asset)


def list_assets(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models_module.Asset).offset(skip).limit(limit).all()
