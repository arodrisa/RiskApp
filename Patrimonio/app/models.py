from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()


class Project(Base):
    __tablename__ = 'projects'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    base_currency = Column(String, nullable=False, default='EUR')
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    owners = relationship('Owner', back_populates='project')
    assets = relationship('Asset', back_populates='project')
    memberships = relationship('ProjectMembership', back_populates='project', cascade='all, delete-orphan')
    invitations = relationship('ProjectInvitation', back_populates='project', cascade='all, delete-orphan')


class Owner(Base):
    __tablename__ = 'owners'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    type = Column(String, default='person')
    project_id = Column(Integer, ForeignKey('projects.id'), index=True)
    is_family_member = Column(Boolean, nullable=False, default=False)
    archived_at = Column(DateTime, nullable=True)

    project = relationship('Project', back_populates='owners')
    user_accounts = relationship('User', back_populates='person_owner')
    companies_owned = relationship(
        'EntityOwnership',
        foreign_keys='EntityOwnership.owner_id',
        back_populates='owner',
        cascade='all, delete-orphan',
    )
    owners_of_company = relationship(
        'EntityOwnership',
        foreign_keys='EntityOwnership.owned_id',
        back_populates='owned',
        cascade='all, delete-orphan',
    )

    # Legacy asset-level ownership is kept only to migrate old databases.
    ownerships = relationship('Ownership', back_populates='owner', cascade='all, delete-orphan')
    positions = relationship('Position', back_populates='owner')
    position_ownerships = relationship('PositionOwnership', back_populates='owner', cascade='all, delete-orphan')


class Asset(Base):
    __tablename__ = 'assets'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    category = Column(String, index=True)
    asset_type = Column(String, index=True)
    valuation_method = Column(String, default='market_direct')
    price_provider = Column(String, default='manual')
    price_symbol = Column(String, index=True)
    is_investment = Column(Boolean, nullable=True)
    is_shared = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    project_id = Column(Integer, ForeignKey('projects.id'), index=True)

    project = relationship('Project', back_populates='assets')

    # Legacy asset-level ownership is kept only to migrate old databases.
    ownerships = relationship('Ownership', back_populates='asset', cascade='all, delete-orphan')
    valuations = relationship('AssetValuation', back_populates='asset', cascade='all, delete-orphan')
    company_valuation_items = relationship('CompanyValuationItem', back_populates='asset', cascade='all, delete-orphan')
    positions = relationship('Position', back_populates='asset', cascade='all, delete-orphan')


class InvestingAsset(Base):
    __tablename__ = 'investing_assets'
    category = Column(String, primary_key=True, index=True)
    is_invested = Column(Boolean, default=True, nullable=False)


class AuditLog(Base):
    __tablename__ = 'audit_log'
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    actor = Column(String, default='api', index=True)
    action = Column(String, nullable=False, index=True)
    entity_type = Column(String, index=True)
    entity_id = Column(String, index=True)
    details = Column(Text)


class AssetValuation(Base):
    __tablename__ = 'asset_valuations'
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey('assets.id'))
    as_of_date = Column(DateTime, index=True)
    value = Column(Float, default=0.0)
    source = Column(String, default='import')
    created_at = Column(DateTime, default=datetime.utcnow)

    asset = relationship('Asset', back_populates='valuations')


class CompanyValuationItem(Base):
    __tablename__ = 'company_valuation_items'
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey('assets.id'), index=True)
    as_of_date = Column(DateTime, index=True)
    item_type = Column(String, index=True)  # asset or liability
    name = Column(String, nullable=False)
    amount = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    asset = relationship('Asset', back_populates='company_valuation_items')


class PriceHistory(Base):
    __tablename__ = 'price_history'
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey('assets.id'), index=True)
    provider = Column(String, index=True)
    symbol = Column(String, index=True)
    price = Column(Float, default=0.0)
    currency = Column(String)
    as_of = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    asset = relationship('Asset')


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False, unique=True, index=True)
    display_name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    person_owner_id = Column(Integer, ForeignKey('owners.id'), nullable=True, index=True)
    is_active = Column(Boolean, nullable=False, default=True)
    session_version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime, nullable=True)

    person_owner = relationship('Owner', back_populates='user_accounts')
    memberships = relationship('ProjectMembership', back_populates='user', cascade='all, delete-orphan')
    sent_invitations = relationship('ProjectInvitation', back_populates='invited_by')
    password_reset_tokens = relationship('PasswordResetToken', back_populates='user', cascade='all, delete-orphan')


class ProjectMembership(Base):
    __tablename__ = 'project_memberships'
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    role = Column(String, nullable=False, default='viewer')
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship('Project', back_populates='memberships')
    user = relationship('User', back_populates='memberships')
    __table_args__ = (UniqueConstraint('project_id', 'user_id', name='uq_project_memberships_project_user'),)


class ProjectInvitation(Base):
    __tablename__ = 'project_invitations'
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False, index=True)
    email = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False, default='viewer')
    token_hash = Column(String, nullable=False, unique=True, index=True)
    invited_by_user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    accepted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship('Project', back_populates='invitations')
    invited_by = relationship('User', back_populates='sent_invitations')


class PasswordResetToken(Base):
    __tablename__ = 'password_reset_tokens'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    token_hash = Column(String, nullable=False, unique=True, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship('User', back_populates='password_reset_tokens')


class EntityOwnership(Base):
    """A dated legal ownership edge: one entity owns part of another entity."""

    __tablename__ = 'entity_ownerships'
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey('owners.id'), nullable=False, index=True)
    owned_id = Column(Integer, ForeignKey('owners.id'), nullable=False, index=True)
    share = Column(Float, nullable=False)
    effective_from = Column(DateTime, nullable=False, index=True)
    effective_to = Column(DateTime, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    owner = relationship('Owner', foreign_keys=[owner_id], back_populates='companies_owned')
    owned = relationship('Owner', foreign_keys=[owned_id], back_populates='owners_of_company')


class Position(Base):
    __tablename__ = 'positions'
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey('assets.id'))
    owner_id = Column(Integer, ForeignKey('owners.id'), nullable=True, index=True)
    as_of_date = Column(DateTime, index=True)
    quantity = Column(Float, default=0.0)
    value = Column(Float, default=0.0)
    broker = Column(String, index=True)
    source = Column(String, default='import')
    created_at = Column(DateTime, default=datetime.utcnow)

    asset = relationship('Asset', back_populates='positions')
    owner = relationship('Owner', back_populates='positions')
    ownerships = relationship('PositionOwnership', back_populates='position', cascade='all, delete-orphan')


class PositionOwnership(Base):
    __tablename__ = 'position_ownership'
    position_id = Column(Integer, ForeignKey('positions.id'), primary_key=True)
    owner_id = Column(Integer, ForeignKey('owners.id'), primary_key=True)
    share = Column(Float, default=1.0)

    position = relationship('Position', back_populates='ownerships')
    owner = relationship('Owner', back_populates='position_ownerships')


class Ownership(Base):
    """Legacy asset-level ownership.

    New data must store ownership on PositionOwnership so the same asset can
    appear in different accounts or dates with different Antonio/Patri splits.
    """

    __tablename__ = 'ownership'
    owner_id = Column(Integer, ForeignKey('owners.id'), primary_key=True)
    asset_id = Column(Integer, ForeignKey('assets.id'), primary_key=True)
    share = Column(Float, default=1.0)

    owner = relationship('Owner', back_populates='ownerships')
    asset = relationship('Asset', back_populates='ownerships')
