"""beta schema updates

Revision ID: 0002_beta_schema_updates
Revises: 0001_initial_schema
Create Date: 2026-07-18
"""

from alembic import op
import sqlalchemy as sa


revision = '0002_beta_schema_updates'
down_revision = '0001_initial_schema'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('assets', sa.Column('price_provider', sa.String(), nullable=True))
    op.add_column('assets', sa.Column('price_symbol', sa.String(), nullable=True))
    op.add_column('assets', sa.Column('is_investment', sa.Boolean(), nullable=True))
    op.create_index(op.f('ix_assets_price_symbol'), 'assets', ['price_symbol'], unique=False)

    op.execute("UPDATE assets SET price_provider = 'manual' WHERE price_provider IS NULL")
    op.execute("""
        UPDATE assets
        SET is_investment = CASE
            WHEN lower(coalesce(asset_type, '')) = 'cash' THEN 0
            WHEN lower(coalesce(category, '')) IN ('cash', 'caja', 'efectivo') THEN 0
            WHEN lower(coalesce(category, '')) = 'casa' THEN 0
            WHEN lower(coalesce(name, '')) = 'casa' THEN 0
            ELSE 1
        END
        WHERE is_investment IS NULL
    """)

    op.create_table(
        'position_ownership',
        sa.Column('position_id', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('share', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['owner_id'], ['owners.id']),
        sa.ForeignKeyConstraint(['position_id'], ['positions.id']),
        sa.PrimaryKeyConstraint('position_id', 'owner_id'),
    )

    op.create_table(
        'company_valuation_items',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('asset_id', sa.Integer(), nullable=True),
        sa.Column('as_of_date', sa.DateTime(), nullable=True),
        sa.Column('item_type', sa.String(), nullable=True),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('amount', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['asset_id'], ['assets.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_company_valuation_items_asset_id'), 'company_valuation_items', ['asset_id'], unique=False)
    op.create_index(op.f('ix_company_valuation_items_as_of_date'), 'company_valuation_items', ['as_of_date'], unique=False)
    op.create_index(op.f('ix_company_valuation_items_id'), 'company_valuation_items', ['id'], unique=False)
    op.create_index(op.f('ix_company_valuation_items_item_type'), 'company_valuation_items', ['item_type'], unique=False)

    op.create_table(
        'investing_assets',
        sa.Column('category', sa.String(), nullable=False),
        sa.Column('is_invested', sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint('category'),
    )
    op.create_index(op.f('ix_investing_assets_category'), 'investing_assets', ['category'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_investing_assets_category'), table_name='investing_assets')
    op.drop_table('investing_assets')

    op.drop_index(op.f('ix_company_valuation_items_item_type'), table_name='company_valuation_items')
    op.drop_index(op.f('ix_company_valuation_items_id'), table_name='company_valuation_items')
    op.drop_index(op.f('ix_company_valuation_items_as_of_date'), table_name='company_valuation_items')
    op.drop_index(op.f('ix_company_valuation_items_asset_id'), table_name='company_valuation_items')
    op.drop_table('company_valuation_items')

    op.drop_table('position_ownership')

    op.drop_index(op.f('ix_assets_price_symbol'), table_name='assets')
    op.drop_column('assets', 'is_investment')
    op.drop_column('assets', 'price_symbol')
    op.drop_column('assets', 'price_provider')
