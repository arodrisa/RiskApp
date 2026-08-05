"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-07-14
"""

from alembic import op
import sqlalchemy as sa


revision = '0001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'owners',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_owners_id'), 'owners', ['id'], unique=False)
    op.create_index(op.f('ix_owners_name'), 'owners', ['name'], unique=True)

    op.create_table(
        'assets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('category', sa.String(), nullable=True),
        sa.Column('asset_type', sa.String(), nullable=True),
        sa.Column('valuation_method', sa.String(), nullable=True),
        sa.Column('is_shared', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_assets_asset_type'), 'assets', ['asset_type'], unique=False)
    op.create_index(op.f('ix_assets_category'), 'assets', ['category'], unique=False)
    op.create_index(op.f('ix_assets_id'), 'assets', ['id'], unique=False)
    op.create_index(op.f('ix_assets_name'), 'assets', ['name'], unique=False)

    op.create_table(
        'asset_valuations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('asset_id', sa.Integer(), nullable=True),
        sa.Column('as_of_date', sa.DateTime(), nullable=True),
        sa.Column('value', sa.Float(), nullable=True),
        sa.Column('source', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['asset_id'], ['assets.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_asset_valuations_as_of_date'), 'asset_valuations', ['as_of_date'], unique=False)
    op.create_index(op.f('ix_asset_valuations_id'), 'asset_valuations', ['id'], unique=False)

    op.create_table(
        'ownership',
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('asset_id', sa.Integer(), nullable=False),
        sa.Column('share', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['asset_id'], ['assets.id']),
        sa.ForeignKeyConstraint(['owner_id'], ['owners.id']),
        sa.PrimaryKeyConstraint('owner_id', 'asset_id'),
    )

    op.create_table(
        'positions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('asset_id', sa.Integer(), nullable=True),
        sa.Column('owner_id', sa.Integer(), nullable=True),
        sa.Column('as_of_date', sa.DateTime(), nullable=True),
        sa.Column('quantity', sa.Float(), nullable=True),
        sa.Column('value', sa.Float(), nullable=True),
        sa.Column('broker', sa.String(), nullable=True),
        sa.Column('source', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['asset_id'], ['assets.id']),
        sa.ForeignKeyConstraint(['owner_id'], ['owners.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_positions_as_of_date'), 'positions', ['as_of_date'], unique=False)
    op.create_index(op.f('ix_positions_broker'), 'positions', ['broker'], unique=False)
    op.create_index(op.f('ix_positions_id'), 'positions', ['id'], unique=False)
    op.create_index(op.f('ix_positions_owner_id'), 'positions', ['owner_id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_positions_owner_id'), table_name='positions')
    op.drop_index(op.f('ix_positions_id'), table_name='positions')
    op.drop_index(op.f('ix_positions_broker'), table_name='positions')
    op.drop_index(op.f('ix_positions_as_of_date'), table_name='positions')
    op.drop_table('positions')
    op.drop_table('ownership')
    op.drop_index(op.f('ix_asset_valuations_id'), table_name='asset_valuations')
    op.drop_index(op.f('ix_asset_valuations_as_of_date'), table_name='asset_valuations')
    op.drop_table('asset_valuations')
    op.drop_index(op.f('ix_assets_name'), table_name='assets')
    op.drop_index(op.f('ix_assets_id'), table_name='assets')
    op.drop_index(op.f('ix_assets_category'), table_name='assets')
    op.drop_index(op.f('ix_assets_asset_type'), table_name='assets')
    op.drop_table('assets')
    op.drop_index(op.f('ix_owners_name'), table_name='owners')
    op.drop_index(op.f('ix_owners_id'), table_name='owners')
    op.drop_table('owners')
