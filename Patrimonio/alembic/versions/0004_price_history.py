"""price history

Revision ID: 0004_price_history
Revises: 0003_audit_log
Create Date: 2026-07-18
"""

from alembic import op
import sqlalchemy as sa


revision = '0004_price_history'
down_revision = '0003_audit_log'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'price_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('asset_id', sa.Integer(), nullable=True),
        sa.Column('provider', sa.String(), nullable=True),
        sa.Column('symbol', sa.String(), nullable=True),
        sa.Column('price', sa.Float(), nullable=True),
        sa.Column('currency', sa.String(), nullable=True),
        sa.Column('as_of', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['asset_id'], ['assets.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_price_history_asset_id'), 'price_history', ['asset_id'], unique=False)
    op.create_index(op.f('ix_price_history_created_at'), 'price_history', ['created_at'], unique=False)
    op.create_index(op.f('ix_price_history_id'), 'price_history', ['id'], unique=False)
    op.create_index(op.f('ix_price_history_provider'), 'price_history', ['provider'], unique=False)
    op.create_index(op.f('ix_price_history_symbol'), 'price_history', ['symbol'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_price_history_symbol'), table_name='price_history')
    op.drop_index(op.f('ix_price_history_provider'), table_name='price_history')
    op.drop_index(op.f('ix_price_history_id'), table_name='price_history')
    op.drop_index(op.f('ix_price_history_created_at'), table_name='price_history')
    op.drop_index(op.f('ix_price_history_asset_id'), table_name='price_history')
    op.drop_table('price_history')
