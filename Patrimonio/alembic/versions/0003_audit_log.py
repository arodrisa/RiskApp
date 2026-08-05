"""audit log

Revision ID: 0003_audit_log
Revises: 0002_beta_schema_updates
Create Date: 2026-07-18
"""

from alembic import op
import sqlalchemy as sa


revision = '0003_audit_log'
down_revision = '0002_beta_schema_updates'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'audit_log',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('actor', sa.String(), nullable=True),
        sa.Column('action', sa.String(), nullable=False),
        sa.Column('entity_type', sa.String(), nullable=True),
        sa.Column('entity_id', sa.String(), nullable=True),
        sa.Column('details', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_audit_log_action'), 'audit_log', ['action'], unique=False)
    op.create_index(op.f('ix_audit_log_actor'), 'audit_log', ['actor'], unique=False)
    op.create_index(op.f('ix_audit_log_created_at'), 'audit_log', ['created_at'], unique=False)
    op.create_index(op.f('ix_audit_log_entity_id'), 'audit_log', ['entity_id'], unique=False)
    op.create_index(op.f('ix_audit_log_entity_type'), 'audit_log', ['entity_type'], unique=False)
    op.create_index(op.f('ix_audit_log_id'), 'audit_log', ['id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_audit_log_id'), table_name='audit_log')
    op.drop_index(op.f('ix_audit_log_entity_type'), table_name='audit_log')
    op.drop_index(op.f('ix_audit_log_entity_id'), table_name='audit_log')
    op.drop_index(op.f('ix_audit_log_created_at'), table_name='audit_log')
    op.drop_index(op.f('ix_audit_log_actor'), table_name='audit_log')
    op.drop_index(op.f('ix_audit_log_action'), table_name='audit_log')
    op.drop_table('audit_log')
