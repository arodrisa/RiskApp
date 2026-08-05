"""projects, application users, and entity ownership

Revision ID: 0005_projects_users_and_entity_ownerships
Revises: 0004_price_history
Create Date: 2026-08-02
"""

from alembic import op
import sqlalchemy as sa


revision = '0005_projects_users_and_entity_ownerships'
down_revision = '0004_price_history'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'projects',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('base_currency', sa.String(), nullable=False, server_default='EUR'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
    )
    op.create_index(op.f('ix_projects_id'), 'projects', ['id'], unique=False)

    op.add_column('owners', sa.Column('project_id', sa.Integer(), nullable=True))
    op.add_column('owners', sa.Column('is_family_member', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('owners', sa.Column('archived_at', sa.DateTime(), nullable=True))
    op.create_index(op.f('ix_owners_project_id'), 'owners', ['project_id'], unique=False)
    if op.get_bind().dialect.name != 'sqlite':
        op.create_foreign_key('fk_owners_project_id_projects', 'owners', 'projects', ['project_id'], ['id'])

    op.add_column('assets', sa.Column('project_id', sa.Integer(), nullable=True))
    op.create_index(op.f('ix_assets_project_id'), 'assets', ['project_id'], unique=False)
    if op.get_bind().dialect.name != 'sqlite':
        op.create_foreign_key('fk_assets_project_id_projects', 'assets', 'projects', ['project_id'], ['id'])

    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('display_name', sa.String(), nullable=False),
        sa.Column('password_hash', sa.String(), nullable=False),
        sa.Column('person_owner_id', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('session_version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('last_login_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['person_owner_id'], ['owners.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_person_owner_id'), 'users', ['person_owner_id'], unique=False)

    op.create_table(
        'project_memberships',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(), nullable=False, server_default='viewer'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('project_id', 'user_id', name='uq_project_memberships_project_user'),
    )
    op.create_index(op.f('ix_project_memberships_id'), 'project_memberships', ['id'], unique=False)
    op.create_index(op.f('ix_project_memberships_project_id'), 'project_memberships', ['project_id'], unique=False)
    op.create_index(op.f('ix_project_memberships_user_id'), 'project_memberships', ['user_id'], unique=False)

    op.create_table(
        'project_invitations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=False, server_default='viewer'),
        sa.Column('token_hash', sa.String(), nullable=False),
        sa.Column('invited_by_user_id', sa.Integer(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('accepted_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['invited_by_user_id'], ['users.id']),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token_hash'),
    )
    op.create_index(op.f('ix_project_invitations_email'), 'project_invitations', ['email'], unique=False)
    op.create_index(op.f('ix_project_invitations_expires_at'), 'project_invitations', ['expires_at'], unique=False)
    op.create_index(op.f('ix_project_invitations_id'), 'project_invitations', ['id'], unique=False)
    op.create_index(op.f('ix_project_invitations_invited_by_user_id'), 'project_invitations', ['invited_by_user_id'], unique=False)
    op.create_index(op.f('ix_project_invitations_project_id'), 'project_invitations', ['project_id'], unique=False)
    op.create_index(op.f('ix_project_invitations_token_hash'), 'project_invitations', ['token_hash'], unique=False)

    op.create_table(
        'entity_ownerships',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('owned_id', sa.Integer(), nullable=False),
        sa.Column('share', sa.Float(), nullable=False),
        sa.Column('effective_from', sa.DateTime(), nullable=False),
        sa.Column('effective_to', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['owned_id'], ['owners.id']),
        sa.ForeignKeyConstraint(['owner_id'], ['owners.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_entity_ownerships_effective_from'), 'entity_ownerships', ['effective_from'], unique=False)
    op.create_index(op.f('ix_entity_ownerships_effective_to'), 'entity_ownerships', ['effective_to'], unique=False)
    op.create_index(op.f('ix_entity_ownerships_id'), 'entity_ownerships', ['id'], unique=False)
    op.create_index(op.f('ix_entity_ownerships_owned_id'), 'entity_ownerships', ['owned_id'], unique=False)
    op.create_index(op.f('ix_entity_ownerships_owner_id'), 'entity_ownerships', ['owner_id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_entity_ownerships_owner_id'), table_name='entity_ownerships')
    op.drop_index(op.f('ix_entity_ownerships_owned_id'), table_name='entity_ownerships')
    op.drop_index(op.f('ix_entity_ownerships_id'), table_name='entity_ownerships')
    op.drop_index(op.f('ix_entity_ownerships_effective_to'), table_name='entity_ownerships')
    op.drop_index(op.f('ix_entity_ownerships_effective_from'), table_name='entity_ownerships')
    op.drop_table('entity_ownerships')
    op.drop_table('project_invitations')
    op.drop_table('project_memberships')
    op.drop_table('users')
    if op.get_bind().dialect.name != 'sqlite':
        op.drop_constraint('fk_assets_project_id_projects', 'assets', type_='foreignkey')
    op.drop_index(op.f('ix_assets_project_id'), table_name='assets')
    op.drop_column('assets', 'project_id')
    if op.get_bind().dialect.name != 'sqlite':
        op.drop_constraint('fk_owners_project_id_projects', 'owners', type_='foreignkey')
    op.drop_index(op.f('ix_owners_project_id'), table_name='owners')
    op.drop_column('owners', 'archived_at')
    op.drop_column('owners', 'is_family_member')
    op.drop_column('owners', 'project_id')
    op.drop_index(op.f('ix_projects_id'), table_name='projects')
    op.drop_table('projects')
