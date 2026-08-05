import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./patrimonio.db')

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith('sqlite') else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def ensure_beta_schema(engine):
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if 'assets' not in table_names:
        return

    asset_columns = {column['name'] for column in inspector.get_columns('assets')}
    missing_columns = []
    if 'price_provider' not in asset_columns:
        missing_columns.append(('price_provider', 'VARCHAR'))
    if 'price_symbol' not in asset_columns:
        missing_columns.append(('price_symbol', 'VARCHAR'))
    if 'is_investment' not in asset_columns:
        missing_columns.append(('is_investment', 'BOOLEAN'))

    with engine.begin() as connection:
        for column_name, column_type in missing_columns:
            connection.execute(text(f'ALTER TABLE assets ADD COLUMN {column_name} {column_type}'))
        connection.execute(text("UPDATE assets SET price_provider = 'manual' WHERE price_provider IS NULL"))
        connection.execute(text("""
            UPDATE assets
            SET is_investment = CASE
                WHEN lower(coalesce(asset_type, '')) = 'cash' THEN 0
                WHEN lower(coalesce(category, '')) IN ('cash', 'caja', 'efectivo') THEN 0
                WHEN lower(coalesce(category, '')) = 'casa' THEN 0
                WHEN lower(coalesce(name, '')) = 'casa' THEN 0
                ELSE 1
            END
            WHERE is_investment IS NULL
        """))


def ensure_application_schema(engine):
    """Keep existing development SQLite databases usable before Alembic is enforced."""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if 'owners' not in table_names or 'assets' not in table_names:
        return

    expected_columns = {
        'owners': {
            'project_id': 'INTEGER',
            'is_family_member': 'BOOLEAN DEFAULT 0',
            'archived_at': 'DATETIME',
        },
        'assets': {'project_id': 'INTEGER'},
    }
    with engine.begin() as connection:
        for table_name, columns in expected_columns.items():
            existing = {column['name'] for column in inspector.get_columns(table_name)}
            for column_name, column_type in columns.items():
                if column_name not in existing:
                    connection.execute(text(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}'))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
