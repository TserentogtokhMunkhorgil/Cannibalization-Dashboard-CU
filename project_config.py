"""Portable roots and explicit local extraction settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def store_metadata_path():
    return Path(os.environ.get('CU_STORE_METADATA', str(ROOT / 'Data' / 'raw' / 'StoreMaster.xlsx')))


def clickhouse_settings():
    password = os.environ.get('CU_CLICKHOUSE_PASSWORD')
    if not password and os.environ.get('CU_CLICKHOUSE_PASSWORD_FILE'):
        password = Path(os.environ['CU_CLICKHOUSE_PASSWORD_FILE']).read_text().strip()
    if not password:
        raise ValueError('Set CU_CLICKHOUSE_PASSWORD or CU_CLICKHOUSE_PASSWORD_FILE before extraction.')
    return dict(host=os.environ.get('CU_CLICKHOUSE_HOST', 'localhost'),
                port=int(os.environ.get('CU_CLICKHOUSE_PORT', '8123')),
                username=os.environ.get('CU_CLICKHOUSE_USER', 'default'),
                database=os.environ.get('CU_CLICKHOUSE_DATABASE', 'default'), password=password)
