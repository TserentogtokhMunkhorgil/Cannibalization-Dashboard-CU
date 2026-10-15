
"""
CU Cannibalization Analysis
Project Configuration and ClickHouse Connection
"""

import os
from pathlib import Path



# ==========================================================
# 1. PROJECT ROOT
# ==========================================================

ROOT = Path(__file__).resolve().parent


# ==========================================================
# 2. STORE METADATA PATH
# ==========================================================

# Default location of StoreMaster.xlsx
STORE_METADATA_FILE = (
    Path.home()
    / "OneDrive - Premium Corporation"
    / "Power BI Data - Power BI Data"
    / "StoreMaster.xlsx"
)


def store_metadata_path() -> Path:
    """Return the StoreMaster Excel file path."""

    configured_path = os.environ.get("CU_STORE_METADATA")

    if configured_path:
        path = Path(configured_path).expanduser()
    else:
        path = STORE_METADATA_FILE

    if not path.is_file():
        raise FileNotFoundError(
            f"StoreMaster Excel file not found: {path}"
        )

    return path


# ==========================================================
# 3. CLICKHOUSE SERVER SETTINGS
# ==========================================================

CLICKHOUSE_HOST = "10.10.90.134"
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = "sales_analyst1"
CLICKHOUSE_DATABASE = "BI_DB"


# ==========================================================
# 4. PASSWORD FILE LOCATION
# ==========================================================

PASSWORD_FILE = Path(
    r"C:\MO files\CU DATALAKE PASSWORD\Sales1_password.txt"
)


# ==========================================================
# 5. LOAD PASSWORD
# ==========================================================

def load_clickhouse_password() -> str:
    """Read ClickHouse password from local text file."""

    password_path = Path(
        os.environ.get(
            "CU_CLICKHOUSE_PASSWORD_FILE",
            str(PASSWORD_FILE)
        )
    ).expanduser()

    if not password_path.is_file():
        raise FileNotFoundError(
            f"ClickHouse password file not found: {password_path}"
        )

    password = password_path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not password:
        raise ValueError(
            "ClickHouse password file is empty."
        )

    return password


# ==========================================================
# 6. CLICKHOUSE CONNECTION SETTINGS
# ==========================================================

def clickhouse_settings() -> dict:
    """Return ClickHouse connection parameters."""

    return {
        "host": CLICKHOUSE_HOST,
        "port": CLICKHOUSE_PORT,
        "username": CLICKHOUSE_USER,
        "password": load_clickhouse_password(),
        "database": CLICKHOUSE_DATABASE,
        "secure": False,
        "connect_timeout": 15,
        "send_receive_timeout": 120,
    }


# ==========================================================
# 7. CREATE CLICKHOUSE CLIENT
# ==========================================================

def get_clickhouse_client():
    """Create and return a ClickHouse client."""

    import clickhouse_connect

    return clickhouse_connect.get_client(
        **clickhouse_settings()
    )


# ==========================================================
# 8. TEST CONNECTION
# ==========================================================

def test_connection():
    """Test the ClickHouse database connection."""

    print("=" * 55)
    print("CU CLICKHOUSE CONNECTION TEST")
    print("=" * 55)

    print(f"Host     : {CLICKHOUSE_HOST}")
    print(f"Port     : {CLICKHOUSE_PORT}")
    print(f"User     : {CLICKHOUSE_USER}")
    print(f"Database : {CLICKHOUSE_DATABASE}")

    client = None

    try:
        client = get_clickhouse_client()

        version = client.command("SELECT version()")

        print("\nClickHouse connection successful!")
        print(f"ClickHouse version: {version}")

        return version

    except Exception as error:
        print("\nClickHouse connection failed!")
        print(f"Error: {error}")
        raise

    finally:
        if client is not None:
            client.close()


# ==========================================================
# 9. MAIN
# ==========================================================

if __name__ == "__main__":
    test_connection()
