"""
Database connection test script for Azure PostgreSQL.

Tests both async (asyncpg) and sync (psycopg2) connections.

Usage:
    python test_db_connection.py
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from ..backend.config.settings import settings


def test_sync_connection():
    """Test synchronous PostgreSQL connection using psycopg2."""
    print("\n" + "="*60)
    print("Testing SYNC PostgreSQL Connection (psycopg2)")
    print("="*60)

    try:
        import psycopg2
        from urllib.parse import urlparse

        # Parse sync connection string
        if not settings.database_url_sync:
            print("❌ DATABASE_URL_SYNC not configured in .env.example")
            return False

        print(f"Connection string: {settings.database_url_sync[:50]}...")

        # Parse URL
        url = urlparse(settings.database_url_sync)

        # Extract connection parameters
        conn_params = {
            'host': url.hostname,
            'port': url.port or 5432,
            'database': url.path.lstrip('/'),
            'user': url.username,
            'password': url.password,
            'sslmode': 'require',  # Azure requires SSL
        }

        print(f"\nConnection parameters:")
        print(f"  Host: {conn_params['host']}")
        print(f"  Port: {conn_params['port']}")
        print(f"  Database: {conn_params['database']}")
        print(f"  User: {conn_params['user']}")
        print(f"  SSL: {conn_params['sslmode']}")

        # Attempt connection
        print("\n⏳ Connecting...")
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()

        # Test query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"✅ Connected successfully!")
        print(f"PostgreSQL version: {version[:80]}...")

        # Check schema exists
        cursor.execute("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name = %s;
        """, (settings.database_schema,))

        schema_result = cursor.fetchone()
        if schema_result:
            print(f"✅ Schema '{settings.database_schema}' exists")
        else:
            print(f"⚠️  Schema '{settings.database_schema}' does NOT exist")
            print(f"   Run this SQL to create it:")
            print(f"   CREATE SCHEMA {settings.database_schema};")

        # Check permissions
        cursor.execute("""
            SELECT has_schema_privilege(%s, %s, 'CREATE');
        """, (settings.database_url_sync.split('@')[0].split('/')[-1].split(':')[0], settings.database_schema))

        can_create = cursor.fetchone()[0]
        if can_create:
            print(f"✅ User has CREATE permission on schema")
        else:
            print(f"⚠️  User does NOT have CREATE permission on schema")

        # Close connection
        cursor.close()
        conn.close()
        print("\n✅ Sync connection test PASSED")
        return True

    except ImportError:
        print("❌ psycopg2 not installed. Run: pip install psycopg2-binary")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"\nError type: {type(e).__name__}")
        return False


async def test_async_connection():
    """Test asynchronous PostgreSQL connection using asyncpg."""
    print("\n" + "="*60)
    print("Testing ASYNC PostgreSQL Connection (asyncpg)")
    print("="*60)

    try:
        import asyncpg
        from urllib.parse import urlparse

        print(f"Connection string: {settings.database_url[:50]}...")

        # Parse URL
        url = urlparse(settings.database_url)

        # Extract connection parameters
        conn_params = {
            'host': url.hostname,
            'port': url.port or 5432,
            'database': url.path.lstrip('/'),
            'user': url.username,
            'password': url.password,
            'ssl': 'require',  # Azure requires SSL
        }

        print(f"\nConnection parameters:")
        print(f"  Host: {conn_params['host']}")
        print(f"  Port: {conn_params['port']}")
        print(f"  Database: {conn_params['database']}")
        print(f"  User: {conn_params['user']}")
        print(f"  SSL: {conn_params['ssl']}")

        # Attempt connection
        print("\n⏳ Connecting...")
        conn = await asyncpg.connect(**conn_params)

        # Test query
        version = await conn.fetchval("SELECT version();")
        print(f"✅ Connected successfully!")
        print(f"PostgreSQL version: {version[:80]}...")

        # Check schema exists
        schema_result = await conn.fetchval("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name = $1;
        """, settings.database_schema)

        if schema_result:
            print(f"✅ Schema '{settings.database_schema}' exists")
        else:
            print(f"⚠️  Schema '{settings.database_schema}' does NOT exist")
            print(f"   Run this SQL to create it:")
            print(f"   CREATE SCHEMA {settings.database_schema};")

        # Check current user
        current_user = await conn.fetchval("SELECT current_user;")
        print(f"✅ Connected as user: {current_user}")

        # Check search path
        search_path = await conn.fetchval("SHOW search_path;")
        print(f"ℹ️  Current search_path: {search_path}")

        # Close connection
        await conn.close()
        print("\n✅ Async connection test PASSED")
        return True

    except ImportError:
        print("❌ asyncpg not installed. Run: pip install asyncpg")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"\nError type: {type(e).__name__}")

        # Common error messages
        if "password authentication failed" in str(e):
            print("\n💡 Tip: Check your username and password in .env.example")
        elif "does not exist" in str(e):
            print("\n💡 Tip: Check the database name in DATABASE_URL")
        elif "could not translate host name" in str(e):
            print("\n💡 Tip: Check the host address in DATABASE_URL")
        elif "SSL" in str(e) or "ssl" in str(e):
            print("\n💡 Tip: Azure PostgreSQL requires SSL connections")

        return False


def test_sqlalchemy_engine():
    """Test SQLAlchemy engine initialization."""
    print("\n" + "="*60)
    print("Testing SQLAlchemy Engine Initialization")
    print("="*60)

    try:
        from backend.database import database

        print("⏳ Initializing engines...")
        database.init_engines()

        if settings.is_postgres():
            if database.async_engine:
                print("✅ Async engine initialized")
                print(f"   Pool size: {settings.db_pool_size}")
            else:
                print("❌ Async engine failed to initialize")

            if database.sync_engine:
                print("✅ Sync engine initialized (for migrations)")
            else:
                print("⚠️  Sync engine not initialized (optional for PostgreSQL)")
        else:
            if database.sync_engine:
                print("✅ Sync engine initialized (SQLite)")
            else:
                print("❌ Sync engine failed to initialize")

        print("\n✅ SQLAlchemy engine test PASSED")
        return True

    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return False


async def main():
    """Run all connection tests."""
    print("\n" + "="*60)
    print("PostgreSQL Connection Test Suite")
    print("="*60)
    print(f"\nEnvironment: {settings.environment}")
    print(f"Database Type: {'PostgreSQL' if settings.is_postgres() else 'SQLite'}")
    print(f"Target Schema: {settings.database_schema}")

    if not settings.is_postgres():
        print("\n⚠️  Warning: DATABASE_URL is not set to PostgreSQL")
        print("   Update DATABASE_URL in .env.example to use PostgreSQL")
        return

    results = []

    # Test 1: Sync connection
    results.append(("Sync Connection", test_sync_connection()))

    # Test 2: Async connection
    results.append(("Async Connection", await test_async_connection()))

    # Test 3: SQLAlchemy engines
    results.append(("SQLAlchemy Engines", test_sqlalchemy_engine()))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed! Database connection is ready.")
        print("\n✅ Next steps:")
        print("   1. Run: alembic init alembic")
        print("   2. Configure alembic/env.py")
        print("   3. Create migration: alembic revision --autogenerate -m 'Initial migration'")
        print("   4. Apply migration: alembic upgrade head")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Check .env.example has correct DATABASE_URL")
        print("   - Ensure schema exists in PostgreSQL")
        print("   - Verify user has necessary permissions")


if __name__ == "__main__":
    asyncio.run(main())
