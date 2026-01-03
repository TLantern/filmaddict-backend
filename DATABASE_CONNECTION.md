# Database Connection Guide

## Quick Setup

### 1. Create/Edit `backend/.env` file

Add your database connection string:

```env
DATABASE_URL=postgresql+asyncpg://username:password@host:port/database_name
```

### 2. Connection String Formats

**Local PostgreSQL:**
```env
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/filmaddict
```

**Neon/Supabase (Cloud PostgreSQL with SSL):**
```env
DATABASE_URL=postgresql+asyncpg://user:password@host.region.neon.tech:5432/dbname?sslmode=require
```

**Example Neon connection:**
```env
DATABASE_URL=postgresql+asyncpg://user:pass@ep-cool-darkness-123456.us-east-2.aws.neon.tech:5432/neondb?sslmode=require
```

### 3. Verify Database is Running

**Check if PostgreSQL is running:**
```bash
# Windows (PowerShell)
Get-Service -Name postgresql*

# Or check if port 5432 is listening
netstat -an | findstr :5432
```

**Test connection manually:**
```bash
# Using psql (if installed)
psql -h localhost -U postgres -d filmaddict

# Or test with Python
python -c "import asyncpg; import asyncio; asyncio.run(asyncpg.connect('postgresql://user:pass@localhost/dbname'))"
```

### 4. Common Issues & Solutions

**Issue: "Authentication timed out"**
- Database server is not running → Start PostgreSQL service
- Wrong host/port → Check DATABASE_URL
- Firewall blocking → Allow port 5432
- Network issues → Check internet connection (for cloud DB)

**Issue: "Connection refused"**
- PostgreSQL not running → Start the service
- Wrong port → Default is 5432
- Wrong host → Use `localhost` for local, or correct hostname for cloud

**Issue: "Authentication failed"**
- Wrong username/password → Check credentials
- User doesn't exist → Create user in PostgreSQL

**Issue: "Database does not exist"**
- Create the database:
```sql
CREATE DATABASE filmaddict;
```

### 5. Test Connection from Backend

Create a test script `backend/test_db_connection.py`:

```python
import asyncio
import os
from dotenv import load_dotenv
from database import engine

load_dotenv()

async def test_connection():
    try:
        async with engine.begin() as conn:
            result = await conn.execute("SELECT 1")
            print("✅ Database connection successful!")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_connection())
```

Run it:
```bash
cd backend
python test_db_connection.py
```

### 6. Connection Parameters Explained

The connection string format:
```
postgresql+asyncpg://[username]:[password]@[host]:[port]/[database]?[params]
```

- `postgresql+asyncpg://` - Protocol (asyncpg driver)
- `username` - Database user
- `password` - Database password
- `host` - Database host (localhost, IP, or domain)
- `port` - Database port (default: 5432)
- `database` - Database name
- `?sslmode=require` - Required for cloud databases (Neon, Supabase, etc.)

### 7. For Cloud Databases (Neon/Supabase)

1. Get connection string from your database provider dashboard
2. Replace `postgresql://` with `postgresql+asyncpg://`
3. Add `?sslmode=require` if not present
4. Remove `channel_binding` parameter if present (asyncpg doesn't support it)

Example transformation:
```
# Original (from Neon dashboard)
postgresql://user:pass@host.neon.tech/db?sslmode=require&channel_binding=prefer

# Use this (for asyncpg)
postgresql+asyncpg://user:pass@host.neon.tech/db?sslmode=require
```

### 8. Restart Backend After Changes

After updating `.env`:
```bash
# Stop the backend (Ctrl+C)
# Then restart
cd backend
uvicorn main:app --reload
```

