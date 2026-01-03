"""Test database connection"""
import asyncio
from sqlalchemy import text
from dotenv import load_dotenv
from database import engine

load_dotenv()

async def test_connection():
    """Test if database connection works"""
    try:
        print("Testing database connection...")
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            if row and row[0] == 1:
                print("✅ Database connection successful!")
                return True
            else:
                print("❌ Database connection test failed")
                return False
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print(f"\nError type: {type(e).__name__}")
        print("\nTroubleshooting:")
        print("1. Check if DATABASE_URL is set in backend/.env")
        print("2. Verify PostgreSQL is running")
        print("3. Verify database credentials are correct")
        print("4. Check network/firewall settings")
        print("5. Check .env file for syntax errors (quotes, special characters)")
        return False

if __name__ == "__main__":
    asyncio.run(test_connection())

