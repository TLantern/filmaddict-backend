#!/usr/bin/env python3
"""
Quick script to add thumbnail_path column to videos table if it doesn't exist.
"""
import asyncio
import os
from sqlalchemy import text
from database import engine

async def add_thumbnail_column():
    """Add thumbnail_path column to videos table if it doesn't exist."""
    async with engine.begin() as conn:
        # Check if column exists
        result = await conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = 'videos' 
                AND column_name = 'thumbnail_path'
            )
        """))
        exists = result.scalar()
        
        if not exists:
            print("Adding thumbnail_path column to videos table...")
            await conn.execute(text("ALTER TABLE videos ADD COLUMN thumbnail_path VARCHAR"))
            print("✅ Column added successfully!")
        else:
            print("✅ Column already exists, skipping.")

if __name__ == "__main__":
    asyncio.run(add_thumbnail_column())

