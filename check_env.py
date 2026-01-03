"""Check .env file for parsing errors"""
import os
from pathlib import Path

env_path = Path(__file__).parent / '.env'

if not env_path.exists():
    print(f"❌ .env file not found at: {env_path}")
    exit(1)

print(f"Checking .env file: {env_path}\n")

with open(env_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

errors = []
for i, line in enumerate(lines, 1):
    line = line.rstrip('\n\r')
    
    # Skip empty lines and comments
    if not line.strip() or line.strip().startswith('#'):
        continue
    
    # Check for common issues
    if '=' not in line:
        errors.append(f"Line {i}: Missing '=' separator")
        continue
    
    key, value = line.split('=', 1)
    key = key.strip()
    value = value.strip()
    
    # Check for unquoted values with special characters that might cause issues
    if value and not (value.startswith('"') and value.endswith('"')) and not (value.startswith("'") and value.endswith("'")):
        # Check for problematic characters
        if any(char in value for char in [' ', '&', '|', ';', '(', ')', '[', ']', '{', '}']):
            # DATABASE_URL is okay to have these, but others might need quotes
            if key != 'DATABASE_URL':
                errors.append(f"Line {i}: Value contains special characters - consider quoting: {key}=...")
    
    # Check for unmatched quotes
    if value.count('"') % 2 != 0:
        errors.append(f"Line {i}: Unmatched double quotes in value")
    if value.count("'") % 2 != 0:
        errors.append(f"Line {i}: Unmatched single quotes in value")

if errors:
    print("⚠️  Found potential issues:\n")
    for error in errors:
        print(f"  {error}")
    print("\n💡 Tips:")
    print("  - Values with spaces or special characters should be quoted")
    print("  - Example: KEY=\"value with spaces\"")
    print("  - DATABASE_URL doesn't need quotes even with special characters")
else:
    print("✅ No obvious syntax errors found in .env file")

print(f"\n📋 Showing lines 31, 35, 36 (where errors were reported):\n")
for line_num in [31, 35, 36]:
    if line_num <= len(lines):
        print(f"Line {line_num}: {lines[line_num-1].rstrip()}")

