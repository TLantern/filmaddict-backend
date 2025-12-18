"""
Configure CORS policy for S3 bucket to allow video playback from frontend.

Run this script once to set up CORS on your S3 bucket:
    python backend/scripts/configure_s3_cors.py
"""

import os
import sys
import boto3
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def configure_cors():
    """Configure CORS policy for S3 bucket."""
    
    bucket_name = os.getenv("S3_BUCKET_NAME")
    access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION", "us-east-1")
    
    if not bucket_name:
        print("❌ Error: S3_BUCKET_NAME not found in environment variables")
        return False
    
    if not access_key_id or not secret_access_key:
        print("❌ Error: AWS credentials not found in environment variables")
        return False
    
    # Get frontend URLs from environment or use defaults
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    additional_origins = os.getenv("ADDITIONAL_CORS_ORIGINS", "http://localhost:3001").split(",")
    
    allowed_origins = [frontend_url] + [origin.strip() for origin in additional_origins if origin.strip()]
    
    # CORS configuration
    cors_configuration = {
        'CORSRules': [
            {
                'AllowedHeaders': ['*'],
                'AllowedMethods': ['GET', 'HEAD'],
                'AllowedOrigins': allowed_origins,
                'ExposeHeaders': ['ETag', 'Content-Length', 'Content-Type', 'Accept-Ranges', 'Content-Range'],
                'MaxAgeSeconds': 3000
            }
        ]
    }
    
    try:
        # Create S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region
        )
        
        # Apply CORS configuration
        s3_client.put_bucket_cors(
            Bucket=bucket_name,
            CORSConfiguration=cors_configuration
        )
        
        print(f"✅ Successfully configured CORS for bucket: {bucket_name}")
        print(f"\nAllowed origins:")
        for origin in allowed_origins:
            print(f"  - {origin}")
        
        print(f"\nAllowed methods: GET, HEAD")
        print(f"Max age: 3000 seconds")
        
        # Verify configuration
        response = s3_client.get_bucket_cors(Bucket=bucket_name)
        print(f"\n✅ CORS configuration verified!")
        print(json.dumps(response['CORSRules'], indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Error configuring CORS: {str(e)}")
        return False


if __name__ == "__main__":
    print("🔧 Configuring S3 CORS policy...\n")
    success = configure_cors()
    sys.exit(0 if success else 1)

