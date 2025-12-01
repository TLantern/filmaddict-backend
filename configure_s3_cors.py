#!/usr/bin/env python3
"""
Configure S3 CORS for localhost development.
Automatically loads credentials from .env file.

Usage:
    python configure_s3_cors.py [bucket-name]
    
If bucket-name is not provided, uses S3_BUCKET_NAME from .env
"""

import os
import sys
import json

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback: manually parse .env file
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Remove inline comments
                        value = value.split('#')[0].strip()
                        os.environ[key.strip()] = value

import boto3
from botocore.exceptions import ClientError

def configure_cors(bucket_name: str):
    """Configure CORS on S3 bucket to allow localhost requests."""
    
    # Get AWS credentials from environment variables
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    
    if not aws_access_key_id or not aws_secret_access_key:
        print("❌ Error: AWS credentials not found!")
        print("   Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file")
        return False
    
    # Clean up region (remove comments)
    if '#' in aws_region:
        aws_region = aws_region.split('#')[0].strip()
    if not aws_region:
        aws_region = 'us-east-1'
    
    # Clean up endpoint URL
    if s3_endpoint_url:
        s3_endpoint_url = s3_endpoint_url.strip()
        if s3_endpoint_url.startswith('#') or not s3_endpoint_url:
            s3_endpoint_url = None
    
    # Create S3 client with credentials
    s3_config = {
        'aws_access_key_id': aws_access_key_id,
        'aws_secret_access_key': aws_secret_access_key,
        'region_name': aws_region,
    }
    
    if s3_endpoint_url:
        s3_config['endpoint_url'] = s3_endpoint_url
    
    try:
        s3_client = boto3.client('s3', **s3_config)
    except Exception as e:
        print(f"❌ Error creating S3 client: {e}")
        return False
    
    cors_configuration = {
        'CORSRules': [
            {
                'AllowedHeaders': ['*'],
                'AllowedMethods': ['GET', 'HEAD'],
                'AllowedOrigins': [
                    'http://localhost:3000',
                    'http://localhost:3001',
                ],
                'ExposeHeaders': ['ETag', 'Content-Length', 'Content-Type'],
                'MaxAgeSeconds': 3000
            }
        ]
    }
    
    try:
        s3_client.put_bucket_cors(
            Bucket=bucket_name,
            CORSConfiguration=cors_configuration
        )
        print(f"✅ CORS configuration applied successfully to bucket: {bucket_name}")
        print(f"✅ Allowed origins: http://localhost:3000, http://localhost:3001")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            print(f"❌ Error: Bucket '{bucket_name}' does not exist")
        elif error_code == 'AccessDenied':
            print(f"❌ Error: Access denied. Check your AWS credentials and bucket permissions")
        else:
            print(f"❌ Error configuring CORS: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    # Get bucket name from command line or environment
    bucket_name = None
    
    if len(sys.argv) > 1:
        bucket_name = sys.argv[1]
    else:
        bucket_name = os.getenv('S3_BUCKET_NAME')
    
    if not bucket_name:
        print("Usage: python configure_s3_cors.py [bucket-name]")
        print("   Or set S3_BUCKET_NAME in .env file")
        sys.exit(1)
    
    success = configure_cors(bucket_name)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

