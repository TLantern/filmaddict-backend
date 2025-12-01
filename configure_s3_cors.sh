#!/bin/bash

# Configure S3 CORS for localhost development
# Usage: ./configure_s3_cors.sh [bucket-name]
# If bucket-name is not provided, will use S3_BUCKET_NAME from .env file

# Load .env file if it exists (skip comments and empty lines)
if [ -f .env ]; then
    set -a
    source <(cat .env | sed -e '/^#/d' -e '/^$/d' -e 's/^/export /')
    set +a
fi

BUCKET_NAME=$1

# Use bucket from .env if not provided as argument
if [ -z "$BUCKET_NAME" ]; then
    BUCKET_NAME=$S3_BUCKET_NAME
fi

if [ -z "$BUCKET_NAME" ]; then
    echo "Usage: ./configure_s3_cors.sh [bucket-name]"
    echo "Example: ./configure_s3_cors.sh my-filmaddict-bucket"
    echo "Or set S3_BUCKET_NAME in .env file"
    exit 1
fi

# Check if AWS credentials are set
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "❌ Error: AWS credentials not found!"
    echo "   Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file"
    echo "   Or export them as environment variables"
    exit 1
fi

# Create CORS configuration file
cat > /tmp/cors-config.json <<EOF
{
    "CORSRules": [
        {
            "AllowedHeaders": ["*"],
            "AllowedMethods": ["GET", "HEAD"],
            "AllowedOrigins": [
                "http://localhost:3000",
                "http://localhost:3001"
            ],
            "ExposeHeaders": ["ETag", "Content-Length", "Content-Type"],
            "MaxAgeSeconds": 3000
        }
    ]
}
EOF

# Apply CORS configuration
aws s3api put-bucket-cors --bucket "$BUCKET_NAME" --cors-configuration file:///tmp/cors-config.json

if [ $? -eq 0 ]; then
    echo "✅ CORS configuration applied successfully to bucket: $BUCKET_NAME"
    echo "✅ Allowed origins: http://localhost:3000, http://localhost:3001"
else
    echo "❌ Failed to apply CORS configuration"
    exit 1
fi

# Clean up
rm /tmp/cors-config.json

