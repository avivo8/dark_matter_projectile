#!/bin/bash
# Netlify build script for static site
# This ensures Netlify doesn't try to install Python dependencies

set -e

echo "🔨 Building static site..."
echo "📁 Publishing directory: website"

# Verify website directory exists
if [ ! -d "website" ]; then
    echo "❌ Error: website directory not found"
    exit 1
fi

# List files that will be published
echo "✅ Website files ready for deployment"
ls -la website/ | head -10

echo "✅ Build complete - static site ready"



