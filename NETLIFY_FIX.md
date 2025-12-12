# Quick Fix for Netlify Python Dependency Error

## Problem
Netlify is trying to install Python dependencies (qiskit-aer) during build, but this is a **static website** that doesn't need Python!

## Solution (Choose One)

### Option 1: Disable Python in Netlify UI (Recommended)

1. Go to your Netlify site dashboard
2. **Site settings** → **Build & deploy** → **Build settings**
3. Click **"Edit settings"**
4. Look for **"Dependencies"** or **"Build environment"** section
5. **Disable Python** or set Python version to **"None"**
6. Save and trigger a new deploy

### Option 2: Use the Build Script (Already Configured)

The `netlify.toml` and `netlify-build.sh` are already set up. Just make sure:
- Build command in Netlify UI matches: `bash netlify-build.sh`
- Or leave it empty and Netlify will use `netlify.toml`

### Option 3: Manual Override via Environment Variables

1. **Site settings** → **Build & deploy** → **Environment**
2. Add environment variable:
   - **Key**: `SKIP_PYTHON_INSTALL`
   - **Value**: `true`
3. Or remove any Python-related variables

## Files Created to Fix This

- ✅ Renamed `requirements.txt` to `python-requirements.txt` - Netlify won't auto-detect Python
- ✅ `netlify.toml` - Configures build settings
- ✅ `netlify-build.sh` - Simple build script

## After Fixing

1. Push these changes to GitHub (if using Git integration)
2. Trigger a new deploy in Netlify
3. The build should complete in seconds (no Python installation)

## Why This Happens

Netlify auto-detects Python when it sees `requirements.txt`. We've renamed it to `python-requirements.txt` so Netlify won't detect it. Since this is a static HTML/CSS/JS site, Python dependencies are only needed for:
- Local development
- Training the model (`train_model.py`)
- Running the API server (`src/api_server.py`)

None of these are needed for serving the static website!

