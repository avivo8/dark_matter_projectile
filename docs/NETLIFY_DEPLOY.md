# Deploying to Netlify

Netlify is a great platform for hosting static websites. Here are three ways to deploy your Dark Matter Projectile website:

## Method 1: Git Integration (Recommended - Auto-deploys on push)

This is the best option if you want automatic deployments whenever you push to GitHub.

### Steps:

1. **Go to Netlify Dashboard**
   - Visit: https://app.netlify.com
   - Log in with your Netlify account

2. **Add New Site**
   - Click "Add new site" → "Import an existing project"
   - Choose "GitHub" and authorize Netlify to access your repositories
   - Select your repository: `avivo8/dark_matter_projectile`

3. **Configure Build Settings**
   - **Base directory**: Leave empty (or set to `/` if needed)
   - **Build command**: Leave empty (no build needed for static site)
   - **Publish directory**: `website`
   - Click "Deploy site"

4. **Set Custom Domain (Optional)**
   - After deployment, go to Site settings → Domain management
   - Click "Add custom domain"
   - Enter: `dark_matter_projectile.com`
   - Follow DNS instructions (similar to GitHub Pages)

5. **Your site will be live at:**
   - `https://your-site-name.netlify.app` (default)
   - `https://dark_matter_projectile.com` (after DNS setup)

---

## Method 2: Drag and Drop (Quickest - Manual)

Perfect for a quick test or one-time deployment.

### Steps:

1. **Prepare the website folder**
   ```bash
   cd /home/aviv/Documents/dark_matter_projectile
   # Make sure all files are in the website/ directory
   ```

2. **Go to Netlify**
   - Visit: https://app.netlify.com
   - Log in

3. **Drag and Drop**
   - Click "Add new site" → "Deploy manually"
   - Drag the entire `website/` folder into the deployment area
   - Wait for upload and deployment

4. **Your site is live!**
   - Netlify will give you a URL like: `https://random-name.netlify.app`

**Note:** With this method, you'll need to drag and drop again for each update.

---

## Method 3: Netlify CLI (For Developers)

If you prefer command-line tools.

### Steps:

1. **Install Netlify CLI**
   ```bash
   npm install -g netlify-cli
   ```

2. **Login to Netlify**
   ```bash
   netlify login
   ```

3. **Deploy**
   ```bash
   cd /home/aviv/Documents/dark_matter_projectile/website
   netlify deploy --prod
   ```

4. **First time setup**
   - Follow prompts to create a new site or link to existing site
   - Choose publish directory: `.` (current directory)

---

## Custom Domain Setup (if using dark_matter_projectile.com)

1. **In Netlify Dashboard:**
   - Go to Site settings → Domain management
   - Click "Add custom domain"
   - Enter: `dark_matter_projectile.com`

2. **Configure DNS at your domain registrar:**
   - Add a CNAME record:
     - **Name**: `www`
     - **Value**: `your-site-name.netlify.app`
   - Add an A record (or use Netlify's DNS):
     - **Name**: `@` (or blank)
     - **Value**: Netlify will provide IP addresses (usually shown in the dashboard)

3. **Enable HTTPS:**
   - Netlify automatically provisions SSL certificates
   - Go to Domain settings → HTTPS
   - Click "Verify DNS configuration" and wait for certificate

---

## Advantages of Netlify over GitHub Pages

- ✅ **Faster deployments** - Usually deploys in seconds
- ✅ **Better performance** - Global CDN
- ✅ **Automatic HTTPS** - Free SSL certificates
- ✅ **Preview deployments** - Test before going live
- ✅ **Form handling** - Built-in form processing
- ✅ **Serverless functions** - Can add backend features later
- ✅ **Better analytics** - Built-in visitor stats

---

## Troubleshooting

### Images not loading?
- Make sure image paths in HTML are relative (e.g., `visualizations/image.png` not `/visualizations/image.png`)
- Check that all files are in the `website/` directory

### API server not working?
- Netlify hosts static sites only
- For the Flask API (`src/api_server.py`), you'll need:
  - Netlify Functions (serverless)
  - Or a separate hosting service (Heroku, Railway, Render, etc.)

### Custom domain not working?
- Wait 24-48 hours for DNS propagation
- Check DNS records at: https://www.whatsmydns.net
- Verify DNS in Netlify dashboard

---

## Quick Start (Recommended)

**Use Method 1 (Git Integration)** - it's the easiest and keeps your site updated automatically!

1. Go to https://app.netlify.com
2. Click "Add new site" → "Import an existing project"
3. Connect GitHub → Select `dark_matter_projectile`
4. Set **Publish directory**: `website`
5. Click "Deploy site"
6. Done! 🎉

