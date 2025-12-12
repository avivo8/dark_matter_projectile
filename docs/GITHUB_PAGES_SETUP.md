# GitHub Pages Setup Guide

## Easiest Way to Deploy Your Website

Since your code is already on GitHub, **GitHub Pages** is the easiest and free way to make your website live on the internet!

## Quick Setup (5 minutes)

### Step 1: Fix Image Paths ✅
Already done! Image paths have been updated to work with GitHub Pages.

### Step 2: Enable GitHub Pages

1. Go to your GitHub repository: `https://github.com/avivo8/dark_matter_projectile`

2. Click on **Settings** (top right of the repository)

3. Scroll down to **Pages** in the left sidebar

4. Under **Source**, select:
   - **Branch:** `main`
   - **Folder:** `/website` (select the website folder)

5. Click **Save**

6. Wait 1-2 minutes for GitHub to build your site

7. Your website will be live at:
   ```
   https://avivo8.github.io/dark_matter_projectile/
   ```

### Step 3: Verify It Works

- Visit the URL above
- Check that all images load correctly
- Test navigation links

## Alternative: Deploy Root Folder

If you want to deploy from the root folder instead:

1. Move `website/index.html` to the root as `index.html`
2. Move all website files to root
3. In GitHub Pages settings, select `/ (root)` as the source folder

## Troubleshooting

**Images not showing?**
- Make sure the `visualizations/` folder is inside the `website/` folder
- Check that image paths use relative paths (not `../`)

**404 Error?**
- Wait a few minutes after enabling Pages
- Check that the branch is `main` and folder is `/website`
- Clear your browser cache

**Want a custom domain?**
- Add a `CNAME` file in the `website/` folder with your domain name
- Configure DNS settings with your domain provider

## Other Hosting Options (If Needed)

### Netlify (Also Free)
1. Go to https://netlify.com
2. Sign up/login with GitHub
3. Click "New site from Git"
4. Select your repository
5. Set build command: (leave empty)
6. Set publish directory: `website`
7. Deploy!

### Vercel (Also Free)
1. Go to https://vercel.com
2. Sign up/login with GitHub
3. Import your repository
4. Set root directory: `website`
5. Deploy!

## Your Live Website URL

Once GitHub Pages is enabled, your site will be at:
**https://avivo8.github.io/dark_matter_projectile/**

Share this URL with anyone to show off your project! 🚀


