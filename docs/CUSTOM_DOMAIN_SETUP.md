# Setting Up Custom Domain: dark_matter_projectile.com

## Step-by-Step Guide

### Prerequisites
- You need to own the domain `dark_matter_projectile.com`
- Access to your domain registrar's DNS settings

---

## Step 1: Add CNAME File ✅
**Already done!** The `CNAME` file has been created in the `website/` folder with your domain name.

---

## Step 2: Configure DNS Settings

Go to your domain registrar (where you bought the domain) and add these DNS records:

### Option A: Using Apex Domain (dark_matter_projectile.com)

Add **4 A records** pointing to GitHub Pages IPs:

| Type | Name | Value | TTL |
|------|------|-------|-----|
| A | @ | 185.199.108.153 | 3600 |
| A | @ | 185.199.109.153 | 3600 |
| A | @ | 185.199.110.153 | 3600 |
| A | @ | 185.199.111.153 | 3600 |

**OR** add a **CNAME record**:

| Type | Name | Value | TTL |
|------|------|-------|-----|
| CNAME | @ | avivo8.github.io | 3600 |

### Option B: Using www Subdomain (www.dark_matter_projectile.com)

Add a **CNAME record**:

| Type | Name | Value | TTL |
|------|------|-------|-----|
| CNAME | www | avivo8.github.io | 3600 |

**Recommended:** Use both A records (for apex) AND CNAME (for www) to support both:
- `dark_matter_projectile.com`
- `www.dark_matter_projectile.com`

---

## Step 3: Enable Custom Domain in GitHub

1. Go to: **https://github.com/avivo8/dark_matter_projectile/settings/pages**

2. Under **"Custom domain"**, enter:
   ```
   dark_matter_projectile.com
   ```

3. Check **"Enforce HTTPS"** (recommended)

4. Click **"Save"**

5. Wait 5-10 minutes for DNS to propagate

---

## Step 4: Verify DNS Propagation

Check if DNS is working:

```bash
# Check A records
dig dark_matter_projectile.com +short

# Check CNAME
dig www.dark_matter_projectile.com +short
```

Or use online tools:
- https://www.whatsmydns.net
- https://dnschecker.org

---

## Step 5: Wait for SSL Certificate

GitHub will automatically provision an SSL certificate (HTTPS) for your domain. This usually takes:
- **5-10 minutes** after DNS is configured
- Up to **24 hours** in some cases

You'll see a green checkmark ✅ next to your domain in GitHub Pages settings when it's ready.

---

## Common Domain Registrars Setup

### Namecheap
1. Go to Domain List → Manage
2. Advanced DNS tab
3. Add A records (use IPs above) or CNAME
4. Save changes

### GoDaddy
1. Go to My Products → DNS
2. Add A records or CNAME
3. Save

### Google Domains
1. Go to DNS → Custom records
2. Add A records or CNAME
3. Save

### Cloudflare
1. Go to DNS → Records
2. Add A records (use IPs above) or CNAME
3. **Important:** Set Proxy status to "DNS only" (gray cloud) for GitHub Pages
4. Save

---

## Troubleshooting

### Domain not working?
1. **Wait longer** - DNS can take up to 48 hours to propagate globally
2. **Check DNS records** - Make sure they're correct
3. **Clear browser cache** - Try incognito mode
4. **Check GitHub Pages settings** - Domain should show green checkmark ✅

### HTTPS not working?
- Wait for GitHub to provision SSL certificate (can take up to 24 hours)
- Make sure "Enforce HTTPS" is checked in GitHub Pages settings
- Clear browser cache

### Still using GitHub subdomain?
- DNS might not have propagated yet
- Check that CNAME file is in the `website/` folder
- Verify DNS records are correct

---

## After Setup

Once configured, your website will be available at:
- **https://dark_matter_projectile.com** (with HTTPS!)
- **https://www.dark_matter_projectile.com** (if you set up www)

The GitHub Pages URL (`avivo8.github.io/dark_matter_projectile`) will automatically redirect to your custom domain.

---

## Need Help?

- GitHub Pages docs: https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site
- Check DNS: https://www.whatsmydns.net

