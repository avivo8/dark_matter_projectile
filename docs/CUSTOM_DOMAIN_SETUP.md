# Custom Domain Setup: dark_matter_projectile.com

## Step 1: DNS Configuration (At Your Domain Provider)

You need to configure DNS records at your domain registrar (where you bought `dark_matter_projectile.com`).

### Option A: Using A Records (Recommended)

Add these **A records** in your DNS settings:

```
Type: A
Name: @ (or blank/root)
Value: 185.199.108.153
TTL: 3600

Type: A
Name: @ (or blank/root)
Value: 185.199.109.153
TTL: 3600

Type: A
Name: @ (or blank/root)
Value: 185.199.110.153
TTL: 3600

Type: A
Name: @ (or blank/root)
Value: 185.199.111.153
TTL: 3600
```

### Option B: Using CNAME (Alternative)

Add this **CNAME record**:

```
Type: CNAME
Name: @ (or blank/root)
Value: avivo8.github.io
TTL: 3600
```

**Note:** Some providers don't allow CNAME on root domain (@). If that's the case, use Option A.

### For www Subdomain

Add this **CNAME record**:

```
Type: CNAME
Name: www
Value: avivo8.github.io
TTL: 3600
```

## Step 2: Enable Custom Domain in GitHub

1. Go to: **https://github.com/avivo8/dark_matter_projectile/settings/pages**

2. Under **"Custom domain"**, enter:
   ```
   dark_matter_projectile.com
   ```

3. Check **"Enforce HTTPS"** (after DNS propagates)

4. Click **"Save"**

## Step 3: Wait for DNS Propagation

- DNS changes can take **5 minutes to 48 hours** to propagate
- Usually takes **15-30 minutes**

Check if it's working:
```bash
dig dark_matter_projectile.com
# or visit: https://www.whatsmydns.net/#A/dark_matter_projectile.com
```

## Step 4: Verify It Works

Once DNS propagates:
1. Visit: **https://dark_matter_projectile.com**
2. GitHub will automatically enable HTTPS (SSL certificate)
3. Both `dark_matter_projectile.com` and `www.dark_matter_projectile.com` will work

## Common Domain Providers

### Namecheap
1. Go to Domain List → Manage
2. Advanced DNS tab
3. Add A records (use Option A above)

### GoDaddy
1. DNS Management
2. Add A records (use Option A above)

### Cloudflare
1. DNS → Records
2. Add A records (use Option A above)
3. Set Proxy status to "DNS only" (gray cloud) initially

### Google Domains
1. DNS → Custom records
2. Add A records (use Option A above)

## Troubleshooting

**Domain not working?**
- Wait longer (DNS can take up to 48 hours)
- Check DNS propagation: https://www.whatsmydns.net
- Verify A records point to GitHub IPs
- Make sure CNAME file is in the `website/` folder

**HTTPS not working?**
- Wait 24 hours after DNS setup for GitHub to issue SSL certificate
- Uncheck "Enforce HTTPS" temporarily, then re-enable after certificate is issued

**www not working?**
- Make sure you added the CNAME record for `www`
- Wait for DNS propagation

## Your Website URLs

After setup, your website will be accessible at:
- **https://dark_matter_projectile.com** ✅
- **https://www.dark_matter_projectile.com** ✅
- **https://avivo8.github.io/dark_matter_projectile/** (still works)

---

**Note:** The CNAME file has been created in the `website/` folder. After you push this and configure DNS, GitHub Pages will automatically use your custom domain!

