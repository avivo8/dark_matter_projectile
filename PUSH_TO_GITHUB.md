# Instructions to Push to GitHub

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Repository name: `dark_energy_density_proj` (or your preferred name)
5. Description: "Dark Matter Halo Detection using Variational Quantum Classifier"
6. Set visibility to **Public**
7. **DO NOT** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

## Step 2: Push to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
cd /home/aviv/Documents/dark_energy_density_proj

# Add the remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/dark_energy_density_proj.git

# Push to GitHub
git push -u origin main
```

If you're using SSH instead of HTTPS:

```bash
git remote add origin git@github.com:YOUR_USERNAME/dark_energy_density_proj.git
git push -u origin main
```

## Alternative: Using GitHub CLI

If you have GitHub CLI installed:

```bash
gh repo create dark_energy_density_proj --public --source=. --remote=origin --push
```

## Troubleshooting

### Authentication Issues
If you get authentication errors:
- Use a Personal Access Token (PAT) instead of password
- Or set up SSH keys for GitHub

### Branch Name
If your default branch is `master` instead of `main`:
```bash
git branch -m main
git push -u origin main
```

## Verify

After pushing, visit your repository on GitHub to verify all files are uploaded correctly.


