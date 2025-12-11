# Website for Dark Matter Projectile

This directory contains a beautiful, modern website showcasing the Dark Matter Detection project using Quantum Machine Learning.

## Files

- `index.html` - Main HTML structure with all content sections
- `styles.css` - Modern CSS styling with dark space theme
- `script.js` - JavaScript for interactivity and smooth animations

## Features

- **Modern Design**: Dark space-themed UI with gradient accents and smooth animations
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile devices
- **Interactive Navigation**: Smooth scrolling and active section highlighting
- **Galaxy Images**: Placeholder sections for galaxy images (using Unsplash as fallback)
- **Visualizations**: Section to display your dark matter visualization PNG
- **Comprehensive Content**: 
  - Project overview
  - Methodology explanations
  - Results and performance metrics
  - Technology stack information

## How to View the Website

### Option 1: Open Directly in Browser

Simply open `index.html` in your web browser:

```bash
# On Linux/Mac
open index.html
# or
xdg-open index.html

# On Windows
start index.html
```

### Option 2: Use Python's Simple HTTP Server

For better functionality (especially for images), use a local server:

```bash
# Python 3
python3 -m http.server 8000

# Then open in browser:
# http://localhost:8000
```

### Option 3: Deploy to GitHub Pages

1. Push the website files to your GitHub repository
2. Go to repository Settings → Pages
3. Select the branch (usually `main`) and folder (usually `/root`)
4. Your site will be available at: `https://avivo8.github.io/dark_matter_projectile/`

### Option 4: Deploy to Other Platforms

- **Netlify**: Drag and drop the folder or connect your GitHub repo
- **Vercel**: Connect your GitHub repository
- **GitHub Pages**: As described above

## Customization

### Adding Real Galaxy Images

Replace the placeholder image URLs in `index.html` (around line 90-120) with your own galaxy images:

```html
<img src="path/to/your/galaxy-image.jpg" alt="Spiral Galaxy">
```

### Adding Your Visualization

Make sure `dark_matter_visualization.png` is in the same directory as `index.html`. The website will automatically display it in the Visualizations section.

### Changing Colors

Edit the CSS variables in `styles.css` (lines 8-16) to customize the color scheme:

```css
:root {
    --primary-color: #6366f1;
    --secondary-color: #8b5cf6;
    /* ... other colors ... */
}
```

## Notes

- The website uses external fonts from Google Fonts (Inter and Space Grotesk)
- Galaxy images use Unsplash as a fallback - replace with your own images for production
- All animations and effects are pure CSS/JavaScript (no external dependencies)
- The website is fully responsive and works on all modern browsers

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

Enjoy your beautiful website! 🌌

