# TREPO Website - Local Development Version

This is a simplified version of the TREPO website that has been stripped of all backend API dependencies and authentication requirements. It's designed to run locally for development and demonstration purposes.

## What's Changed

- **Removed Auth0 authentication** - No login required, uses mock user data
- **Removed all backend API calls** - All data is now mock data
- **Removed HTTPS redirect** - Can run on localhost without SSL
- **Simplified JavaScript** - All backend dependencies replaced with mock implementations

## Features

The local version includes all the main features with mock data:

- **Home Page** - Landing page with demo content
- **Insights** - Mock consumption insights and scores
- **Week Overview** - Mock weekly progress and recommendations
- **Shopping List** - Mock shopping list with search functionality
- **Beeps** - Mock scan history and statistics
- **Guide** - Setup guide with interactive steps
- **Points** - Mock points system and rewards

## How to Run

1. **Clone or download** this repository to your local machine

2. **Open the project folder** in your terminal/command prompt

3. **Start a local server** using one of these methods:

   **Option A: Using Python (if installed)**
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Python 2
   python -m SimpleHTTPServer 8000
   ```

   **Option B: Using Node.js (if installed)**
   ```bash
   npx http-server -p 8000
   ```

   **Option C: Using PHP (if installed)**
   ```bash
   php -S localhost:8000
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:8000
   ```

5. **Click "Login"** to access the demo features (no real authentication required)

## File Structure

```
userHome/
├── index.html          # Main landing page
├── insights.html       # Insights and analytics page
├── week.html          # Weekly overview page
├── shopping.html      # Shopping list page
├── beeps.html         # Scan history page
├── guide.html         # Setup guide page
├── privacy.html       # Privacy policy
├── accessibility.html # Accessibility info
├── css/              # Stylesheets
├── js/               # JavaScript files (simplified)
├── img/              # Images and assets
└── fonts/            # Custom fonts
```

## Mock Data

All the mock data is defined in the JavaScript files:

- `js/auth.js` - Mock authentication and user data
- `js/index.js` - Mock home page data
- `js/insights.js` - Mock insights and scores
- `js/week.js` - Mock weekly data and progress
- `js/shopping.js` - Mock shopping list and search
- `js/beeps.js` - Mock scan history
- `js/guide.js` - Mock setup guide
- `js/points.js` - Mock points and rewards

## Development Notes

- All API calls have been replaced with mock functions that return sample data
- The UI remains fully functional and interactive
- No external dependencies or API keys required
- Perfect for development, testing, and demonstrations

## Original Features Preserved

- Responsive design
- Interactive UI elements
- Navigation between pages
- Form interactions
- Search functionality (mock)
- Data visualization (mock)

## Browser Compatibility

This version should work in all modern browsers:
- Chrome
- Firefox
- Safari
- Edge

## Troubleshooting

If you encounter issues:

1. **Check the browser console** for any JavaScript errors
2. **Ensure you're running a local server** (not just opening the HTML files directly)
3. **Try a different port** if 8000 is already in use
4. **Clear browser cache** if you see old content

## License

This is a demo version for development purposes. The original TREPO application and its features are proprietary. 