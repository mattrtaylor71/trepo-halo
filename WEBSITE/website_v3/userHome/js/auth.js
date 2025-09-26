// Simplified auth.js for local development
// Removes all Auth0 and backend dependencies

let isLoggedIn = false;
let mockUser = {
    sub: "mock-user-123",
    name: "Demo User",
    email: "demo@trepo.ai",
    picture: "https://via.placeholder.com/150"
};

// Mock authentication functions
async function configureAuth0() {
    console.log("Mock Auth0 client initialized");
    return Promise.resolve();
}

async function login() {
    isLoggedIn = true;
    localStorage.setItem("user", JSON.stringify(mockUser));
    localStorage.setItem("isLoggedIn", "true");
    console.log("Mock login successful");
    updateUI();
}

async function logout() {
    isLoggedIn = false;
    localStorage.removeItem("user");
    localStorage.removeItem("isLoggedIn");
    console.log("Mock logout successful");
    updateUI();
}

async function isAuthenticated() {
    return isLoggedIn || localStorage.getItem("isLoggedIn") === "true";
}

async function getUser() {
    const user = localStorage.getItem("user");
    return user ? JSON.parse(user) : null;
}

async function handleAuthRedirect() {
    // No redirect handling needed for local development
    return Promise.resolve();
}

// Update UI based on authentication state
async function updateUI() {
    const authButton = document.getElementById("auth-button");
    const authLinks = document.getElementById("auth-links");
    const signupSection = document.getElementById("signup-section");
    const guideButton = document.getElementById("guide-button");
    const heroCenter = document.getElementById("hero-center");
    const weeklySummary = document.getElementById("weekly-summary");

    const isAuthenticatedUser = await isAuthenticated();

    if (isAuthenticatedUser) {
        if (authButton) {
            authButton.textContent = "Logout";
            authButton.onclick = logout;
        }
        if (heroCenter) heroCenter.classList.add("hidden");
        if (guideButton) guideButton.style.display = "inline-block";
        if (signupSection) signupSection.style.display = "none";
        if (authLinks) authLinks.style.display = "flex";
        if (weeklySummary) {
            weeklySummary.innerHTML = `
                <div style="padding: 20px; background: #f5f5f5; border-radius: 8px; margin: 20px 0;">
                    <h3>Weekly Summary (Demo)</h3>
                    <p>This is a demo version. In the full app, you would see your personalized weekly consumption insights here.</p>
                    <p><strong>Score:</strong> 85/100</p>
                    <p><strong>Items Scanned:</strong> 23</p>
                    <p><strong>Recommendations:</strong> Try more organic produce this week!</p>
                </div>
            `;
        }
    } else {
        if (authButton) {
            authButton.textContent = "Login";
            authButton.onclick = login;
        }
        if (heroCenter) heroCenter.classList.remove("hidden");
        if (guideButton) guideButton.style.display = "none";
        if (signupSection) signupSection.style.display = "block";
        if (authLinks) authLinks.style.display = "none";
        if (weeklySummary) weeklySummary.innerHTML = "";
    }
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", async () => {
    await configureAuth0();
    
    // Check if user was previously logged in
    if (localStorage.getItem("isLoggedIn") === "true") {
        isLoggedIn = true;
    }
    
    await updateUI();
    
    // Set up logout button
    const logoutButton = document.getElementById("logout-button");
    if (logoutButton) {
        logoutButton.addEventListener("click", logout);
    }
});
