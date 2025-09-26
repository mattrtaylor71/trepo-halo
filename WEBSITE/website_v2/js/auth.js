// auth.js

let auth0Client = null;

// Initialize Auth0 client with client options
async function configureAuth0() {
    auth0Client = await createAuth0Client({
        domain: "dev-7kg4syk6znb1wov7.us.auth0.com",
        client_id: "ViMFxI16zyjdmkY59yuDxajqHQFtj2Dx",
        cacheLocation: "localstorage", // Ensures session persistence across page reloads
        useRefreshTokens: true, // Enables refresh tokens for longer sessions
        redirect_uri: window.location.origin // Redirect back to the homepage after login
    });
    console.log("Auth0 client initialized");
}

// Login function that redirects to the Auth0 Universal Login page
async function login() {
    const loginOptions = {
        authorizationParams: {
            screen_hint: "signup", // Hints to show the signup form on first login
            redirect_uri: window.location.origin
        }
    };
    await auth0Client.loginWithRedirect(loginOptions);
}

// Logout function with redirection to homepage
async function logout() {
    const logoutOptions = {
        logoutParams: {
            returnTo: window.location.origin
        }
    };
    await auth0Client.logout(logoutOptions);
}

// Check if user is authenticated
async function isAuthenticated() {
    return await auth0Client.isAuthenticated();
}

// Retrieve authenticated user profile
async function getUser() {
    return await auth0Client.getUser();
}

// Handle redirect callback from Auth0
async function handleAuthRedirect() {
    const query = window.location.search;
    if (query.includes("code=") && query.includes("state=")) {
        try {
            await auth0Client.handleRedirectCallback();
            window.history.replaceState({}, document.title, "/");
            console.log("Redirect handled and user logged in");
        } catch (error) {
            console.error("Error handling Auth0 redirect:", error);
        }
    }
}

// Attach the logout function to the Logout button
document.addEventListener("DOMContentLoaded", async () => {
    await configureAuth0(); // Ensure Auth0 is initialized

    const logoutButton = document.getElementById("logout-button");
    if (logoutButton) {
        logoutButton.addEventListener("click", async () => {
            await logout();
        });
    }
});
