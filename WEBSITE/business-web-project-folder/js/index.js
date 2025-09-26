// index.js

async function updateUI() {
    const authButton = document.getElementById("auth-button");
    const welcomeMessage = document.getElementById("welcome-message");
    const authLinks = document.getElementById("auth-links");
    const userProfileSection = document.getElementById("user-profile");
    const profileDataElement = document.getElementById("profile-data");

    // Ensure auth0 is defined
    if (!auth0Client) {
        console.error("Auth0 client not initialized");
        return;
    }

    // Check if the user is authenticated
    const isAuthenticatedUser = await auth0Client.isAuthenticated();

    if (isAuthenticatedUser) {
        const user = await auth0Client.getUser();
        authButton.textContent = "Logout";
        authButton.onclick = logout;
        welcomeMessage.textContent = `Welcome, ${user.name}`;

        // Store user info in localStorage
        localStorage.setItem("user", JSON.stringify(user));

        // Show navigation links if authenticated
        if (authLinks) {
            authLinks.style.display = "flex";
        }

        // Display user profile information
        if (userProfileSection && profileDataElement) {
            userProfileSection.classList.remove("hidden");
            profileDataElement.textContent = JSON.stringify(user, null, 2); // Pretty-print JSON
        }

        console.log("User Profile:", user); // Log user information in the console
    } else {
        authButton.textContent = "Login / Sign Up";
        authButton.onclick = login;
        welcomeMessage.textContent = "Log in to get started.";

        // Hide navigation links if not authenticated
        if (authLinks) {
            authLinks.style.display = "none";
        }

        // Hide user profile section if not authenticated
        if (userProfileSection) {
            userProfileSection.classList.add("hidden");
            profileDataElement.textContent = ""; // Clear the content
        }

        // Clear user info from localStorage if logged out
        localStorage.removeItem("user");
    }
}

// Run when the page loads
window.onload = async () => {
    await configureAuth0(); // Ensure auth0 client is initialized
    await handleAuthRedirect();
    await updateUI();
    console.log("Auth0 client initialized and UI updated");
};
