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

    // Detect if viewed on a phone
    const isMobile = window.innerWidth <= 768; // Adjust breakpoint if necessary
    const mainContent = document.querySelector("body > *"); // Select all children of body
    const desktopMessage = document.createElement("div");

    desktopMessage.id = "desktop-message";
    desktopMessage.style.cssText =
        "display: none; text-align: center; margin: 20%; font-size: 1.5rem; color: #333;";
    desktopMessage.textContent = "Please open this website on your phone for the best experience.";

    // Append the desktop message only once
    if (!document.getElementById("desktop-message")) {
        document.body.appendChild(desktopMessage);
    }

    if (!isMobile) {
        Array.from(document.body.children).forEach((child) => {
            if (child.id !== "desktop-message") {
                child.style.display = "none"; // Hide all elements except the desktop message
            }
        });
        desktopMessage.style.display = "block";
        return; // Stop execution if not on a mobile device
    } else {
        Array.from(document.body.children).forEach((child) => {
            if (child.id !== "desktop-message") {
                child.style.display = "block"; // Show all elements except the desktop message
            }
        });
        desktopMessage.style.display = "none";
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
