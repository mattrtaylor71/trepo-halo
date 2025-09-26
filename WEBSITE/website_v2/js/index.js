async function updateUI() {
    const authButton = document.getElementById("auth-button");
    const welcomeMessage = document.getElementById("welcome-message");
    const authLinks = document.getElementById("auth-links");
    const userProfileSection = document.getElementById("user-profile");
    const profileDataElement = document.getElementById("profile-data");
    const userImagesContainer = document.getElementById("user-images-container");
    const userImagesContainer1 = document.getElementById("user-images-container1");
    const userImagesContainer2 = document.getElementById("user-images-container2");
    const setupGuideTitle = document.getElementById("setup-guide-title");
    const mountingTitle = document.getElementById("mounting-title");
    const howToUseTitle = document.getElementById("how-to-use-title");

    if (!auth0Client) {
        console.error("Auth0 client not initialized");
        return;
    }

    const isAuthenticatedUser = await auth0Client.isAuthenticated();
    console.log("Is user authenticated:", isAuthenticatedUser);

    if (isAuthenticatedUser) {
        try {
            const user = await auth0Client.getUser();
            console.log("User is authenticated:", user);
            localStorage.setItem("user", JSON.stringify(user));
            console.log("User data stored in localStorage:", localStorage.getItem("user"));

            authButton.textContent = "Logout";
            authButton.onclick = logout;
            welcomeMessage.textContent = `Welcome, ${user.name}`;
            welcomeMessage.classList.add("authenticated-welcome");

            if (authLinks) {
                authLinks.style.display = "flex";
            }

            if (userProfileSection && profileDataElement) {
                userProfileSection.classList.remove("hidden");
                profileDataElement.textContent = JSON.stringify(user, null, 2);
            }

            //  SHOW IMAGES WHEN LOGGED IN
            if (userImagesContainer) {
                userImagesContainer.style.display = "block";
            }
            if (userImagesContainer1) {
                userImagesContainer1.style.display = "block";
            }
            if (userImagesContainer2) {
                userImagesContainer2.style.display = "block";
            }
            if (setupGuideTitle) setupGuideTitle.classList.remove("hidden");
            if (mountingTitle) mountingTitle.classList.remove("hidden");
            if (howToUseTitle) howToUseTitle.classList.remove("hidden");
            // hide the buy button
            document
              .querySelectorAll("stripe-buy-button, .BuyButton-ButtonTextContainer")
              .forEach(el => (el.style.display = "none"));

            // hide the “Don’t have an account yet?” line
            document
              .querySelectorAll(".dont-have-an-accoun, .span0-1")
              .forEach(el => (el.style.display = "none"));

            // hide the default hero headline 
            document
              .querySelectorAll(".healthier-eating-ev, .span0")
              .forEach(el => (el.style.display = "none"));

            // hide all of the preview panels
            document
              .querySelectorAll(".preview-container, .preview-image, .preview-gif")
              .forEach(el => (el.style.display = "none"));

            // hide the “Setup Guide” title
            const setupTitle = document.getElementById("setup-guide-title");
            if (setupTitle) setupTitle.style.display = "none";

            console.log("🔐 User logged in; now loading weekly summary");
            await fetchWeeklySummary();

        } catch (error) {
            console.error("Error retrieving user data:", error);
        }
    } else {
        console.log("User is not authenticated.");
        const summaryContainer = document.getElementById("weekly-summary");
        if (summaryContainer) summaryContainer.innerHTML = "";
        authButton.textContent = "Login";
        authButton.onclick = login;
        welcomeMessage.textContent = "Put your kitchen on autopilot.";
        welcomeMessage.classList.remove("authenticated-welcome");

        if (authLinks) {
            authLinks.style.display = "none";
        }

        localStorage.removeItem("user");
        console.log("User data removed from localStorage.");

        // ❌ HIDE IMAGES WHEN LOGGED OUT
        if (userImagesContainer) {
            userImagesContainer.style.display = "none";
        }
        if (userImagesContainer1) {
            userImagesContainer1.style.display = "none";
        }
        if (userImagesContainer2) {
            userImagesContainer2.style.display = "none";
        }
        if (setupGuideTitle) setupGuideTitle.classList.add("hidden");
        if (mountingTitle) mountingTitle.classList.add("hidden");
        if (howToUseTitle) howToUseTitle.classList.add("hidden");
    }
}

document.addEventListener("DOMContentLoaded", function () {
    const pageContent = document.getElementById("page-content");
    const images = document.querySelectorAll("img");
    let imagesLoaded = 0;

    function checkAllImagesLoaded() {
        imagesLoaded++;
        if (imagesLoaded === images.length) {
            pageContent.style.display = "block"; // Show content after all images load
        }
    }

    images.forEach((img) => {
        if (img.complete) {
            checkAllImagesLoaded(); // Already loaded images
        } else {
            img.onload = checkAllImagesLoaded; // Wait for image to load
            img.onerror = checkAllImagesLoaded; // Handle errors gracefully
        }
    });
});




// Fetch the most recent "summary" HTML and inject it
async function fetchWeeklySummary() {
  try {
    const user = JSON.parse(localStorage.getItem("user"));
    if (!user) {
      console.error("No user logged in. Cannot fetch weekly summary.");
      return;
    }

    const webId = user.sub;
    const res = await fetch(
      `https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchInsights?web_id=${encodeURIComponent(webId)}`
    );
    if (!res.ok) throw new Error(`Status ${res.status}`);

    const data = await res.json();
    if (!Array.isArray(data) || data.length === 0) {
      console.warn("No insights found for user.");
      return;
    }

    // pick the item with the latest _createdDate
    const mostRecent = data.reduce((latest, item) => {
      const d1 = new Date(item._createdDate);
      const d2 = new Date(latest._createdDate);
      return d1 > d2 ? item : latest;
    }, data[0]);

    // Inject its summary (which is raw HTML)
    const container = document.getElementById("weekly-summary");
    container.innerHTML = mostRecent.summary || "<p>No summary available.</p>";
  } catch (err) {
    console.error("Error fetching weekly summary:", err);
  }
}


// Run when the page loads
window.onload = async () => {
    try {
        await configureAuth0(); // Ensure auth0 client is initialized
        await handleAuthRedirect();
        await updateUI();
        console.log("Auth0 client initialized and UI updated");
    } catch (error) {
        console.error("Error initializing UI:", error);
    }
};
