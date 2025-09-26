// Simplified index.js for local development
// Removes all Auth0 and backend dependencies

async function updateUI() {
    const authButton = document.getElementById("auth-button");
    const authLinks = document.getElementById("auth-links");
    const userProfileSection = document.getElementById("user-profile");
    const profileDataElement = document.getElementById("profile-data");
    const userImagesContainer = document.getElementById("user-images-container");
    const userImagesContainer1 = document.getElementById("user-images-container1");
    const userImagesContainer2 = document.getElementById("user-images-container2");
    const setupGuideTitle = document.getElementById("setup-guide-title");
    const mountingTitle = document.getElementById("mounting-title");
    const howToUseTitle = document.getElementById("how-to-use-title");
    const signupSection = document.getElementById("signup-section");
    const guideButton = document.getElementById("guide-button");
    const diveButton = document.getElementById("dive-button");
    const heroCenter = document.getElementById("hero-center");

    const isAuthenticatedUser = await isAuthenticated();
    console.log("Is user authenticated:", isAuthenticatedUser);

    const klScript = document.querySelector(
      'script[src*="static.klaviyo.com/onsite/js"]'
    );

    if (isAuthenticatedUser) {
        try {
            if (signupSection) signupSection.style.display = "none";
            const user = await getUser();
            console.log("User is authenticated:", user);

            localStorage.setItem("user", JSON.stringify(user));
            console.log("User data stored in localStorage:", localStorage.getItem("user"));

            authButton.textContent = "Logout";
            authButton.onclick = logout;
            if (heroCenter) heroCenter.classList.add("hidden");   // ⬅ hide splash
            if (klScript) klScript.remove();
            if (guideButton) guideButton.style.display = "inline-block";
            if (diveButton) diveButton.style.display = "inline-block";
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
        if (heroCenter) heroCenter.classList.remove("hidden"); // ⬅ show splash
        if (guideButton) guideButton.style.display = "none";
        if (diveButton) diveButton.style.display = "none";
        if (signupSection) signupSection.style.display = "block";

        if (authLinks) {
            authLinks.style.display = "none";
        }
        if (!klScript) {
          const s = document.createElement("script");
          s.src = "https://static.klaviyo.com/onsite/js/VxKUiv/klaviyo.js?v=20250108.1037";
          s.async = true;
          document.head.appendChild(s);
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

async function fetchWeeklySummary() {
  try {
    const user = JSON.parse(localStorage.getItem("user"));
    if (!user) {
      console.error("No user logged in. Cannot fetch weekly summary.");
      return;
    }

    // Mock weekly summary data
    const container = document.getElementById("weekly-summary");
    if (container) {
        container.innerHTML = `
            <div style="padding: 20px; background: #f5f5f5; border-radius: 8px; margin: 20px 0;">
                <h3>Weekly Summary (Demo)</h3>
                <p>This is a demo version. In the full app, you would see your personalized weekly consumption insights here.</p>
                <p><strong>Score:</strong> 85/100</p>
                <p><strong>Items Scanned:</strong> 23</p>
                <p><strong>Recommendations:</strong> Try more organic produce this week!</p>
                <p><strong>Top Categories:</strong> Fruits (40%), Vegetables (30%), Dairy (20%), Snacks (10%)</p>
            </div>
        `;
    }
  } catch (err) {
    console.error("Error fetching weekly summary:", err);
  }
}

document.addEventListener("DOMContentLoaded", ()=>{
  const btn  = document.getElementById("contact-btn");
  const tip  = document.getElementById("email-tip");
  const addr = "matt@trepo.ai";

  btn.addEventListener("click", async ()=>{
    try{
      await navigator.clipboard.writeText(addr);
    }catch{ /* clipboard may fail (http context), fallback ≈ no-op */ }

    tip.classList.remove("hidden");          // show tooltip
    setTimeout(()=> tip.classList.add("hidden"), 2500); // hide after 2.5 s
  });
});

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

