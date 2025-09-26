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

    if (!auth0Client) {
        console.error("Auth0 client not initialized");
        return;
    }

    const isAuthenticatedUser = await auth0Client.isAuthenticated();
    console.log("Is user authenticated:", isAuthenticatedUser);

    const klScript = document.querySelector(
      'script[src*="static.klaviyo.com/onsite/js"]'
    );

    if (isAuthenticatedUser) {
        try {
            if (signupSection) signupSection.style.display = "none";
            const user = await auth0Client.getUser();
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

