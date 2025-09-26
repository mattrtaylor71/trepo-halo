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
    const logItemButton = document.getElementById("log-item-button");

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

            if (logItemButton) logItemButton.style.display = "inline-block";

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

        if (logItemButton) logItemButton.style.display = "none";

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

    if (!images.length || !pageContent) return;

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

document.addEventListener("DOMContentLoaded", () => {
  const logItemButton = document.getElementById("log-item-button");
  const video = document.getElementById("camera-stream");
  const cameraBox = document.getElementById("camera-box");
  const captureButton = document.getElementById("capture-button");
  const capturedImage = document.getElementById("captured-image");
  const videoWrapper   = document.getElementById("video-wrapper");


  let currentStream = null;

  if (logItemButton) {
    logItemButton.addEventListener("click", async (e) => {
      e.preventDefault();
      if (!video || !cameraBox || !captureButton || !capturedImage) return;

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" }
        });

        currentStream = stream;
        video.srcObject = stream;
        video.play();
        //video.style.display = "block";
        cameraBox.style.display      = "block";
        videoWrapper.style.display   = "inline-block";   // 👉 show live preview
        captureButton.style.display  = "inline-block";
        capturedImage.style.display  = "none";           // 👉 hide old snapshot
        capturedImage.src = ""; // clear previous image
      } catch (err) {
        console.error("Camera error:", err);
      }
    });
  }

  if (captureButton) {
    captureButton.addEventListener("click", async () => {

      const user = JSON.parse(localStorage.getItem("user"));
      const webId = user.sub;            // e.g. “0b358632-…”
      // 1) draw to canvas
      const canvas = document.createElement("canvas");
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);

      const imageData   = canvas.toDataURL("image/jpeg");
      const base64Image = imageData.replace(/^data:image\/jpeg;base64,/, "");

      // 2) show snapshot below the wrapper
      capturedImage.src           = imageData;
      capturedImage.style.display = "block";
      // show the “logged” frost overlay after 0.5s
      setTimeout(() => {
        // create (or reuse) the overlay div
        let overlay = document.querySelector("#snapshot-container .frost-overlay");
        if (!overlay) {
          overlay = document.createElement("div");
          overlay.className = "frost-overlay";
          overlay.innerText = "Logged!";
          document.getElementById("snapshot-container").appendChild(overlay);
        }
        // trigger fade-in
        // insert overlay if needed… then:
        requestAnimationFrame(() => {
          overlay.classList.add("visible");

          // 1s later, fade it out and remove:
          setTimeout(() => {
            overlay.classList.remove("visible");
            // give CSS a moment to revert, then remove entirely:
            setTimeout(() => overlay.remove(), 300);
            document.getElementById("captured-image").style.display = "none";
          }, 1000);
        });


      }, 500);


      // 3) hide the live preview wrapper and the button
      videoWrapper.style.display  = "none";
      captureButton.style.display = "none";

      // 4) stop camera
      currentStream.getTracks().forEach(t => t.stop());
      video.srcObject = null;

      const payload = {
        image: base64Image,
        auth0_sub: webId
      };

      // 5) call your API
      try {
        const res = await fetch("https://c1x2wqys36.execute-api.us-east-1.amazonaws.com/identify", {
          method:  "POST",
          headers: {"Content-Type":"application/json"},
          body:    JSON.stringify(payload),
        });
        const {description} = await res.json();
        //alert(`Identified: ${description}`);
      } catch(err) {
        console.error(err);
        //alert("Failed to identify image.");
      }
    });

  }
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

