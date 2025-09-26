function shiny(text){
  return `<span class="loader-shimmer">${text}</span>`;
}

function renderRecipes(recipesJson) {
  [0,1,2].forEach(i => {
    const slot = document.querySelector(`#recipe-${i} .span2`);
    const html  = recipesJson[`recipe${i}`] || "";
    if (slot) slot.innerHTML = marked.parse(html);
  });
}

/* place near the top of insights.js */
function clearRecipeSlots () {
  [0,1,2].forEach(i => {
    const span   = document.querySelector(`#recipe-${i} .span2`);
    const loader = document.querySelector(`#recipe-${i} .ai-loader`);
    if (span)   span.innerHTML = "";      // ⬅️ remove previous recipe
    if (loader) loader.classList.add("hidden");
  });
}

/* --- LOADER MESSAGES & ENGINE --------------------------------- */
const loaderLines = [
  "Warming up the ovens 🔥",
  "Messaging Gordon Ramsay 📞🍳",
  "Negotiating with your taste buds 🤝👅",
  "Querying secret grandma recipes 👵📜",
  "Counting macros on an abacus 🧮🔢",
  "Bribing veggies to taste better 🥦💰",
  "Searching the spice multiverse 🌶️🪐",
  "Asking AI nutritionists for gossip 🤖🥑",
  "Sprinkling extra dopamine ✨😋",
  "Plating your culinary victory 🍽️🏆"
];

let loaderTimer = null;

function startFancyLoader(box){
  let idx = 0;
  box.innerHTML = shiny(loaderLines[idx]);
  box.classList.remove("hidden");
  box.classList.add("fade-cycle");    // <- gets the opacity transition

  const swapText = () => {
    box.classList.add("fade-out");          // start lift+fade
    setTimeout(() => {
      idx = (idx + 1) % loaderLines.length; // swap the line
      box.innerHTML = shiny(loaderLines[idx]); 
      box.classList.remove("fade-out");     // drop back & fade in
    }, 400);                                // matches the CSS .4 s
  };

  /* run immediately after first 4 s and keep going every 4 s */
  loaderTimer = setInterval(swapText, 4000);
}


/** stops the cycle and clears the timer */
function stopFancyLoader(){
  if (loaderTimer){
    clearInterval(loaderTimer);
    loaderTimer = null;
  }
}


async function fetchInsights() {
    console.log("Fetching insights...");

    const scoreElement = document.querySelector(".number");
    const reasoningBox = document.querySelector(".highlights-your-di");
    const improvementBox = document.querySelector(".areas-for-improvemen");
    const wellDoneParagraph = document.querySelector(".well-done-this-is-a");

    if (!scoreElement || !reasoningBox || !improvementBox || !wellDoneParagraph) {
        console.error("One or more elements not found. Check class names in HTML.");
        return;
    }

    try {
        const user = JSON.parse(localStorage.getItem("user"));
        if (!user) {
            console.error("User data not found in localStorage. Please log in.");
            alert("Please log in to access insights.");
            return;
        }

        const webId = user.sub;

        const response = await fetch(
            `https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchInsights?web_id=${webId}`
        );
        if (!response.ok) {
            console.error(`API call failed with status: ${response.status}`);
            return;
        }

        const data = await response.json();

        if (data.length === 0) {
            scoreElement.textContent = "0";
            reasoningBox.textContent = "Get scanning to generate insights!";
            improvementBox.textContent = "";
            wellDoneParagraph.textContent = "";
            return;
        }

        const mostRecentScore = data.reduce((latest, item) => {
            const itemDate = new Date(item._createdDate);
            return !latest || itemDate > new Date(latest._createdDate) ? item : latest;
        }, null);

        if (mostRecentScore) {
            const roundedScore = Math.round(mostRecentScore.score || 0);

            scoreElement.textContent = roundedScore;

            // reason
            reasoningBox.querySelector(".span1").textContent =
                mostRecentScore.reasoning || "No reasoning available.";

            // recommendation
            improvementBox.querySelector(".span1").textContent =
                mostRecentScore.recommendation || "No recommendations available.";


            //const recipeSpan = document.querySelector(".span2").innerHTML = marked.parse(mostRecentScore.recipe || "");

            // for each slot 0,1,2
            let recipesJson = {};
            try {
              recipesJson = JSON.parse(mostRecentScore.recipe || "{}");
            } catch(e) {
              console.error("Invalid recipe JSON", e);
            }
            renderRecipes(recipesJson);



        }

    } catch (error) {
        console.error("Error fetching insights:", error);
    }
}

async function updateUPFAndHarmfulIngredients() {
    console.log("Fetching items to calculate UPF percentage and harmful ingredients count...");

    try {
        const user = JSON.parse(localStorage.getItem("user"));
        if (!user) {
            console.error("User data not found in localStorage. Please log in.");
            return;
        }

        const userName = user.sub;

        // Fetch recyclable items (this is the correct API)
        const response = await fetch(
            "https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchRecyclables",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                },
                body: JSON.stringify({ user_name: userName })
            }
        );

        if (!response.ok) {
            console.error(`API call failed with status: ${response.status}`);
            return;
        }

        const data = await response.json();

        if (data.length === 0) {
            console.warn("No items found.");
            document.querySelector(".upf-number").textContent = "0";
            document.querySelector(".harmful-number").textContent = "0";
            return;
        }

        // Count total items and UPF items
        let totalItems = data.length;
        let upfCount = data.filter(item =>
            item.UPF && (item.UPF.trim().toLowerCase() === "yes" || item.UPF.trim().toLowerCase() === "yes.")
        ).length;

        // Count harmful ingredients
        let harmfulIngredientCount = data.reduce((sum, item) => {
            if (item.harmful_ingredients && typeof item.harmful_ingredients === "string") {
                let ingredients = item.harmful_ingredients.split(",").map(i => i.trim());
                return sum + ingredients.length;
            }
            return sum;
        }, 0);

        // Calculate UPF percentage
        let upfPercentage = totalItems > 0 ? Math.round((upfCount / totalItems) * 100) : 0;

        console.log(`Total Items: ${totalItems}, UPF Items: ${upfCount}, UPF %: ${upfPercentage}%`);
        console.log(`Total Harmful Ingredients: ${harmfulIngredientCount}`);

        // Update the UI with the UPF percentage and harmful ingredient count
        document.querySelector(".upf-number").textContent = upfPercentage;
        document.querySelector(".harmful-number").textContent = harmfulIngredientCount;
    } catch (error) {
        console.error("Error calculating UPF percentage and harmful ingredients:", error);
    }
}

async function fetchWeeklyScores(webId) {
    const response = await fetch(
        `https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchInsights?web_id=${webId}`
    );
    if (!response.ok) {
        throw new Error(`Failed to fetch insights: ${response.status}`);
    }

    const data = await response.json();
    const scoresByWeek = {};

    const now = new Date();
    for (let i = 0; i < 6; i++) {
        const startOfWeek = new Date(now.getFullYear(), now.getMonth(), now.getDate() - now.getDay() - i * 7);
        const endOfWeek = new Date(startOfWeek.getFullYear(), startOfWeek.getMonth(), startOfWeek.getDate() + 6);

        const weeklyScores = data.filter((item) => {
            const itemDate = new Date(item._createdDate);
            return itemDate >= startOfWeek && itemDate <= endOfWeek;
        });

        const lastScore = weeklyScores.reduce((latest, item) => {
            return !latest || new Date(item._createdDate) > new Date(latest._createdDate) ? item : latest;
        }, null);

        if (lastScore) {
            scoresByWeek[`${endOfWeek.getMonth() + 1}/${endOfWeek.getDate()}`] = Math.round(lastScore.score || 0);
        }
    }

    return Object.entries(scoresByWeek).map(([weekLabel, lastScore]) => ({ weekLabel, lastScore })).reverse();
}

function renderBars(scoresByWeek) {
    const overlayContent = document.querySelector(".product-comparison-1");
    overlayContent.innerHTML = ""; // Clear existing content

    const title = document.createElement("h3");
    title.textContent = "Weekly Scores";
    title.style.textAlign = "center";
    title.className = "violetsans-regular-normal-green-house-20px";
    overlayContent.appendChild(title);

    const barContainer = document.createElement("div");
    barContainer.style.display = "flex";
    barContainer.style.justifyContent = "space-around";
    barContainer.style.alignItems = "flex-end";
    barContainer.style.height = "300px"; // Restore the original height
    barContainer.style.width = "90%";
    barContainer.style.color = "var(--reef)";
    barContainer.style.borderRadius = "5px";

    const maxScore = 100; // Maximum score value for scaling

    scoresByWeek.forEach(({ weekLabel, lastScore }) => {
        const barWrapper = document.createElement("div");
        barWrapper.className = "bar-wrapper";
        barWrapper.style.display = "flex";
        barWrapper.style.flexDirection = "column";
        barWrapper.style.alignItems = "center";
        barWrapper.style.justifyContent = "flex-end";
        barWrapper.style.height = "100%";
        barWrapper.style.width = "14%"; // Restore original width for each bar

        const scoreLabel = document.createElement("div");
        scoreLabel.textContent = lastScore;
        scoreLabel.style.textAlign = "center";
        scoreLabel.style.marginBottom = "5px";
        scoreLabel.style.fontSize = "14px";
        scoreLabel.style.color = "var(--green-house)";
        scoreLabel.style.fontFamily = "var(--font-family-violet_sans-regular)";
        scoreLabel.style.fontStyle = "normal";
        scoreLabel.style.fontWeight = "400";

        const bar = document.createElement("div");
        bar.className = "bar";
        bar.style.width = "100%"; // Restore full width for bars
        bar.style.height = `${(lastScore / maxScore) * 100}%`; // Calculate height dynamically
        bar.style.backgroundColor = "var(--green-house)";
        bar.style.borderRadius = "5px";

        const dateLabel = document.createElement("div");
        dateLabel.textContent = weekLabel;
        dateLabel.style.textAlign = "center";
        dateLabel.style.marginTop = "10px";
        dateLabel.style.fontSize = "12px";
        dateLabel.style.color = "var(--green-house)";
        dateLabel.style.fontFamily = "var(--font-family-violet_sans-regular)";
        dateLabel.style.fontStyle = "normal";
        dateLabel.style.fontWeight = "400";

        barWrapper.appendChild(scoreLabel);
        barWrapper.appendChild(bar);
        barWrapper.appendChild(dateLabel);
        barContainer.appendChild(barWrapper);
    });

    overlayContent.appendChild(barContainer);
}

async function toggleOverlay(show, type = "scores") {
    const overlay = document.getElementById("product-comparison-overlay");
    const overlayContent = document.querySelector(".product-comparison-1");

    if (show) {
        console.log(`🔄 Opening overlay for: ${type}`);

        // Show the overlay
        overlay.style.display = "flex";

        // Clear the existing content
        overlayContent.innerHTML = "";

        // Create title
        const title = document.createElement("h3");
        title.textContent = type === "scores" ? "Weekly Scores" : "UPF Items";
        title.style.textAlign = "center";
        title.className = "violetsans-regular-normal-green-house-20px";
        overlayContent.appendChild(title);

        if (type === "scores") {
            // Fetch and render the weekly scores
            const user = JSON.parse(localStorage.getItem("user"));
            if (!user) {
                console.error("User data not found in localStorage.");
                return;
            }

            const scoresByWeek = await fetchWeeklyScores(user.sub);
            renderBars(scoresByWeek);
        } else if (type === "upf") {
            // Fetch and render UPF items
            await displayUPFItemsOverlay();
        }

        // Add Close Button
        let closeButton = document.createElement("button");
        closeButton.className = "frame-67";
        closeButton.innerHTML = "&#10006;"; // "X" symbol
        closeButton.onclick = () => toggleOverlay(false);
        closeButton.style.position = "absolute";
        closeButton.style.top = "10px";
        closeButton.style.right = "10px";
        closeButton.style.background = "none";
        closeButton.style.border = "none";
        closeButton.style.fontSize = "18px";
        closeButton.style.cursor = "pointer";

        overlayContent.appendChild(closeButton);
    } else {
        overlay.style.display = "none"; // Hide overlay
    }
}


// Initialize data fetch
document.addEventListener("DOMContentLoaded", () => {
    fetchInsights();
    updateUPFAndHarmfulIngredients();
});


// Add click event to UPF number
document.querySelector(".upf-number").addEventListener("click", async () => {
    await displayUPFItemsOverlay();
});

async function displayUPFItemsOverlay() {
    console.log("Fetching UPF items...");

    try {
        const user = JSON.parse(localStorage.getItem("user"));
        if (!user) {
            console.error("User data not found in localStorage. Please log in.");
            return;
        }

        const userName = user.sub;

        // Fetch recyclable items
        const response = await fetch(
            "https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchRecyclables",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                },
                body: JSON.stringify({ user_name: userName })
            }
        );

        if (!response.ok) {
            console.error(`API call failed with status: ${response.status}`);
            return;
        }

        const data = await response.json();

        if (data.length === 0) {
            console.warn("No items found.");
            return;
        }

        // Filter UPF items
        const upfItems = data.filter(item => 
            item.UPF && (item.UPF.trim().toLowerCase() === "yes" || item.UPF.trim().toLowerCase() === "yes.")
        );

        // Get the overlay content container
        const overlayContent = document.querySelector(".product-comparison-1");
        overlayContent.innerHTML = ""; // Clear previous content

        // Add title
        const title = document.createElement("h2");
        title.textContent = "UPF Items";
        title.style.textAlign = "center";
        title.style.fontSize = "22px"; // Increased header size
        title.style.fontWeight = "bold";
        title.className = "violetsans-regular-normal-green-house-18px";
        overlayContent.appendChild(title);

        // Create list container
        const listContainer = document.createElement("div");
        listContainer.style.padding = "10px";
        listContainer.style.marginTop = "10px";

        if (upfItems.length === 0) {
            const emptyMessage = document.createElement("p");
            emptyMessage.textContent = "No ultra-processed food items found.";
            emptyMessage.style.textAlign = "center";
            overlayContent.appendChild(emptyMessage);
        } else {
            upfItems.forEach(item => {
                const listItem = document.createElement("div");
                listItem.style.display = "flex";
                listItem.style.alignItems = "center";
                listItem.style.padding = "10px";
                listItem.style.borderBottom = "1px solid #ccc";
                listItem.style.fontFamily = "var(--font-family-violet_sans-regular)";
                listItem.style.fontSize = "18px";
                listItem.style.justifyContent = "flex-start"; // Align text to the left

                // Create image
                const image = document.createElement("img");
                image.src = item.images || "img/default-image.png"; // Default if no image exists
                image.alt = item.title;
                image.style.width = "40px";
                image.style.height = "40px";
                image.style.borderRadius = "5px";
                image.style.marginRight = "10px";

                // Create text container
                const text = document.createElement("span");
                text.textContent = item.title;
                text.style.color = "var(--green-house)";
                text.style.textAlign = "left"; // Ensure text aligns left

                listItem.appendChild(image);
                listItem.appendChild(text);
                listContainer.appendChild(listItem);
            });
        }

        overlayContent.appendChild(listContainer);

        // Add Close Button
        let closeButton = document.createElement("button");
        closeButton.className = "frame-67";
        closeButton.innerHTML = "&#10006;"; // "X" symbol
        closeButton.onclick = () => toggleOverlay(false);
        closeButton.style.position = "absolute";
        closeButton.style.top = "10px";
        closeButton.style.right = "10px";
        closeButton.style.background = "none";
        closeButton.style.border = "none";
        closeButton.style.fontSize = "18px";
        closeButton.style.cursor = "pointer";

        overlayContent.appendChild(closeButton);
    } catch (error) {
        console.error("Error fetching UPF items:", error);
    }
}

function renderUPFItemsOverlay(items) {
    const overlay = document.getElementById("product-comparison-overlay");
    const itemList = document.getElementById("upf-item-list");

    if (!itemList) {
        console.error("❌ ERROR: Element #upf-item-list not found in the DOM.");
        return;
    }

    itemList.innerHTML = ""; // Clear existing content

    if (items.length === 0) {
        itemList.innerHTML = "<p>No ultra-processed food items found.</p>";
    } else {
        items.forEach(item => {
            const listItem = document.createElement("li");
            listItem.textContent = item.title;
            itemList.appendChild(listItem);
        });
    }

    overlay.style.display = "flex";
}

// Add click event to Harmful Ingredients number
// Add click event to Harmful Ingredients number

// Function to fetch and display items with harmful ingredients
async function displayHarmfulItemsOverlay() {
    console.log("Fetching items with harmful ingredients...");

    try {
        const user = JSON.parse(localStorage.getItem("user"));
        if (!user) {
            console.error("User data not found in localStorage. Please log in.");
            return;
        }

        const userName = user.sub;

        // Fetch recyclable items (this is the correct API)
        const response = await fetch(
            "https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchRecyclables",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                },
                body: JSON.stringify({ user_name: userName })
            }
        );

        if (!response.ok) {
            console.error(`API call failed with status: ${response.status}`);
            return;
        }

        const data = await response.json();

        if (data.length === 0) {
            console.warn("No items found.");
            return;
        }

        // Filter items with harmful ingredients
        const harmfulItems = data.filter(item => item.harmful_ingredients && item.harmful_ingredients.trim() !== "");

        // Open the overlay before rendering content
        toggleOverlay(true, "harmful");

        // Get the overlay content container
        const overlayContent = document.querySelector(".product-comparison-1");
        overlayContent.innerHTML = ""; // Clear previous content

        // Add title
        const title = document.createElement("h2");
        title.textContent = "Harmful Ingredients";
        title.style.textAlign = "center";
        title.style.fontSize = "22px"; // Larger header
        title.style.fontWeight = "bold";
        title.className = "violetsans-regular-normal-green-house-18px";
        overlayContent.appendChild(title);

        // Create list container
        const listContainer = document.createElement("div");
        listContainer.style.padding = "10px";
        listContainer.style.marginTop = "10px";

        if (harmfulItems.length === 0) {
            const emptyMessage = document.createElement("p");
            emptyMessage.textContent = "No items with harmful ingredients found.";
            emptyMessage.style.textAlign = "center";
            overlayContent.appendChild(emptyMessage);
        } else {
            harmfulItems.forEach(item => {
                const listItem = document.createElement("div");
                listItem.style.display = "flex";
                listItem.style.flexDirection = "column";
                listItem.style.padding = "10px";
                listItem.style.borderBottom = "1px solid #ccc";
                listItem.style.fontFamily = "var(--font-family-violet_sans-regular)";
                listItem.style.fontSize = "18px";
                listItem.style.justifyContent = "flex-start"; // Align text to the left
                listItem.style.textAlign = "left"; // Ensure entire container aligns left

                // Create row for image + product title
                const productRow = document.createElement("div");
                productRow.style.display = "flex";
                productRow.style.alignItems = "center";
                productRow.style.marginBottom = "5px";
                productRow.style.textAlign = "left"; // Ensure row content is aligned left

                // Create image
                const image = document.createElement("img");
                image.src = item.images || "img/default-image.png"; // Default if no image exists
                image.alt = item.title;
                image.style.width = "40px";
                image.style.height = "40px";
                image.style.borderRadius = "5px";
                image.style.marginRight = "10px";

                // Create product title text
                const productText = document.createElement("span");
                productText.textContent = item.title;
                productText.style.color = "var(--green-house)";
                productText.style.textAlign = "left"; // Ensure text aligns left
                productText.style.flex = "1"; // Allow it to expand naturally

                productRow.appendChild(image);
                productRow.appendChild(productText);

                // Create harmful ingredients list
                const ingredientsText = document.createElement("p");
                ingredientsText.textContent = `❌ Harmful: ${item.harmful_ingredients}`;
                ingredientsText.style.fontSize = "14px";
                ingredientsText.style.color = "red";
                ingredientsText.style.marginTop = "5px";
                ingredientsText.style.marginLeft = "0"; // Remove indentation
                ingredientsText.style.textAlign = "left"; // Align ingredients to the left

                listItem.appendChild(productRow);
                listItem.appendChild(ingredientsText);
                listContainer.appendChild(listItem);
            });
        }

        overlayContent.appendChild(listContainer);

        // Add Close Button
        let closeButton = document.createElement("button");
        closeButton.className = "frame-67";
        closeButton.innerHTML = "&#10006;"; // "X" symbol
        closeButton.onclick = () => toggleOverlay(false);
        closeButton.style.position = "absolute";
        closeButton.style.top = "10px";
        closeButton.style.right = "10px";
        closeButton.style.background = "none";
        closeButton.style.border = "none";
        closeButton.style.fontSize = "18px";
        closeButton.style.cursor = "pointer";

        overlayContent.appendChild(closeButton);
    } catch (error) {
        console.error("Error fetching items with harmful ingredients:", error);
    }
}

function showSlotLoaders() {
  [0,1,2].forEach(i => {
    const bar = document.querySelector(`#recipe-${i} .ai-loader`);
    if (!bar) return;
    bar.classList.remove("hidden");
    console.log(`#recipe-${i} loader is`, getComputedStyle(bar).width,
            'wide – animation', getComputedStyle(bar).animationName);

    // force a reflow so the animation actually starts now
    // this reads a layout property and makes the browser commit the change
    // (without blocking user-visible pixels)
    void bar.offsetWidth;
  });
}


async function getAIResponse() {
  const userInput   = document.getElementById("user-input").value.trim();
  const responseBox = document.getElementById("chatbot-response");
  const apiUrl      = "https://1ix2pu4lt9.execute-api.us-east-1.amazonaws.com/RecipeGeneratorBot";

  if (!userInput) { alert("Type your goal first 🙂"); return; }
  const user = JSON.parse(localStorage.getItem("user"));
  if (!user)      { alert("Please log in first.");   return; }

  responseBox.innerHTML = "";
  responseBox.classList.add("hidden");

  // 1) show the little slot loaders
  clearRecipeSlots();  
  showSlotLoaders();

  responseBox.innerHTML = "";

  // 3) give the browser one tick to paint that loader bar
  await new Promise(r => requestAnimationFrame(r));
  startFancyLoader(responseBox); 

  try {
    const res = await fetch(apiUrl, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ user_id: user.sub, additional_request: userInput })
    });
    if (!res.ok) throw new Error(`API error ${res.status}`);
    const recipes = await res.json();

    stopFancyLoader();
    if (recipes.description) {
      responseBox.innerHTML = window.marked
          ? marked.parse(recipes.description)
          : recipes.description;
      responseBox.classList.remove("hidden");     // ⬅️ show it
    }

    ["dinner","lunch_snack","dessert"].forEach((key, idx) => {
      const bar  = document.querySelector(`#recipe-${idx} .ai-loader`);
      const slot = document.querySelector(`#recipe-${idx} .span2`);
      const html = marked.parse(recipes[key] || "");

      if (slot) {
        slot.innerHTML = html;
        slot.classList.add("loaded");
      }
      if (bar) bar.classList.add("hidden");
    });

  } catch(err) {
    stopFancyLoader();
    console.error("❌ getAIResponse error:", err);
    document.querySelectorAll(".ai-loader").forEach(b => b.classList.add("hidden"));
    responseBox.textContent = "Sorry—couldn’t fetch recipes right now.";
  } finally {
  }
}

document.querySelector(".harmful-number").addEventListener("click", async () => {
    await displayHarmfulItemsOverlay();
});

//document.querySelector(".frame-46").addEventListener("click", () => {
//    toggleOverlay(true, "scores");
//});

document.querySelector(".number").style.cursor = "pointer"; // optional, for hover feedback
document.querySelector(".number").addEventListener("click", () => {
    // same logic we had before, e.g. opening weekly scores overlay
    toggleOverlay(true, "scores");
});


document.querySelector(".upf-number").addEventListener("click", () => {
    toggleOverlay(true, "upf");
});

