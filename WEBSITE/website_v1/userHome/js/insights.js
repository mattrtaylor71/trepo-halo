// insights.js

function shiny(text){
  return `<span class="loader-shimmer">${text}</span>`;
}

function renderRecipes(recipesJson) {
  [0,1,2].forEach(i => {
    const slot = document.querySelector(`#recipe-${i} .span2`);
    const html = recipesJson[`recipe${i}`] || "";
    if (slot) slot.innerHTML = marked.parse(html);
  });
}

function clearRecipeSlots () {
  [0,1,2].forEach(i => {
    const span   = document.querySelector(`#recipe-${i} .span2`);
    const loader = document.querySelector(`#recipe-${i} .ai-loader`);
    if (span)   span.innerHTML = "";
    if (loader) loader.classList.add("hidden");
  });
}

const loaderLines = [
  "Warming up the ovens",
  "Messaging Gordon Ramsay",
  "Negotiating with your taste buds",
  "Querying secret grandma recipes",
  "Counting macros on an abacus",
  "Bribing veggies to taste better",
  "Searching the spice multiverse",
  "Asking AI nutritionists for gossip",
  "Sprinkling extra dopamine",
  "Plating your culinary victory"
];

let loaderTimer = null;

function startFancyLoader(box){
  let idx = 0;
  box.innerHTML = shiny(loaderLines[idx]);
  box.classList.remove("hidden","fade-out");
  box.classList.add("fade-cycle");

  const swapText = () => {
    box.classList.add("fade-out");
    setTimeout(() => {
      idx = (idx + 1) % loaderLines.length;
      box.innerHTML = shiny(loaderLines[idx]);
      box.classList.remove("fade-out");
    }, 400);
  };

  loaderTimer = setInterval(swapText, 4000);
}

function stopFancyLoader(){
  if (loaderTimer){
    clearInterval(loaderTimer);
    loaderTimer = null;
  }
}

async function fetchInsights() {
  console.log("Fetching insights...");

  try {
    const user = JSON.parse(localStorage.getItem("user"));
    if (!user) { alert("Please log in to access insights."); return; }

    const resp = await fetch(
      `https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchInsights?web_id=${user.sub}`
    );
    if (!resp.ok) {
      console.error(`API error ${resp.status}`);
      return;
    }

    const data = await resp.json();

    // pick most recent record
    const latest = data.reduce((a,b) =>
      new Date(b._createdDate) > new Date(a._createdDate) ? b : a
    , data[0]);

    // render recipe slots if present
    let recipesJson = {};
    try { recipesJson = JSON.parse(latest.recipe || "{}"); }
    catch(e){ console.error("Invalid recipe JSON", e); }
    renderRecipes(recipesJson);

  } catch (e) {
    console.error("Error fetching insights:", e);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  fetchInsights();
});

function showSlotLoaders() {
  [0,1,2].forEach(i => {
    const bar = document.querySelector(`#recipe-${i} .ai-loader`);
    if (!bar) return;
    bar.classList.remove("hidden");
    void bar.offsetWidth; // force reflow
  });
}

async function getAIResponse() {
  const userInput   = document.getElementById("user-input")?.value.trim();
  const responseBox = document.getElementById("chatbot-response");
  if (!userInput) { alert("Type your goal first 🙂"); return; }
  const user = JSON.parse(localStorage.getItem("user"));
  if (!user)      { alert("Please log in first."); return; }

  responseBox.innerHTML = "";
  clearRecipeSlots();
  showSlotLoaders();
  await new Promise(r => requestAnimationFrame(r));
  startFancyLoader(responseBox);

  try {
    const res = await fetch("https://1ix2pu4lt9.execute-api.us-east-1.amazonaws.com/RecipeGeneratorBot", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ user_id: user.sub, additional_request: userInput })
    });
    if (!res.ok) throw new Error(`API error ${res.status}`);

    const recipes = await res.json();
    stopFancyLoader();

    if (recipes.description) {
      responseBox.innerHTML = marked.parse(recipes.description);
      responseBox.classList.remove("hidden");
    }

    ["dinner","lunch_snack","dessert"].forEach((key, idx) => {
      const bar  = document.querySelector(`#recipe-${idx} .ai-loader`);
      const slot = document.querySelector(`#recipe-${idx} .span2`);
      if (slot) slot.innerHTML = marked.parse(recipes[key] || "");
      if (bar)  bar.classList.add("hidden");
    });

  } catch(err) {
    stopFancyLoader();
    console.error("❌ getAIResponse error:", err);
    document.querySelectorAll(".ai-loader").forEach(b => b.classList.add("hidden"));
    responseBox.textContent = "Sorry—couldn’t fetch recipes right now.";
  }
}

document.getElementById("user-input-submit")?.addEventListener("click", getAIResponse);

// Removed all references to:
//  • document.querySelector(".number")
//  • document.querySelector(".upf-number")
//  • document.querySelector(".harmful-number")
//  • document.querySelector(".well-done-this-is-a")
//  • the old .frame-69 / .recipe-class elements
// Any event listeners on those deleted selectors have been stripped or guarded above.
