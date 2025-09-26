
/**********************************************
 * WHOLE FOODS API SETTINGS
 **********************************************/
const wholeFoodsAPIBaseURL = "https://www.wholefoodsmarket.com/api/search";
const storeId = "10044"; // Replace with the desired Whole Foods store ID
let userShoppingList = {}; // Ensure this exists to avoid ReferenceError


/**********************************************
 * GLOBAL STATE
 **********************************************/
let allItems = []; // We'll store all fetched items here exactly once

/**********************************************
 * OVERLAY FUNCTIONS
 **********************************************/
function showOverlay(overlayId) {
  const overlay = document.getElementById(overlayId);
  if (overlay) {
    overlay.style.display = "flex"; // or "block"
  }
}

function hideOverlay(overlayId) {
  const overlay = document.getElementById(overlayId);
  if (overlay) {
    overlay.style.display = "none";
  }
}

/**********************************************
 * FETCH WHOLE FOODS ITEMS BASED ON USER INPUT
 **********************************************/
async function fetchWholeFoodsItems(query) {
  // Your actual API Gateway endpoint
  const apiUrl = "https://w9xb8ti6l6.execute-api.us-east-1.amazonaws.com/getWF";
  const encodedQuery = encodeURIComponent(query);

  try {
    // Fetch data from the API Gateway
    const response = await fetch(`${apiUrl}?q=${encodedQuery}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();

    // Ensure the response is an array before returning
    if (!Array.isArray(data)) {
      console.error("Unexpected API response format:", data);
      return [];
    }

    // Map and return the relevant product details
    return data.map((item) => ({
      name: item.name || "",
      brand: item.brand || "",
      imageThumbnail: item.image || "placeholder.jpg",
    }));
  } catch (error) {
    console.error("Error fetching Whole Foods items:", error);
    return [];
  }
}


/**********************************************
 * RENDER SEARCH RESULTS IN OVERLAY
 **********************************************/
function renderSearchResults(items) {
  const resultsContainer = document.createElement("div");
  resultsContainer.className = "results-container";

  if (items.length === 0) {
    resultsContainer.innerHTML = "<p>No items found. Try a different search term.</p>";
  } else {
    items.forEach((item) => {
      const itemElement = document.createElement("div");
      itemElement.className = "result-item";

      itemElement.innerHTML = `
        <div class="result-item-row">
          <img 
            src="${item.imageThumbnail || ''}" 
            alt="${item.name}" 
            class="result-item-image" 
          />
          <div class="result-item-info">
            <p class="result-item-name">${item.name}</p>
            <p class="result-item-brand">${item.brand || 'Unknown Brand'}</p>
          </div>
        </div>
      `;

      resultsContainer.appendChild(itemElement);
    });
  }

  const searchBoxContainer = document.querySelector(".search-box-container");
  if (searchBoxContainer) {
    // Clear previous results
    const existingResults = searchBoxContainer.querySelector(".results-container");
    if (existingResults) {
      searchBoxContainer.removeChild(existingResults);
    }

    // Append new results
    searchBoxContainer.appendChild(resultsContainer);
  }
}

/**********************************************
 * HANDLE ADD ITEM BUTTON CLICK
 **********************************************/

async function handleAddItemButtonClick() {
    const userInputElement = document.getElementById("newItemInput");
    if (!userInputElement) return;

    const userInput = userInputElement.value.trim();
    if (!userInput) {
        alert("Please enter an item name.");
        return;
    }

    const userData = JSON.parse(localStorage.getItem("user")) || {};
    if (!userData || !userData.sub) {
        alert("You need to be signed in.");
        return;
    }

    // ✅ Prepare request payload
    const requestBody = {
        user_name: userData.sub,
        item_id: userInput,
        source: "manual",
        action: "add"
    };

    try {
        const response = await fetch("https://j8uicd4my7.execute-api.us-east-1.amazonaws.com/getShopping", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody),
        });

        const result = await response.json();
        if (response.ok) {
            console.log("✅ Item added successfully:", result);
            fetchShoppingItems(); // 🔄 Refresh shopping list

            // ✅ Automatically close the overlay
            hideOverlay("shopping-list-search");
        } else {
            console.error("❌ Failed to add item:", result.error);
        }
    } catch (error) {
        console.error("❌ Error adding item:", error);
    }

    // ✅ Clear input box after adding item
    userInputElement.value = "";
}



async function fetchShoppingItems() {
  const contentContainer = document.getElementById("content-container");
  if (!contentContainer) {
    console.error("❌ Content container not found.");
    return;
  }

  allItems = [];
  contentContainer.innerHTML = "<p>Loading items...</p>";

  // ✅ Ensure userData is always retrieved
  const userData = JSON.parse(localStorage.getItem("user")) || {};
  if (!userData || !userData.sub) {
    console.error("🚨 userData is missing:", userData);
    contentContainer.innerHTML = "<p>Please sign in to view your shopping list.</p>";
    return;
  }

  try {
    const response = await fetch("https://hdtyg7j6d3.execute-api.us-east-1.amazonaws.com/fetchShopping", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_name: userData.sub }),
    });

    if (!response.ok) {
      throw new Error(`❌ HTTP error! Status: ${response.status}`);
    }

    const shoppingData = await response.json();
    
    // ✅ Ensure `action` field exists, defaulting to "1" if missing
    shoppingData.items.forEach((item) => {
      item.action = item.action || "1"; // If action is missing, default to "1"
    });

    allItems = [...shoppingData.items, ...shoppingData.whole_foods];

    console.log("🛒 Fetched Shopping Items with Actions:", allItems);

    contentContainer.innerHTML = "";
    const groupedItems = groupItemsByCategory(allItems);
    renderItemsByCategoryWithoutHeader(groupedItems, contentContainer);
    highlightAllButton();
  } catch (error) {
    console.error("❌ Error fetching shopping items:", error);
    contentContainer.innerHTML = "<p>Failed to load items. Please try again later.</p>";
  }
}


/**********************************************
 * GROUP ITEMS BY CATEGORY
 **********************************************/
function groupItemsByCategory(items) {
  return items.reduce((grouped, item) => {
    const category = item.simplified_category || "Other";
    if (!grouped[category]) grouped[category] = [];
    grouped[category].push(item);
    return grouped;
  }, {});
}

/**********************************************
 * RENDER ITEMS WITHOUT HEADERS
 **********************************************/
function renderItemsByCategoryWithoutHeader(groupedItems, container) {
  // We'll create a container for each category and append items to it
  Object.values(groupedItems).forEach((items) => {
    const itemsContainer = document.createElement("div");
    itemsContainer.className = "items-container";

    items.forEach((item) => {
      const itemElement = createItemElement(item);
      itemsContainer.appendChild(itemElement);
    });

    container.appendChild(itemsContainer);
  });
}

/**********************************************
 * CREATE ITEM ELEMENT
 **********************************************/
function createItemElement(item) {
  const isWholeFoods = item.source === "whole_foods";
  const displayTitle = item.title || item.name;
  const displayBrand = item.brand || "";
  const displayImg = item.source === "manual" ? "data:image/gif;base64,R0lGODlhAQABAIAAAAUEBA==" : 
      (item.images ? item.images.split(",")[0].trim() : item.image || "placeholder.jpg");

  const highlightClass = item.action === "2" ? "highlight" : "";
  const toggleActive = item.action === "2" ? "active" : "";
  const textColorClass = item.action === "2" ? "grey-text" : "";

  const itemDiv = document.createElement("div");
  itemDiv.className = `item ${highlightClass}`;
  itemDiv.dataset.id = item._id;

  itemDiv.innerHTML = `
      <div class="frame-1">
        <div class="flex-row">
          ${displayImg ? `<img class="image-6 ${item.source === "manual" ? 'invisible-img' : ''}" src="${displayImg}" alt="${displayTitle}">` : ""}
          <div class="flex-col-7 flex-col-10">
            <div class="item-title violetsans-regular-normal-green-house-16px ${textColorClass}">
              ${displayTitle}
            </div>
            <div class="item-brand violetsans-regular-normal-green-house-12px ${textColorClass}">
              ${displayBrand} ${isWholeFoods ? "(Whole Foods)" : ""}
            </div>
          </div>
          <div class="frame-59-2 frame-59-5 toggle-btn ${toggleActive}" data-id="${item._id}">
            <div class="rectangle-2 rectangle"></div>
            <div class="rectangle-1 rectangle"></div>
          </div>
        </div>
      </div>
  `;

  addSwipeListener(itemDiv);
  return itemDiv;
}


function addSwipeListener(element) {
  let startX = 0;
  let startTime = 0;
  let isSwiping = false;

  element.addEventListener("touchstart", (e) => {
    startX = e.touches[0].clientX;
    startTime = new Date().getTime();
    isSwiping = false;
  });

  element.addEventListener("touchmove", (e) => {
    const deltaX = e.touches[0].clientX - startX;

    if (deltaX < -50) { // Ensure significant left swipe
      isSwiping = true;
      element.style.transform = `translateX(${deltaX}px)`;
    }
  });

  element.addEventListener("touchend", () => {
    const endTime = new Date().getTime();
    const timeDiff = endTime - startTime;

    if (isSwiping && timeDiff < 500) { 
      // Register swipe only if it's a quick left swipe
      element.style.transition = "transform 0.3s ease-out, opacity 0.3s ease-out";
      element.style.transform = "translateX(-100%)";
      element.style.opacity = "0";

      setTimeout(() => {
        console.log(`Element dataset: ${element.dataset.id}`);
        updateInventory(element.dataset.id, "0"); // Update the backend
        element.remove(); // Remove the item from the UI
      }, 300);
    } else {
      // If it's not a swipe, reset position
      element.style.transition = "transform 0.2s ease-in-out";
      element.style.transform = "translateX(0)";
    }
  });

  // Support Mouse Drag on Desktop
  element.addEventListener("mousedown", (e) => {
    startX = e.clientX;
    isSwiping = false;
  });

  element.addEventListener("mousemove", (e) => {
    const deltaX = e.clientX - startX;
    if (deltaX < -50) {
      isSwiping = true;
      element.style.transform = `translateX(${deltaX}px)`;
    }
  });

  element.addEventListener("mouseup", () => {
    if (isSwiping) {
      element.style.transition = "transform 0.3s ease-out, opacity 0.3s ease-out";
      element.style.transform = "translateX(-100%)";
      element.style.opacity = "0";

      setTimeout(() => {
        console.log(`Element dataset: ${element.dataset.id}`);
        updateInventory(element.dataset.id, "0");
        element.remove();
      }, 300);
    } else {
      element.style.transition = "transform 0.2s ease-in-out";
      element.style.transform = "translateX(0)";
    }
  });
}



function getToggledInventory(current) {
  const isCurrentlyChecked = current.endsWith("_2") || current === "2";

  // If it's already checked
  if (isCurrentlyChecked) {
    // Remove trailing _2 or revert '2' to '1'
    if (current === "2") return "1";
    if (current.endsWith("_2")) {
      return current.slice(0, -2);
    }
  } else {
    // If it's not checked yet, add _2 or switch '1' to '2'
    if (current === "1") return "2";
    if (current.startsWith("wf_") && !current.endsWith("_2")) {
      return current + "_2";
    }
  }

  // Fallback if we missed a case
  return current;
}

async function toggleButton(button, itemId) {
  let textElements = [];
  const userData = JSON.parse(localStorage.getItem("user")) || {};

  if (!userData || !userData.sub) {
    console.error("🚨 userData is missing:", userData);
    return;
  }

  const shoppingItem = allItems.find((i) => i._id === itemId);
  if (!shoppingItem || !shoppingItem.shopping_id) {
    console.error(`❌ No shopping list _id found for product ${itemId}`);
    return;
  }

  const shoppingId = shoppingItem.shopping_id;
  const newAction = shoppingItem.action === "2" ? "1" : "2"; // Toggle between checked and unchecked

  // ✅ IMMEDIATE UI UPDATE BEFORE API REQUEST
  button.classList.toggle("active", newAction === "2");

  const parentItem = button.closest(".item");
  if (parentItem) {
    parentItem.classList.toggle("highlight", newAction === "2");

    // ✅ Update text color instantly
    const textElements = parentItem.querySelectorAll(".item-title, .item-brand");
    textElements.forEach((el) => {
      el.classList.toggle("grey-text", newAction === "2");
    });
  }

  // ✅ Update global state immediately so UI remains consistent
  shoppingItem.action = newAction;

  // 🔄 SEND API REQUEST IN THE BACKGROUND
  try {
    const requestBody = {
        user_name: userData.sub,  // ✅ Ensure user_name is set
        shopping_id: shoppingItem.shopping_id || null,  // ✅ Ensure shopping_id is sent
        item_id: shoppingItem._id || null,  // ✅ Ensure item_id is sent
        action: newAction,  // ✅ Ensure '1' or '2' is sent
        source: shoppingItem.source || "main"  // ✅ Ensure source is sent
    };

    console.log("🔹 Sending update request:", JSON.stringify(requestBody, null, 2));


    const response = await fetch(
      "https://j8uicd4my7.execute-api.us-east-1.amazonaws.com/getShopping",
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      }
    );

    if (!response.ok) {
      throw new Error(`❌ Failed to update shopping list. Status: ${response.status}`);
    }

    console.log(`✅ Shopping list successfully updated: ${newAction} ${shoppingId} from ${shoppingItem.source}`);
  } catch (error) {
    console.error("❌ Error updating shopping list:", error);

    // ❌ Revert UI changes if API call fails
    button.classList.toggle("active", shoppingItem.action !== "2");
    if (parentItem) {
      parentItem.classList.toggle("highlight", shoppingItem.action !== "2");
      textElements.forEach((el) => {
        el.classList.toggle("grey-text", shoppingItem.action !== "2");
      });
    }
  }
}


async function updateShoppingList(userName, shoppingId, action, source) {
  try {
    const response = await fetch(
      "https://j8uicd4my7.execute-api.us-east-1.amazonaws.com/getShopping",
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_name: userName, shopping_id: shoppingId, action, source }),
      }
    );

    if (!response.ok) {
      throw new Error(`❌ Failed to update shopping list. Status: ${response.status}`);
    }

    console.log(`🛒 Shopping list updated: ${action} ${shoppingId} from ${source}`);
  } catch (error) {
    console.error("❌ Error updating shopping list:", error);
  }
}


/**********************************************
 * EVENT DELEGATION & DOMContentLoaded
 **********************************************/

document.addEventListener("DOMContentLoaded", function () {
  const button = document.querySelector(".frame-4");

  if (button) {
    button.addEventListener("touchstart", function () {
      button.classList.add("pressed");
    });

    button.addEventListener("touchend", function () {
      setTimeout(() => button.classList.remove("pressed"), 150);
    });

    button.addEventListener("mousedown", function () {
      button.classList.add("pressed");
    });

    button.addEventListener("mouseup", function () {
      setTimeout(() => button.classList.remove("pressed"), 150);
    });
  }
});

document.addEventListener("DOMContentLoaded", () => {
  // 1) Toggle checkboxes in one container-level listener
  const orderButton = document.querySelector(".frame-4");
  if (orderButton) {
    orderButton.addEventListener("click", createInstacartShoppingList);
  }
  const contentContainer = document.getElementById("content-container");
  if (contentContainer) {
    contentContainer.addEventListener("click", (e) => {
      const btn = e.target.closest(".toggle-btn");
      if (btn) {
        const itemId = btn.getAttribute("data-id");
        toggleButton(btn, itemId);
      }
    });
  const addItemButton = document.getElementById("addItemButton");
  if (addItemButton) {
      addItemButton.addEventListener("click", handleAddItemButtonClick);
  }
  }

  if (archiveListButton) {
    archiveListButton.addEventListener("click", async () => {
      console.log("🗑 Clearing shopping list...");

      const userData = JSON.parse(localStorage.getItem("user")) || {};
      if (!userData || !userData.sub) {
        console.error("🚨 User data is missing.");
        return;
      }

      if (allItems.length === 0) {
        console.log("🛒 No items to clear.");
        return;
      }

      // ✅ STEP 1: Remove all items from the UI instantly
      const contentContainer = document.getElementById("content-container");
      if (contentContainer) contentContainer.innerHTML = "";

      // ✅ STEP 2: Clear the `allItems` array to prevent reloading old items
      allItems = [];

      // ✅ STEP 3: Send a single API request to clear all items in the backend
      try {
        const response = await fetch(
          "https://j8uicd4my7.execute-api.us-east-1.amazonaws.com/getShopping",
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              user_name: userData.sub,
              action: "clear_all", // Send a new "clear_all" action to the backend
            }),
          }
        );

        if (!response.ok) {
          throw new Error(`❌ Failed to clear shopping list. Status: ${response.status}`);
        }

        console.log("✅ Shopping list cleared successfully.");
      } catch (error) {
        console.error("❌ Error clearing shopping list:", error);
      }
    });
  } else {
    console.error("❌ archiveListButton not found in the DOM.");
  }
  // 3) CATEGORY FILTERS
  const buttons = document.querySelectorAll(".frame");
  function displayFilteredCategory(category) {
    contentContainer.innerHTML = "";
    if (category === "All") {
      const grouped = groupItemsByCategory(allItems);
      renderItemsByCategoryWithoutHeader(grouped, contentContainer);
    } else {
      const subset = allItems.filter(
        (it) => it.simplified_category === category
      );
      const grouped = groupItemsByCategory(subset);
      renderItemsByCategoryWithoutHeader(grouped, contentContainer);
    }
  }

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      // Deactivate all
      buttons.forEach((b) => {
        b.classList.remove("active");
        const textEl = b.querySelector("div");
        if (textEl) {
          textEl.style.color = "var(--green-house)";
        }
      });

      // Activate this button
      button.classList.add("active");
      const textEl = button.querySelector("div");
      if (textEl) {
        textEl.style.color = "var(--reef)";
      }

      // Filter & re-render
      const category = button.textContent.trim();
      displayFilteredCategory(category);
    });
  });

  // 4) OVERLAY BUTTONS: "openShoppingListSearchOverlay" & "closeShoppingListSearchOverlay"
  const openOverlayBtn = document.getElementById("openShoppingListSearchOverlay");
  if (openOverlayBtn) {
    openOverlayBtn.addEventListener("click", () => {
      showOverlay("shopping-list-search");
    });
  }

  const closeOverlayBtn = document.getElementById("closeShoppingListSearchOverlay");
  if (closeOverlayBtn) {
    closeOverlayBtn.addEventListener("click", () => {
      hideOverlay("shopping-list-search");
    });
  }

  // 4b) If you have a text entry inside your overlay, handle the "Add Item" button:
  const addItemButton = document.getElementById("addItemButton"); // optional
  if (addItemButton) {
    addItemButton.addEventListener("click", () => {
      const userInput = document.getElementById("newItemInput");
      if (!userInput) return;

      const newItemName = userInput.value.trim();
      if (newItemName) {
        console.log("User typed:", newItemName);
        // TODO: Implement logic to add item to the user's list, if desired.
        // e.g.: addNewItemToShoppingList(newItemName);

        // Clear the field
        userInput.value = "";

        // Optionally hide overlay
        //hideOverlay("shopping-list-search");
      }
    });
  }

  // 5) Finally, fetch the items once the DOM is loaded
  fetchShoppingItems();
});

function removeItemFromUI(itemId) {
  const itemElement = document.querySelector(`[data-id='${itemId}']`);
  if (itemElement) {
    itemElement.style.transition = "transform 0.3s ease-out, opacity 0.3s ease-out";
    itemElement.style.transform = "translateX(-100%)";
    itemElement.style.opacity = "0";

    setTimeout(() => {
      itemElement.remove();
    }, 300);
  }
}

async function updateInventory(itemId, source) {
  const userData = JSON.parse(localStorage.getItem("user")) || {};

  if (!userData || !userData.sub) {
    console.error("🚨 userData is missing:", userData);
    return;
  }

  // 🔍 Find the correct shopping ID for the given item
  const shoppingItem = allItems.find((item) => item._id === itemId);

  if (!shoppingItem || !shoppingItem.shopping_id) {
    console.error(`❌ No shopping list ID found for item: ${itemId}`);
    return;
  }

  const shoppingId = shoppingItem.shopping_id; // ✅ Now we have the correct ID

  const requestBody = {
    user_name: userData.sub,
    shopping_id: shoppingId, // ✅ Ensure the correct shopping ID is included
    action: "remove", // Use "remove" as the action
    item_id: itemId, // The actual item ID to be removed
    source: source || "main", // Ensure source is always present
  };

  console.log("🗑️ Sending REMOVE request:", JSON.stringify(requestBody, null, 2));

  try {
    const response = await fetch(
      "https://j8uicd4my7.execute-api.us-east-1.amazonaws.com/getShopping",
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      }
    );

    const responseText = await response.text();
    console.log(`📩 API Response Status: ${response.status}`);
    console.log("📩 API Response Body:", responseText);

    if (!response.ok) {
      throw new Error(`❌ Failed to remove item. Status: ${response.status}, Response: ${responseText}`);
    }

    console.log(`✅ Item ${itemId} removed successfully.`);

    // ✅ Remove the item from global state so it doesn't reappear
    allItems = allItems.filter(item => item._id !== itemId);

    // ✅ Remove item from UI
    removeItemFromUI(itemId);

  } catch (error) {
    console.error("❌ Error removing item:", error);
  }
}

async function createInstacartShoppingList() {
    if (!allItems || allItems.length === 0) {
        alert("Your shopping list is empty.");
        return;
    }

    // Use simplified_title if available, otherwise fallback to title
    const itemNames = allItems
        .map(item => item.simplified_title && item.simplified_title.trim().length > 0 
            ? item.simplified_title.trim() 
            : item.title && item.title.trim().length > 0 
                ? item.title.trim() 
                : null
        )
        .filter(name => name !== null); // Remove any null values

    console.log("📦 Items to send to Instacart:", itemNames);

    if (itemNames.length === 0) {
        alert("Your shopping list contains no valid items.");
        return;
    }

    try {
        const response = await fetch("https://wpphq7xnk0.execute-api.us-east-1.amazonaws.com/getListURL", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ items: itemNames }),
        });

        const data = await response.json();

        if (data.shopping_list_url) {
            console.log("✅ Instacart URL received:", data.shopping_list_url);

            // Attempt to open the link in a new tab
            const newTab = window.open(data.shopping_list_url, "_blank");

            if (!newTab || newTab.closed || typeof newTab.closed === "undefined") {
                console.warn("❌ Popup blocked. Using fallback method.");

                // Create a hidden anchor element as a fallback
                const instacartLink = document.createElement("a");
                instacartLink.href = data.shopping_list_url;
                instacartLink.target = "_blank";
                instacartLink.rel = "noopener noreferrer";
                instacartLink.style.display = "none";
                document.body.appendChild(instacartLink);
                instacartLink.click();
                document.body.removeChild(instacartLink);
            }
        } else {
            alert("Failed to create Instacart shopping list. Try again.");
        }
    } catch (error) {
        console.error("❌ Error creating Instacart shopping list:", error);
        alert("Error creating Instacart shopping list.");
    }
}


/**********************************************
 * HELPER: HIGHLIGHT "ALL" BUTTON BY DEFAULT
 **********************************************/
function highlightAllButton() {
  const buttons = document.querySelectorAll(".frame");
  // Remove active from all first
  buttons.forEach((b) => {
    b.classList.remove("active");
    const textEl = b.querySelector("div");
    if (textEl) {
      textEl.style.color = "var(--green-house)";
    }
  });

  // Find the "All" button
  const allBtn = Array.from(buttons).find(
    (b) => b.textContent.trim() === "All"
  );
  if (allBtn) {
    allBtn.classList.add("active");
    const txt = allBtn.querySelector("div");
    if (txt) {
      txt.style.color = "var(--reef)";
    }
  }
}

document.addEventListener("DOMContentLoaded", function () {
  const archiveButton = document.getElementById("archiveListButton");

  if (archiveButton) {
    archiveButton.addEventListener("touchstart", function () {
      archiveButton.classList.add("pressed");
    });

    archiveButton.addEventListener("touchend", function () {
      setTimeout(() => archiveButton.classList.remove("pressed"), 150);
    });

    archiveButton.addEventListener("mousedown", function () {
      archiveButton.classList.add("pressed");
    });

    archiveButton.addEventListener("mouseup", function () {
      setTimeout(() => archiveButton.classList.remove("pressed"), 150);
    });
  }
});


/**********************************************
 * WINDOW ONLOAD
 **********************************************/
window.onload = () => {
  // Optionally call fetchShoppingItems() here as well
  // if you prefer window.onload over DOMContentLoaded.
};




