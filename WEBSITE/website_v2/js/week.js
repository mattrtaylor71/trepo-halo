/**********************************************
 * GLOBAL STATE
 **********************************************/
let allItems = []; // We'll populate this once on page load
let allAlternates = {}; // Stores preloaded alternate items
let userShoppingList = {}; // Stores user's shopping list (item_id -> shopping_id)


/**********************************************
 * HIGHLIGHT FUNCTION
 **********************************************/

async function preloadAllAlternates() {
  const allAltIds = new Set();

  // Collect all alternate IDs from `allItems`
  allItems.forEach(item => {
    if (item.alternates) {
      item.alternates.split(',').forEach(id => allAltIds.add(id.trim()));
    }
  });

  if (allAltIds.size === 0) return; // No alternates to fetch

  try {
    const response = await fetch('https://t3oy9jb08d.execute-api.us-east-1.amazonaws.com/getAlternates', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ alternates: Array.from(allAltIds) })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const alternates = await response.json();

    // Store alternates in a dictionary for quick lookup
    alternates.forEach(alt => {
      allAlternates[alt._id] = alt;
    });

    console.log("Preloaded alternates:", allAlternates);
  } catch (error) {
    console.error("Error preloading alternates:", error);
  }
}

 
function highlightAddedItems(data) {
    console.log("🔎 Running highlightAddedItems()...");

    // 1) Un-highlight every .add-to-cart-container on the page
    const allContainers = document.querySelectorAll('.add-to-cart-container');
    allContainers.forEach(container => {
        container.style.backgroundColor = 'var(--reef)';
        const addToCartText = container.querySelector('.add-to-cart');
        addToCartText.style.color = 'var(--green-house)';
        addToCartText.textContent = 'Add to List';
    });

    // 2) Highlight containers for items with inventory
    let highlightedCount = 0;
    data.forEach(item => {
        if (userShoppingList[item._id]) {
            const itemEl = document.querySelector(`.add-to-cart-container[data-item-id="${item._id}"]`);
            if (itemEl) {
                itemEl.style.backgroundColor = 'var(--green-house)';
                const textEl = itemEl.querySelector('.add-to-cart');
                textEl.style.color = 'var(--reef)';
                textEl.textContent = 'Added!';
                highlightedCount++;
            }
        }
    });

    // ✅ ALSO highlight alternates if they are in the shopping list
    Object.values(allAlternates).forEach(alt => {
        if (userShoppingList[alt._id]) {
            const altEl = document.querySelector(`.add-to-cart-container[data-item-id="${alt._id}"]`);
            if (altEl) {
                altEl.style.backgroundColor = 'var(--green-house)';
                const textEl = altEl.querySelector('.add-to-cart');
                textEl.style.color = 'var(--reef)';
                textEl.textContent = 'Added!';
                highlightedCount++;
            }
        }
    });

    console.log(`✅ Highlighted ${highlightedCount} items.`);
    updateShoppingListButton();
}



/**********************************************
 * UPDATE SHOPPING LIST BUTTON
 **********************************************/
function updateShoppingListButton() {
  console.log("🔄 Updating shopping list button...");

  const shoppingListButton = document.querySelector('.frame-104');
  const shoppingListText = shoppingListButton.querySelector('.submit');
  const addedItemsCount = Object.keys(userShoppingList).length;

  console.log(`🛒 Shopping list contains ${addedItemsCount} items.`);

  if (addedItemsCount > 0) {
    shoppingListButton.style.backgroundColor = 'var(--green-house)';
    shoppingListText.style.color = 'var(--reef)';
    shoppingListText.textContent = `View Shopping List (${addedItemsCount} items)`;
  } else {
    shoppingListButton.style.backgroundColor = 'var(--reef)';
    shoppingListText.style.color = 'var(--green-house)';
    shoppingListText.textContent = 'View Shopping List (0 items)';
  }
}

/**********************************************
 * FETCH + RENDER ENTRY POINT
 **********************************************/
 async function fetchRecyclables() {
   const contentContainer = document.getElementById('content-container');
   const userData = JSON.parse(localStorage.getItem('user'));

   if (!userData || !userData.name) {
     displaySignInMessage(contentContainer);
     return;
   }

   contentContainer.innerHTML = '<p>Loading items...</p>';

   try {
     // Fetch main recyclable items
     allItems = await fetchRecyclableData(userData.sub);
     
     // Fetch shopping list **before** updating UI
     await fetchShoppingList(userData.sub);

     // Fetch alternates **after** shopping list
     await preloadAllAlternates();

     contentContainer.innerHTML = '';

     const categoryPercentages = computeScaledPercentages(allItems);
     updateCategoryBars(categoryPercentages);

     const groupedItems = groupItemsByDate(allItems);
     const sortedDates = sortDatesDescending(groupedItems);
     displayGroupedItems(contentContainer, groupedItems, sortedDates);

     // ✅ Ensure we highlight items AFTER fetching everything
     highlightAddedItems(allItems);
   } catch (error) {
     console.error('Error fetching recyclables:', error);
     contentContainer.innerHTML = '<p>Failed to load items. Please try again later.</p>';
   }
 }

async function fetchShoppingList(userName) {
  console.log("🔄 Fetching user's shopping list...");
  try {
    const response = await fetch('https://j8uicd4my7.execute-api.us-east-1.amazonaws.com/getShopping', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_name: userName })
    });

    if (!response.ok) throw new Error(`❌ Error fetching shopping list. HTTP Status: ${response.status}`);

    const shoppingData = await response.json();
    console.log("✅ Fetched shopping list data:", shoppingData);

    // 🛠 FIX: Ensure we properly store shopping list data
    userShoppingList = {};  // Reset before filling

    shoppingData.forEach(item => {
      if (item.item) {
        userShoppingList[item.item] = { shopping_id: item._id };  
      } else {
        console.warn("⚠️ Skipping item with missing 'item_id':", item);
      }
    });

    console.log("🛒 Fixed shopping list object:", userShoppingList);

    // ✅ Ensure shopping list button updates
    updateShoppingListButton();
  } catch (error) {
    console.error('❌ Error fetching shopping list:', error);
  }
}



/**********************************************
 * FETCH HELPER
 **********************************************/
async function fetchRecyclableData(userName) {
  const response = await fetch(
    'https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchRecyclables',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      },
      body: JSON.stringify({ user_name: userName })
    }
  );
  if (!response.ok) {
    throw new Error(`HTTP error! Status: ${response.status}`);
  }
  return await response.json();
}

/**********************************************
 * DISPLAY HELPERS
 **********************************************/
function displaySignInMessage(container) {
  const userGreetingElement = document.getElementById('user-greeting');
  userGreetingElement.textContent = 'Please sign in!';
  container.innerHTML = ''; // Clear content
}

function computeScaledPercentages(data) {
  const categories = [
    'Beverages',
    'Protein',
    'Snacks',
    'Grains',
    'Dairy',
    'Cooking',
    'Produce',
    'Other'
  ];
  const categoryCounts = {};
  let totalItems = 0;

  // Initialize counts
  categories.forEach(cat => (categoryCounts[cat] = 0));

  // Tally items
  data.forEach(item => {
    const cat = item.simplified_category;
    if (categories.includes(cat)) {
      categoryCounts[cat]++;
      totalItems++;
    }
  });

  // Compute raw
  const rawPercentages = {};
  for (const c in categoryCounts) {
    rawPercentages[c] = categoryCounts[c] / totalItems;
  }

  // Scale so max is 100%
  const maxPercentage = Math.max(...Object.values(rawPercentages));
  const scaledPercentages = {};
  for (const c in rawPercentages) {
    scaledPercentages[c] = (rawPercentages[c] / maxPercentage) * 100;
  }

  return scaledPercentages;
}

function updateCategoryBars(categoryPercentages) {
  const categoryBars = {
    Dairy: document.querySelector('.frame-40'),
    Protein: document.querySelector('.frame-40-1'),
    Snacks: document.querySelector('.frame-40-4'),
    Grains: document.querySelector('.frame-40-3'),
    Beverages: document.querySelector('.frame-40-7'),
    Cooking: document.querySelector('.frame-40-6'),
    Produce: document.querySelector('.frame-40-2'),
    Other: document.querySelector('.frame-40-8')
  };

  for (const category in categoryPercentages) {
    const bar = categoryBars[category];
    if (bar) {
      bar.style.width = `${categoryPercentages[category]}%`;
    }
  }
}

function groupItemsByDate(data) {
  const groupedItems = {};
  data.forEach(item => {
    // skip inventory='-1'
    if (item.inventory === '-1') return;

    const dateObj = new Date(item._createdDate);
    const formattedDate = dateObj.toLocaleDateString('en-US', {
      weekday: 'long',
      month: 'long',
      day: 'numeric'
    });

    if (!groupedItems[formattedDate]) {
      groupedItems[formattedDate] = { dateObj, items: [] };
    }
    groupedItems[formattedDate].items.push(item);
  });
  return groupedItems;
}

function sortDatesDescending(groupedItems) {
  return Object.keys(groupedItems).sort(
    (a, b) => groupedItems[b].dateObj - groupedItems[a].dateObj
  );
}

function displayGroupedItems(container, groupedItems, sortedDates) {
  sortedDates.forEach((formattedDate, index) => {
    const section = document.createElement('div');
    section.className = `day-section violetsans-regular-normal-green-house-18px section-${index}`;
    section.textContent = formattedDate;

    const itemContainer = document.createElement('div');
    itemContainer.className = 'item-container';

    groupedItems[formattedDate].items.forEach(item => {
      const itemDiv = createItemElement(item);
      itemContainer.appendChild(itemDiv);
    });

    section.appendChild(itemContainer);
    container.appendChild(section);
  });
}

function createItemElement(item) {
  /*console.log("Creating element for item:", item);*/
  
  // Create the item container
  const itemDiv = createItemContainer(item);

  // Attach event listeners
  attachOverlayButtonEvent(itemDiv, item);
  attachAddToCartEvent(itemDiv, item);

  return itemDiv;
}

/**********************************************
 * CREATE ITEM CONTAINER
 **********************************************/
function createItemContainer(item) {
  const itemDiv = document.createElement('div');
  itemDiv.className = 'item';

  itemDiv.innerHTML = `
    <div class="flex-row flex">
      <img class="image-6" src="${item.images}" alt="${item.title}">
    </div>
    <div class="flex-row-1 flex-row-7">
      <div class="flex-col flex">
        <div class="chobani-greek-yoghurt violetsans-regular-normal-green-house-14px">
          ${item.title}
        </div>
      </div>
      <button class="frame-62-1 frame-62-3">
        <div class="number violetsans-regular-normal-green-house-14px">
          ${item.score || 0}
        </div>
      </button>
    </div>
    <div class="frame-46-1 frame-46-3 add-to-cart-container"
         data-item-id="${item._id}">
      <div class="add-to-cart violetsans-regular-normal-green-house-14px">
        Add to List
      </div>
    </div>
  `;

  return itemDiv;
}

/**********************************************
 * ATTACH OVERLAY BUTTON EVENT
 **********************************************/
function attachOverlayButtonEvent(itemDiv, item) {
  const overlayButton = itemDiv.querySelector('.frame-62-1');
  overlayButton.addEventListener('click', () => {
    // Ensure the latest item data is fetched
    const freshItem = allItems.find(x => x._id === item._id);
    toggleOverlay(true, freshItem);
  });
}

/**********************************************
 * ATTACH ADD TO CART EVENT
 **********************************************/
function attachAddToCartEvent(itemDiv, item) {
    const addToCartContainer = itemDiv.querySelector('.add-to-cart-container');
    const addToCartText = addToCartContainer.querySelector('.add-to-cart');

    addToCartContainer.addEventListener('click', async () => {
        const userData = JSON.parse(localStorage.getItem('user'));
        if (!userData || !userData.sub) {
            console.error('❌ User is not authenticated.');
            return;
        }

        const isCurrentlyAdded = userShoppingList[item._id] !== undefined;
        const action = isCurrentlyAdded ? "remove" : "add";
        const source = allAlternates[item._id] ? "whole_foods" : "main";

        // ✅ INSTANT UI UPDATE (Optimistic UI)
        if (action === "add") {
            userShoppingList[item._id] = "temp_id"; // Temporary ID for UI update
            addToCartContainer.style.backgroundColor = 'var(--green-house)';
            addToCartText.style.color = 'var(--reef)';
            addToCartText.textContent = 'Added!';
        } else {
            delete userShoppingList[item._id];
            addToCartContainer.style.backgroundColor = 'var(--reef)';
            addToCartText.style.color = 'var(--green-house)';
            addToCartText.textContent = 'Add to List';
        }

        try {
            const shoppingId = userShoppingList[item._id] || null;
            await updateShoppingList(userData.sub, item._id, action, source, shoppingId);

            // ✅ Confirm the backend update was successful
            if (action === "add") {
                userShoppingList[item._id] = shoppingId; // Replace temp_id with real ID
            }
        } catch (error) {
            console.error('❌ Error updating shopping list:', error);
            alert('Failed to update shopping list. Please try again.');

            // ✅ Revert UI if API call fails
            if (action === "add") {
                delete userShoppingList[item._id];
                addToCartContainer.style.backgroundColor = 'var(--reef)';
                addToCartText.style.color = 'var(--green-house)';
                addToCartText.textContent = 'Add to List';
            } else {
                userShoppingList[item._id] = "temp_id"; // Restore UI to previous state
                addToCartContainer.style.backgroundColor = 'var(--green-house)';
                addToCartText.style.color = 'var(--reef)';
                addToCartText.textContent = 'Added!';
            }
        }

        updateShoppingListButton(); // ✅ Update the shopping list count
    });
}

/**********************************************
 * UPDATE ADD-TO-CART UI
 **********************************************/
function updateAddToCartUI(container, textElement, inventoryState) {
  const isAdded = inventoryState === '1';
  container.style.backgroundColor = isAdded ? 'var(--green-house)' : 'var(--reef)';
  textElement.style.color = isAdded ? 'var(--reef)' : 'var(--green-house)';
  textElement.textContent = isAdded ? 'Added!' : 'Add to List';
}

/**********************************************
 * UPDATE ITEM INVENTORY VIA API
 **********************************************/
async function updateItemInventory(userName, itemId, newInventory) {
  const response = await fetch(
    'https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchRecyclables',
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_name: userName,
        items: [{ _id: itemId, inventory: newInventory }]
      })
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to update item. Status: ${response.status}`);
  }
  
  console.log(`Inventory updated for item ${itemId} to ${newInventory}`);
}

/**********************************************
 * UPDATE SHOPPING LIST API CALL
 **********************************************/
async function updateShoppingList(userName, itemId, action, source, shoppingId = null) {
  const body = {
      user_name: userName,
      item_id: itemId,
      action: action,
      source: source,
  };

  // ✅ Fix: Ensure `shopping_id` is sent as a STRING, not an object
  if (action === "remove" && shoppingId) {
      body.shopping_id = typeof shoppingId === "object" ? shoppingId.shopping_id : shoppingId;  
  }

  const response = await fetch(
    'https://j8uicd4my7.execute-api.us-east-1.amazonaws.com/getShopping',
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to update shopping list. Status: ${response.status}`);
  }

  const responseData = await response.json();
  console.log(`✅ Shopping list updated: ${action} ${itemId} from ${source}`);

  return responseData.shopping_id || null;
}


function attachAlternateCartEvent(altItemDiv, alt) {
    const addToCartContainer = altItemDiv.querySelector('.add-to-cart-container');
    const addToCartText = addToCartContainer.querySelector('.add-to-cart');

    addToCartContainer.addEventListener('click', async () => {
        const userData = JSON.parse(localStorage.getItem('user'));
        if (!userData || !userData.sub) {
            console.error('❌ User is not authenticated.');
            return;
        }

        const isCurrentlyAdded = userShoppingList[alt._id] !== undefined;
        const action = isCurrentlyAdded ? "remove" : "add";
        const source = "whole_foods"; // ✅ Always whole_foods for alternates

        try {
            const shoppingId = userShoppingList[alt._id] || null;
            await updateShoppingList(userData.sub, alt._id, action, source, shoppingId);

            if (action === "add") {
                userShoppingList[alt._id] = "temp_id"; // Optimistic UI update
            } else {
                delete userShoppingList[alt._id];
            }

            // ✅ Ensure UI updates after add/remove
            updateAddToCartUI(addToCartContainer, addToCartText, action === "add" ? "1" : "0");

            // ✅ Highlight all added items (including alternates)
            highlightAddedItems(allItems);
        } catch (error) {
            console.error('❌ Error updating shopping list:', error);
            alert('Failed to update shopping list. Please try again.');
        }
    });
}


/**********************************************
 * MODIFIED OVERLAY FUNCTION
 **********************************************/

async function toggleOverlay(show, item = null) {
    const overlay = document.getElementById('product-comparison-overlay');

    if (show && item) {
        console.log("🔍 Opening overlay for item:", item);

        // Select overlay elements
        const titleEl = overlay.querySelector('.graza-extra-virgin-olive-oil');
        const imageEl = overlay.querySelector('.image-7');
        const scoreEl = overlay.querySelector('.number');
        const ingredientsEl = overlay.querySelector('.ingredients-info');
        const harmfulIngredientsEl = overlay.querySelector('.harmful-ingredients-info');
        const overviewEl = overlay.querySelector('.this-product-is-made');
        const supermarketLogoContainer = overlay.querySelector('.supermarket-logo-container');
        const upfBadgeEl = overlay.querySelector('.frame-53');  // UPF Badge
        const upfTextEl = overlay.querySelector('.upf-number'); // UPF Text
        const alternatesRowEl = overlay.querySelector('.alternative-items-row'); // ✅ Alternates container

        // Set content
        if (titleEl) titleEl.textContent = item.title || 'Unknown Item';
        if (imageEl) imageEl.src = item.images || 'img/default-image.png';
        if (scoreEl) scoreEl.textContent = item.score || 'N/A';
        if (ingredientsEl) ingredientsEl.textContent = item.ingredients || 'No ingredients listed';
        if (overviewEl) overviewEl.textContent = item.score_reasoning || 'No details available';

        // ✅ Restore UPF Badge Functionality
        if (item.UPF && item.UPF.toLowerCase() === "yes") {
            upfBadgeEl.style.display = 'block';
            upfTextEl.textContent = 'UPF';
        } else {
            upfBadgeEl.style.display = 'none';
        }

        // ✅ Restore Red Font for Harmful Ingredients
        if (harmfulIngredientsEl) {
            harmfulIngredientsEl.textContent = item.harmful_ingredients || 'No harmful ingredients detected';
            harmfulIngredientsEl.style.color = item.harmful_ingredients ? 'red' : 'inherit';
        }

        // Clear previous recommendations
        supermarketLogoContainer.innerHTML = `
            <img class="supermarket-logo" src="img/Whole.png" alt="Whole Foods Logo">
        `;

        // ✅ Restore Alternates
        alternatesRowEl.innerHTML = ''; // Clear previous alternates

        if (item.alternates && allAlternates) {
            const altIds = item.alternates.split(',').map(id => id.trim());

            let alternateItems = altIds
                .map(altId => allAlternates[altId])
                .filter(alt => alt && alt.score !== undefined);

            // Sort by score (highest first)
            alternateItems.sort((a, b) => b.score - a.score);

            alternateItems.forEach(alt => {
                const altItemDiv = document.createElement('div');
                altItemDiv.className = 'item';

                const altImage = (alt.image_urls && alt.image_urls.split(',')[0].trim()) || 'img/default-image.png';

                altItemDiv.innerHTML = `
                    <img class="image-6" src="${altImage}" alt="${alt.name}">
                    <div class="chobani-greek-yoghurt violetsans-regular-normal-green-house-14px">
                        ${alt.name}
                    </div>
                    <div class="x500ml violetsans-regular-normal-green-house-12px">
                        ${alt.brand}
                    </div>
                    <button class="frame-62-1 frame-62-3">
                        <div class="number violetsans-regular-normal-green-house-14px">
                            ${alt.score || 0}
                        </div>
                    </button>
                    <div class="frame-46-1 frame-46-3 add-to-cart-container" data-item-id="${alt._id}">
                        <div class="add-to-cart violetsans-regular-normal-green-house-14px">
                            Add to List
                        </div>
                    </div>
                `;

                // Attach event listener for adding alternates to cart
                attachAlternateCartEvent(altItemDiv, alt);

                // ✅ Highlight added alternates
                const addToCartContainer = altItemDiv.querySelector('.add-to-cart-container');
                const addToCartText = addToCartContainer.querySelector('.add-to-cart');

                if (userShoppingList[alt._id]) {
                    addToCartContainer.style.backgroundColor = 'var(--green-house)';
                    addToCartText.style.color = 'var(--reef)';
                    addToCartText.textContent = 'Added!';
                }

                alternatesRowEl.appendChild(altItemDiv);
            });
        }

        // Show overlay
        overlay.style.display = 'flex';
    } else {
        console.log("❌ Closing overlay...");
        overlay.style.display = 'none';
    }
}

/*
async function toggleOverlay(show, item = null) {
    const overlay = document.getElementById('product-comparison-overlay');
    const alternativeItemsRow = overlay.querySelector('.alternative-items-row');

    if (show && item) {
        console.log("🔍 Opening overlay for item:", item);

        // Clear old alt items
        alternativeItemsRow.innerHTML = '';

        if (item.alternates && allAlternates) {
            const altIds = item.alternates.split(',').map(id => id.trim());

            let alternateItems = altIds
                .map(altId => allAlternates[altId])
                .filter(alt => alt && alt.score !== undefined);

            alternateItems.sort((a, b) => b.score - a.score);

            alternateItems.forEach(alt => {
                const altItemDiv = document.createElement('div');
                altItemDiv.className = 'item';

                const altImage = (alt.image_urls && alt.image_urls.split(',')[0].trim()) || 'img/default-image.png';

                altItemDiv.innerHTML = `
                    <img class="image-6" src="${altImage}" alt="${alt.name}">
                    <div class="chobani-greek-yoghurt violetsans-regular-normal-green-house-14px">
                        ${alt.name}
                    </div>
                    <div class="x500ml violetsans-regular-normal-green-house-12px">
                        ${alt.brand}
                    </div>
                    <button class="frame-62-1 frame-62-3">
                        <div class="number violetsans-regular-normal-green-house-14px">
                            ${alt.score || 0}
                        </div>
                    </button>
                    <div class="frame-46-1 frame-46-3 add-to-cart-container" data-item-id="${alt._id}">
                        <div class="add-to-cart violetsans-regular-normal-green-house-14px">
                            Add to List
                        </div>
                    </div>
                `;

                // Attach event listener
                attachAlternateCartEvent(altItemDiv, alt);

                // ✅ Ensure alternates highlight correctly when opening overlay
                const addToCartContainer = altItemDiv.querySelector('.add-to-cart-container');
                const addToCartText = addToCartContainer.querySelector('.add-to-cart');

                if (userShoppingList[alt._id]) {
                    addToCartContainer.style.backgroundColor = 'var(--green-house)';
                    addToCartText.style.color = 'var(--reef)';
                    addToCartText.textContent = 'Added!';
                }

                alternativeItemsRow.appendChild(altItemDiv);
            });
        }
    } else {
        console.log("❌ Closing overlay...");
    }

    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
}
*/

/**********************************************
 * ONLOAD
 **********************************************/
window.onload = fetchRecyclables;
