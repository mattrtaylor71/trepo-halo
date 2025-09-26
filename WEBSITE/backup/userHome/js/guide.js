/**********************************************
 * GLOBAL STATE
 **********************************************/
let allItems = []; // We'll populate this once on page load
let allAlternates = {}; // Stores preloaded alternate items
let userShoppingList = {}; // Stores user's shopping list (item_id -> shopping_id)

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
    allItems = await fetchRecyclableData(userData.sub);
    console.log('Fetched items:', allItems);

    contentContainer.innerHTML = '';

    allItems.forEach(item => {
      const itemElement = createItemElement(item);
      contentContainer.appendChild(itemElement);
    });

  } catch (error) {
    console.error('Error fetching recyclables:', error);
    contentContainer.innerHTML = '<p>Failed to load items. Please try again later.</p>';
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


function createItemElement(item) {
  /*console.log("Creating element for item:", item);*/
  
  // Create the item container
  const itemDiv = createItemContainer(item);

  return itemDiv;
}

/**********************************************
 * CREATE ITEM CONTAINER
 **********************************************/
 function createItemContainer(item) {
   const score = item.points || 0;

   // Define colors based on score ranges
   let textColor = 'var(--green-house)';
   let backgroundColor = 'var(--reef)';

   if (score >= 15) {
     textColor = 'var(--reef)';
     backgroundColor = 'var(--green-house)';
   }

   const itemDiv = document.createElement('div');
   itemDiv.className = 'item';

   itemDiv.innerHTML = `
     <div class="item-content">
       <img class="image-6" src="${item.images}" alt="${item.title}">
       <div class="chobani-greek-yoghurt violetsans-regular-normal-green-house-14px">
         ${item.title}
       </div>
       <div class="frame-62-1 frame-62-3" style="background-color: ${backgroundColor};">
         <div class="number violetsans-regular-normal-green-house-14px" style="color: ${textColor};">
           ${score}
         </div>
       </div>
     </div>
   `;

   return itemDiv;
 }


/**********************************************
 * ONLOAD
 **********************************************/
window.onload = fetchRecyclables;
