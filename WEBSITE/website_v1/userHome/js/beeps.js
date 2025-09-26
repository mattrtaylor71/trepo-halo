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
    allItems.sort((a, b) => new Date(b._createdDate) - new Date(a._createdDate));
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

let userPoints = 0;

document.addEventListener("DOMContentLoaded", async () => {
  userPoints = await fetchAndDisplayPoints();
  console.log("User has points:", userPoints); // ✅ Should now print correctly

  document.querySelectorAll(".redeem-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const cost = parseInt(button.getAttribute("data-cost"));
      const prize = button.getAttribute("data-prize");

      if (userPoints >= cost) {
        const confirmRedemption = confirm(
          `Are you sure you want to redeem ${prize} for ${cost} points?\n\nThis will deduct points from your balance.`
        );

        if (confirmRedemption) {
          alert(`That feels good! Check your email for a ${prize}. Points will be deducted from your account.`);
          await notifyRedemption(prize, cost);
        }
      } else {
        alert("Not enough points to redeem this reward.");
      }

    });
  });
});


async function notifyRedemption(prize, cost) {
  const user = JSON.parse(localStorage.getItem("user"));

  const payload = {
    to: "matt@trepo.ai",
    subject: `New Redemption: ${prize}`,
    body: `User ${user?.email || 'Unknown'} redeemed ${prize} for ${cost} points.`
  };

  try {
    const res = await fetch("https://lr2vi6sf7f.execute-api.us-east-1.amazonaws.com/send-reward-email", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    console.log("✅ Email sent via API:", data);
  } catch (err) {
    console.error("❌ API request failed:", err);
  }
}




/**********************************************
 * ONLOAD
 **********************************************/
window.onload = fetchRecyclables;
