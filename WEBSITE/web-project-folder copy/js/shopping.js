// Function to toggle the display of the alternative or original item
function handleItemClick(itemDiv, item, isAlternative) {
    // Scroll the tapped item to the center of the page
    itemDiv.scrollIntoView({
        behavior: 'smooth',  // Smooth scrolling animation
        block: 'center',     // Align to the center of the viewport
    });

    const existingAlternative = document.querySelector(`#alt-${item._id}`); // Check if the alternative box already exists

    // If an alternative is already shown, remove it when the item is clicked again
    if (existingAlternative) {
        existingAlternative.remove();
    } else {
        // Remove any existing alternative containers to ensure only one is shown at a time
        document.querySelectorAll('.alternative-container').forEach(container => container.remove());

        // Create the alternative box as a separate element
        const alternativeDiv = document.createElement('div');
        alternativeDiv.className = 'alternative-container';
        alternativeDiv.id = `alt-${item._id}`; // Add an ID to keep track of this alternative box

        if (!isAlternative) {
            // Use default styles for alternative: white background and blue border
            alternativeDiv.style.backgroundColor = '#ffffff';
            alternativeDiv.style.border = '2px solid #00b5e2';

            alternativeDiv.innerHTML = `
                <div class="prompt-text" style="color: #00b5e2;"><em>Want a healthier alternative?</em></div>
                <img src="${item.alt_image}" alt="${item.alt_title}">
                <div><strong>${item.alt_title}</strong></div>
                <div class="reasoning">${item.alt_reasoning}</div>
                <button class="select-item">Select Alternative</button>
            `;
            alternativeDiv.querySelector('.select-item').addEventListener('click', () => {
                itemDiv.querySelector('img').src = item.alt_image;
                itemDiv.querySelector('.item-details strong').textContent = item.alt_title;
                itemDiv.querySelector('.brand').style.display = 'none';
                itemDiv.classList.remove('item-expanded');
                alternativeDiv.remove();
                item.isAlternativeSelected = true;
            });
        } else {
            alternativeDiv.style.backgroundColor = '#ffffff';
            alternativeDiv.style.border = '2px solid #777';

            alternativeDiv.innerHTML = `
                <div class="prompt-text" style="color: #777;"><em>Want the original back?</em></div>
                <img src="${item.images}" alt="${item.title}">
                <div><strong>${item.title}</strong></div>
                <button class="select-item">Select Original</button>
            `;
            alternativeDiv.querySelector('.select-item').addEventListener('click', () => {
                itemDiv.querySelector('img').src = item.images;
                itemDiv.querySelector('.item-details strong').textContent = item.title;
                itemDiv.querySelector('.brand').style.display = '';
                itemDiv.classList.remove('item-expanded');
                alternativeDiv.remove();
                item.isAlternativeSelected = false;
            });
        }

        itemDiv.parentNode.insertBefore(alternativeDiv, itemDiv.nextSibling);
    }
}

// Function to handle swipe and remove items
function enableSwipeForItems(itemDiv) {
    let touchStartX = 0;

    itemDiv.addEventListener('touchstart', (e) => {
        touchStartX = e.touches[0].clientX;
    });

    itemDiv.addEventListener('touchend', (e) => {
        const touchEndX = e.changedTouches[0].clientX;
        const swipeThreshold = 100;

        if (touchStartX - touchEndX > swipeThreshold) {
            handleSwipe(itemDiv);
        }
    });
}

// Function to handle the actual swipe and remove animation
function handleSwipe(itemDiv) {
    itemDiv.classList.add('swipe-left');
    const itemId = itemDiv.dataset.id;

    itemDiv.addEventListener('animationend', async () => {
        itemDiv.remove();

        const userData = JSON.parse(localStorage.getItem("user"));
        if (!userData || !userData.sub) return;

        try {
            const response = await fetch('https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchRecyclables', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ user_name: userData.sub, items: [{ id: itemId, inventory: '2' }] })
            });

            if (!response.ok) {
                throw new Error(`Failed to update item. Status: ${response.status}`);
            }

            const result = await response.json();
            console.log('Swipe Update response:', result);
        } catch (error) {
            console.error('Error updating item on swipe:', error);
        }
    });
}

// Fetch shopping items from the backend API
async function fetchShoppingItems() {
    const feed = document.getElementById('feed');
    feed.innerHTML = '<p>Loading items...</p>';

    const userData = JSON.parse(localStorage.getItem("user"));
    if (!userData || !userData.sub) {
        document.getElementById("user-greeting").textContent = "Please sign in!";
        feed.innerHTML = "";
        return;
    }

    try {
        const response = await fetch('https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchRecyclables', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_name: userData.sub })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        feed.innerHTML = '';

        const itemsByCategory = {};
        data.forEach(item => {
            if (item.inventory === '1') {
                const category = item.simplified_category || 'Uncategorized';
                if (!itemsByCategory[category]) {
                    itemsByCategory[category] = [];
                }
                itemsByCategory[category].push(item);
            }
        });

        for (const [category, items] of Object.entries(itemsByCategory)) {
            const section = document.createElement('div');
            section.className = 'day-section';

            const categoryTitle = document.createElement('h2');
            categoryTitle.className = 'day-title';
            categoryTitle.textContent = category;
            section.appendChild(categoryTitle);

            const grid = document.createElement('div');
            grid.className = 'feed-grid';

            items.forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'item';
                itemDiv.dataset.id = item._id;

                const img = document.createElement('img');
                img.src = item.images;
                img.alt = item.title;

                const itemDetails = document.createElement('div');
                itemDetails.className = 'item-details';
                itemDetails.innerHTML = `<strong>${item.title}</strong><br><span class="brand">${item.brand}</span>`;

                itemDiv.appendChild(img);
                itemDiv.appendChild(itemDetails);
                grid.appendChild(itemDiv);

                enableSwipeForItems(itemDiv);
                itemDiv.addEventListener('click', () => handleItemClick(itemDiv, item, item.isAlternativeSelected || false));
            });

            section.appendChild(grid);
            feed.appendChild(section);
        }

        enableClearAllButton();

    } catch (error) {
        console.error('Error fetching shopping items:', error);
        feed.innerHTML = `<p>Failed to load items. Please try again later.</p>`;
    }
}

// Function to enable the "Clear All" button
function enableClearAllButton() {
    const clearAllBtn = document.getElementById('clearAllBtn');
    clearAllBtn.addEventListener('click', async () => {
        await updateAllInventoryToCleared();
    });
}

// Function to update inventory for all items and display feedback
async function updateAllInventoryToCleared() {
    const feed = document.getElementById('feed');
    const feedbackMessage = document.getElementById('feedbackMessage') || document.createElement('div');
    feedbackMessage.id = 'feedbackMessage';
    document.body.appendChild(feedbackMessage);

    const items = Array.from(feed.querySelectorAll('.item'));
    const userData = JSON.parse(localStorage.getItem("user"));
    if (!userData || !userData.sub) return;

    const updates = items.map(item => ({ id: item.dataset.id, inventory: '2' }));

    try {
        const response = await fetch('https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchRecyclables', {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user_name: userData.sub, items: updates })
        });

        const result = await response.json();
        console.log('Clear all response:', result);

        feed.innerHTML = '';
        feedbackMessage.textContent = "Shopping list has been cleared!";
        feedbackMessage.style.visibility = 'visible';
        feedbackMessage.style.color = 'green';
    } catch (error) {
        console.error('Error clearing all items:', error);
        feedbackMessage.textContent = "Failed to clear shopping list. Please try again.";
        feedbackMessage.style.visibility = 'visible';
        feedbackMessage.style.color = 'red';
    }
}

// Call the function when the page loads
window.onload = fetchShoppingItems;
