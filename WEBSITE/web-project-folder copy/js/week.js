// Fetch recyclable items from the backend API
async function fetchRecyclables() {
    const feed = document.getElementById('feed');
    const userData = JSON.parse(localStorage.getItem("user"));

    // Check if the user is logged in; if not, show "Please sign in!" message
    if (!userData || !userData.name) {
        document.getElementById("user-greeting").textContent = "Please sign in!";
        feed.innerHTML = ""; // Clear any previous content
        return;
    }

    // If user is logged in, display their name and fetch recyclables
    //document.getElementById("user-greeting").textContent = `Hello, ${userData.name}! Welcome to your week overview.`;
    feed.innerHTML = '<p>Loading items...</p>'; // Display loading message

    try {
        // Make a POST request to pass the user name to the backend
        const response = await fetch('https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchRecyclables', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_name: userData.sub }) // Pass user_name in the request body
        });

        console.log('Response status:', response.status);

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Data received from API:', data);

        feed.innerHTML = '';  // Clear loading message

        // Group items by date and simplified category
        const groupedItems = {};
        const categoryCounts = {};

        data.forEach(item => {
            if (item.inventory === '-1') {
                return; // Skip items with inventory '-1'
            }

            item.isHighlighted = item.inventory === '1'; // Mark item for highlighting

            const createdDate = new Date(item._createdDate);
            const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
            const dateString = createdDate.toLocaleDateString(undefined, options);
            
            if (!groupedItems[dateString]) {
                groupedItems[dateString] = [];
            }
            groupedItems[dateString].push(item);

            // Count items by simplified category
            categoryCounts[item.simplified_category] = (categoryCounts[item.simplified_category] || 0) + 1;
        });

        // Sort the dates in descending order
        const sortedDates = Object.keys(groupedItems).sort((a, b) => new Date(b) - new Date(a));

        // Create sections for each date
        sortedDates.forEach(date => {
            const section = document.createElement('div');
            section.className = 'day-section';

            const dayTitle = document.createElement('h2');
            dayTitle.className = 'day-title';
            dayTitle.textContent = date;
            section.appendChild(dayTitle);

            const grid = document.createElement('div');
            grid.className = 'feed-grid';

            groupedItems[date].forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'item';
                if (item.isHighlighted) {
                    itemDiv.classList.add('selected'); // Add green highlight class
                }

                const img = document.createElement('img');
                img.src = item.images;
                img.alt = item._id;

                const itemDetails = document.createElement('div');
                itemDetails.className = 'item-details';
                itemDetails.innerHTML = `<strong>${item.title}</strong><br><span class="brand">${item.brand}</span>`;

                const iconButton = document.createElement('button');
                iconButton.className = 'icon-button';
                iconButton.onclick = (event) => {
                    event.stopPropagation(); // Prevent triggering item click
                    itemDiv.classList.toggle('highlight'); // Toggle highlight on click
                };

                itemDiv.onclick = () => {
                    itemDiv.classList.toggle('selected');
                    if (itemDiv.classList.contains('highlight')) {
                        itemDiv.classList.remove('highlight'); // Remove red highlight
                    }
                };

                itemDiv.appendChild(img);
                itemDiv.appendChild(itemDetails);
                itemDiv.appendChild(iconButton);
                grid.appendChild(itemDiv);
            });

            section.appendChild(grid);
            feed.appendChild(section);
        });

        // Create Chart
        createCategoryChart(categoryCounts);

    } catch (error) {
        console.error('Error fetching recyclables:', error);
        feed.innerHTML = `<p>Failed to load items. Please try again later.</p>`;
    }
}

// Function to create category chart
function createCategoryChart(categoryCounts) {
    const ctx = document.getElementById('categoryChart').getContext('2d');
    const categories = Object.keys(categoryCounts);
    const counts = Object.values(categoryCounts);

    const chart = new Chart(ctx, {
        type: 'bar', 
        data: {
            labels: categories,
            datasets: [{
                label: 'Number of Items by Category',
                data: counts,
                backgroundColor: '#00B5E2',
                borderColor: '#00B5E2',
                borderWidth: 1,
                borderRadius: 10
            }]
        },
        options: {
            layout: {
                padding: {
                    top: 40 
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        display: false 
                    }
                },
                x: {
                    ticks: {
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                datalabels: {
                    anchor: 'end',
                    align: 'end',
                    color: 'black',
                    font: {
                        weight: 'bold',
                        size: 14
                    },
                    formatter: (value) => value
                }
            }
        },
        plugins: [ChartDataLabels]
    });
}

// Event listener for the submit button
document.getElementById('submitBtn').addEventListener('click', async () => {
    const itemsToUpdate = [];

    document.querySelectorAll('.item').forEach(item => {
        const itemId = item.querySelector('img').alt;
        let inventoryStatus;

        if (item.classList.contains('selected')) {
            inventoryStatus = "1"; 
        } else if (item.classList.contains('highlight')) {
            inventoryStatus = "-1"; 
        } else {
            inventoryStatus = "0";
        }

        itemsToUpdate.push({ id: itemId, inventory: inventoryStatus });
    });

    console.log('Items to update:', itemsToUpdate);

    // Retrieve user data from localStorage
    const userData = JSON.parse(localStorage.getItem("user"));

    try {
        const response = await fetch('https://g4trvf312e.execute-api.us-east-1.amazonaws.com/fetchRecyclables', {
            method: 'PUT', // Change this to PUT
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user_name: userData.sub, items: itemsToUpdate }) // Include user_name in the request body
        });

        const result = await response.json();
        console.log('Update response:', result);
        alert('Inventory updated successfully!');
    } catch (error) {
        console.error('Error updating inventory:', error);
        alert('Failed to update inventory. Please try again later.');
    }
});



// Fetch recyclables only if the user is authenticated and content is displayed
window.onload = () => {
    const contentContainer = document.getElementById("content-container");
    if (contentContainer && contentContainer.style.display !== "none") {
        fetchRecyclables();
    }
};
