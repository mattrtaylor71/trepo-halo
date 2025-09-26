// Store JavaScript
document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize the application
    initApp();
    
    // Add smooth scrolling for all internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

function initApp() {
    // Initialize animations
    initAnimations();
    
    // Initialize interactive elements
    initInteractiveElements();
    
    // Initialize cart functionality
    initCart();
    
    // Initialize favorites system
    initFavorites();
    
    // Initialize smart swaps
    initSmartSwaps();
    
    // Initialize bundle kits
    initBundleKits();
    
    // Add scroll effects
    initScrollEffects();
}

// Animation System
function initAnimations() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe all animated elements
    document.querySelectorAll('.category-section, .product-card, .action-card, .swap-card, .kit-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
    
    // Floating animation for hero cards
    const floatingCards = document.querySelectorAll('.floating-card');
    floatingCards.forEach((card, index) => {
        card.style.animationDelay = `${index * 2}s`;
    });
}

// Interactive Elements
function initInteractiveElements() {
    // Add to cart buttons
    document.querySelectorAll('.btn-add').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            addToCart(this);
        });
    });
    
    // Heart buttons
    document.querySelectorAll('.btn-heart').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            toggleFavorite(this);
        });
    });
    
    // Action cards
    document.querySelectorAll('.action-card').forEach(card => {
        card.addEventListener('click', function() {
            handleActionCardClick(this);
        });
    });
    
    // Product cards
    document.querySelectorAll('.product-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            addHoverEffect(this);
        });
        
        card.addEventListener('mouseleave', function() {
            removeHoverEffect(this);
        });
    });
}

// Cart System
function initCart() {
    let cart = JSON.parse(localStorage.getItem('trepoCart')) || [];
    let cartCount = 0;
    
    // Create cart indicator
    const cartIndicator = document.createElement('div');
    cartIndicator.className = 'cart-indicator';
    cartIndicator.innerHTML = `
        <i class="fas fa-shopping-cart"></i>
        <span class="cart-count">0</span>
    `;
    cartIndicator.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 50%;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        cursor: pointer;
        z-index: 1000;
        transition: all 0.3s ease;
        display: none;
    `;
    
    document.body.appendChild(cartIndicator);
    
    // Update cart count
    function updateCartCount() {
        const countElement = cartIndicator.querySelector('.cart-count');
        countElement.textContent = cartCount;
        
        if (cartCount > 0) {
            cartIndicator.style.display = 'block';
            cartIndicator.style.animation = 'bounce 0.6s ease';
        } else {
            cartIndicator.style.display = 'none';
        }
    }
    
    // Add to cart function
    window.addToCart = function(button) {
        const productCard = button.closest('.product-card');
        const productName = productCard.querySelector('h4').textContent;
        const productImage = productCard.querySelector('.product-image img').src;
        const healthScore = productCard.querySelector('.health-score').textContent;
        const points = productCard.querySelector('.points').textContent;
        
        // Add to cart array
        cart.push({
            name: productName,
            image: productImage,
            healthScore: healthScore,
            points: points,
            timestamp: Date.now()
        });
        
        cartCount++;
        
        // Save to localStorage
        localStorage.setItem('trepoCart', JSON.stringify(cart));
        
        // Update UI
        updateCartCount();
        
        // Show success animation
        showSuccessAnimation(button);
        
        // Show notification
        showNotification(`${productName} added to cart!`, 'success');
    };
    
    // Cart indicator click
    cartIndicator.addEventListener('click', function() {
        showCartModal();
    });
    
    updateCartCount();
}

// Favorites System
function initFavorites() {
    let favorites = JSON.parse(localStorage.getItem('trepoFavorites')) || [];
    
    // Update heart buttons based on saved favorites
    document.querySelectorAll('.btn-heart').forEach(button => {
        const productCard = button.closest('.product-card');
        const productName = productCard.querySelector('h4').textContent;
        
        if (favorites.includes(productName)) {
            button.innerHTML = '<i class="fas fa-heart"></i>';
            button.style.color = '#ef4444';
            button.style.background = '#fee2e2';
        }
    });
    
    window.toggleFavorite = function(button) {
        const productCard = button.closest('.product-card');
        const productName = productCard.querySelector('h4').textContent;
        const icon = button.querySelector('i');
        
        if (favorites.includes(productName)) {
            // Remove from favorites
            favorites = favorites.filter(item => item !== productName);
            icon.className = 'far fa-heart';
            button.style.color = '#6b7280';
            button.style.background = '#f3f4f6';
            showNotification(`${productName} removed from favorites`, 'info');
        } else {
            // Add to favorites
            favorites.push(productName);
            icon.className = 'fas fa-heart';
            button.style.color = '#ef4444';
            button.style.background = '#fee2e2';
            showNotification(`${productName} added to favorites!`, 'success');
        }
        
        // Save to localStorage
        localStorage.setItem('trepoFavorites', JSON.stringify(favorites));
        
        // Add heart animation
        button.style.animation = 'heartBeat 0.6s ease';
        setTimeout(() => {
            button.style.animation = '';
        }, 600);
    };
}

// Smart Swaps
function initSmartSwaps() {
    document.querySelectorAll('.btn-swap').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            showSwapModal(this);
        });
    });
}

// Bundle Kits
function initBundleKits() {
    document.querySelectorAll('.btn-kit').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            addKitToCart(this);
        });
    });
}

// Scroll Effects
function initScrollEffects() {
    let lastScrollTop = 0;
    
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        // Parallax effect for hero section
        const hero = document.querySelector('.hero');
        if (hero) {
            const scrolled = scrollTop * 0.5;
            hero.style.transform = `translateY(${scrolled}px)`;
        }
        
        // Header background opacity
        const header = document.querySelector('.header');
        if (header) {
            const opacity = Math.min(scrollTop / 200, 0.95);
            header.style.background = `rgba(255, 255, 255, ${opacity})`;
        }
        
        lastScrollTop = scrollTop;
    });
}

// Utility Functions
function showSuccessAnimation(button) {
    button.style.transform = 'scale(1.1)';
    button.style.background = 'linear-gradient(135deg, #764ba2, #667eea)';
    
    setTimeout(() => {
        button.style.transform = 'scale(1)';
        button.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
    }, 200);
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? 'linear-gradient(135deg, #667eea, #764ba2)' : 
                     type === 'error' ? 'linear-gradient(135deg, #ef4444, #dc2626)' : 
                     'linear-gradient(135deg, #fbbf24, #f59e0b)'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        z-index: 10000;
        transform: translateX(400px);
        transition: transform 0.3s ease;
        font-weight: 600;
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(400px)';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

function addHoverEffect(card) {
    card.style.transform = 'translateY(-10px) scale(1.02)';
    card.style.boxShadow = '0 20px 40px rgba(0, 0, 0, 0.2)';
}

function removeHoverEffect(card) {
    card.style.transform = 'translateY(0) scale(1)';
    card.style.boxShadow = '0 5px 20px rgba(0, 0, 0, 0.1)';
}

function handleActionCardClick(card) {
    const action = card.querySelector('h3').textContent.toLowerCase();
    
    switch(action) {
        case 'shop store':
            addAllToCart();
            break;
        case 'favorites':
            showFavoritesModal();
            break;
        case 'smart swaps':
            scrollToSection('.smart-swaps');
            break;
        case 'eco picks':
            filterEcoPicks();
            break;
    }
}

function addAllToCart() {
    const productCards = document.querySelectorAll('.product-card:not(.swap)');
    let addedCount = 0;
    
    productCards.forEach((card, index) => {
        setTimeout(() => {
            const addButton = card.querySelector('.btn-add');
            if (addButton) {
                addToCart(addButton);
                addedCount++;
                
                if (addedCount === productCards.length) {
                    showNotification(`Added ${addedCount} items to your cart!`, 'success');
                }
            }
        }, index * 100);
    });
}

function showSwapModal(button) {
    const productCard = button.closest('.product-card');
    const productName = productCard.querySelector('h4').textContent;
    
    // Create modal
    const modal = document.createElement('div');
    modal.className = 'swap-modal';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>Smart Swap Available</h3>
                <button class="close-modal">&times;</button>
            </div>
            <div class="modal-body">
                <div class="swap-comparison">
                    <div class="current-product">
                        <h4>Current Choice</h4>
                        <p>${productName}</p>
                        <span class="score-low">Lower Health Score</span>
                    </div>
                    <div class="swap-arrow">
                        <i class="fas fa-arrow-right"></i>
                    </div>
                    <div class="better-product">
                        <h4>Better Choice</h4>
                        <p>Healthier Alternative</p>
                        <span class="score-high">Higher Health Score</span>
                    </div>
                </div>
                <div class="modal-actions">
                    <button class="btn-swap-confirm">View Swap Options</button>
                    <button class="btn-close">Keep Current</button>
                </div>
            </div>
        </div>
    `;
    
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
        opacity: 0;
        transition: opacity 0.3s ease;
    `;
    
    document.body.appendChild(modal);
    
    // Animate in
    setTimeout(() => {
        modal.style.opacity = '1';
    }, 100);
    
    // Close modal
    modal.querySelector('.close-modal').addEventListener('click', () => {
        modal.style.opacity = '0';
        setTimeout(() => {
            document.body.removeChild(modal);
        }, 300);
    });
    
    modal.querySelector('.btn-close').addEventListener('click', () => {
        modal.style.opacity = '0';
        setTimeout(() => {
            document.body.removeChild(modal);
        }, 300);
    });
}

function addKitToCart(button) {
    const kitCard = button.closest('.kit-card');
    const kitName = kitCard.querySelector('h3').textContent;
    
    showNotification(`${kitName} added to cart!`, 'success');
    
    // Add success animation
    button.style.transform = 'scale(1.05)';
    button.style.background = 'linear-gradient(135deg, #059669, #047857)';
    
    setTimeout(() => {
        button.style.transform = 'scale(1)';
        button.style.background = 'linear-gradient(135deg, #10b981, #059669)';
    }, 200);
}

function scrollToSection(selector) {
    const section = document.querySelector(selector);
    if (section) {
        section.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

function filterEcoPicks() {
    const productCards = document.querySelectorAll('.product-card');
    
    productCards.forEach(card => {
        const isEcoPick = card.querySelector('.product-badge')?.textContent === 'Eco Pick';
        
        if (isEcoPick) {
            card.style.opacity = '1';
            card.style.transform = 'scale(1.05)';
            card.style.border = '2px solid #667eea';
        } else {
            card.style.opacity = '0.3';
            card.style.transform = 'scale(0.95)';
        }
    });
    
    showNotification('Showing Eco Picks only', 'info');
    
    // Reset after 3 seconds
    setTimeout(() => {
        productCards.forEach(card => {
            card.style.opacity = '1';
            card.style.transform = 'scale(1)';
            card.style.border = '2px solid transparent';
        });
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes heartBeat {
        0% { transform: scale(1); }
        14% { transform: scale(1.3); }
        28% { transform: scale(1); }
        42% { transform: scale(1.3); }
        70% { transform: scale(1); }
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    .modal-content {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        max-width: 500px;
        width: 90%;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        transform: scale(0.9);
        transition: transform 0.3s ease;
    }
    
    .modal-content.show {
        transform: scale(1);
    }
    
    .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .close-modal {
        background: none;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        color: #6b7280;
    }
    
    .swap-comparison {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .modal-actions {
        display: flex;
        gap: 1rem;
    }
    
    .btn-swap-confirm, .btn-close {
        flex: 1;
        padding: 0.8rem;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .btn-swap-confirm {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    
    .btn-close {
        background: #f3f4f6;
        color: #6b7280;
    }
`;

document.head.appendChild(style); 