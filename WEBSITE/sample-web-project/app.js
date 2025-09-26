let auth0 = null;

// Initialize Auth0 client
async function configureClient() {
  auth0 = await createAuth0Client({
    domain: 'dev-7kg4syk6znb1wov7.us.auth0.com',
    client_id: 'Xgcbr87FzTiXsIgFSG4wQ2vgugnQfgqa',
    redirect_uri: window.location.origin
  });
}

// Process login state and display appropriate UI
async function updateUI() {
  const isAuthenticated = await auth0.isAuthenticated();

  document.getElementById("login").style.display = isAuthenticated ? "none" : "inline-block";
  document.getElementById("logout").style.display = isAuthenticated ? "inline-block" : "none";

  if (isAuthenticated) {
    const user = await auth0.getUser();
    document.getElementById("user-info").innerText = `Hello, ${user.name}`;
  } else {
    document.getElementById("user-info").innerText = "";
  }
}

// Login and logout functionality
async function login() {
  await auth0.loginWithRedirect();
}

async function logout() {
  auth0.logout({
    returnTo: window.location.origin
  });
}

// Handle the authentication callback
async function handleAuthCallback() {
  if (window.location.search.includes("code=") && window.location.search.includes("state=")) {
    await auth0.handleRedirectCallback();
    window.history.replaceState({}, document.title, "/");
  }
}

// Initialize and configure Auth0 client and UI
window.onload = async () => {
  await configureClient();

  // Handle redirect callback from Auth0
  await handleAuthCallback();

  // Set up login and logout buttons
  document.getElementById("login").addEventListener("click", login);
  document.getElementById("logout").addEventListener("click", logout);

  // Update UI based on the user's login state
  updateUI();
};
