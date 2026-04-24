// auth.js — lightweight client-side auth for the hackathon demo.
// No real OAuth; we key per-user progress off the email address the user provides.
// Stored in localStorage as { email, name, provider, signedInAt }.
// Progress keys are namespaced: ic_completed:<email>

const Auth = (() => {
  const USER_KEY = 'ic_user';

  // Canned identities for the social buttons — lets judges flip between
  // "fresh new user" and "returning user with progress" instantly.
  const DEMO_IDENTITIES = {
    google:   { email: 'demo.google@asme.ai',   name: 'Google Demo',   provider: 'Google' },
    apple:    { email: 'demo.apple@asme.ai',    name: 'Apple Demo',    provider: 'Apple' },
    github:   { email: 'demo.github@asme.ai',   name: 'GitHub Demo',   provider: 'GitHub' },
  };

  function currentUser() {
    const s = localStorage.getItem(USER_KEY);
    if (!s) return null;
    try { return JSON.parse(s); } catch { return null; }
  }

  function isSignedIn() { return !!currentUser(); }

  function deriveName(email) {
    const local = (email || '').split('@')[0] || 'Engineer';
    return local.replace(/[._-]+/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  function signInWithEmail(email) {
    email = (email || '').trim().toLowerCase();
    if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      throw new Error('Please enter a valid email address.');
    }
    const user = { email, name: deriveName(email), provider: 'Email', signedInAt: Date.now() };
    localStorage.setItem(USER_KEY, JSON.stringify(user));
    return user;
  }

  function signInWithProvider(providerKey) {
    const id = DEMO_IDENTITIES[providerKey];
    if (!id) throw new Error('Unknown provider');
    const user = { ...id, signedInAt: Date.now() };
    localStorage.setItem(USER_KEY, JSON.stringify(user));
    return user;
  }

  function signOut() {
    localStorage.removeItem(USER_KEY);
  }

  function progressKey() {
    const u = currentUser();
    return u ? `ic_completed:${u.email}` : 'ic_completed';
  }

  function loadCompleted() {
    try { return JSON.parse(localStorage.getItem(progressKey()) || '[]'); }
    catch { return []; }
  }

  function saveCompleted(list) {
    localStorage.setItem(progressKey(), JSON.stringify(list));
  }

  return { currentUser, isSignedIn, signInWithEmail, signInWithProvider, signOut, progressKey, loadCompleted, saveCompleted };
})();

window.Auth = Auth;
