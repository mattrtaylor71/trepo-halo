// False-confirmation guard: detectActionClaim maps a completed-action CLAIM to
// the domain whose tool must have run. Covers the P2 dish-drop broadening
// (Brad's 07-09 "Logged: …" shape), the P1 removal domain, and non-triggers
// (Q&A, "logged out", and App-Guide instructional text).
import { test } from "node:test";
import assert from "node:assert/strict";
import { detectActionClaim } from "../lib/device-assistant.mjs";

const domainOf = (s) => (detectActionClaim(s) || {}).domain || null;

test("dishes: Brad's exact 07-09 sentence triggers the guard", () => {
  assert.equal(domainOf("Logged: Two Eggo waffles and one shot of espresso."), "dishes");
});
test("dishes: food-noun + sentence-initial completed claims trigger", () => {
  for (const s of [
    "Logged two Eggo waffles and one shot of espresso.",
    "Okay, logged: oatmeal and coffee.",
    "I logged your breakfast.",
    "Done — logged your dinner.",
    "Saved: your lunch.",
  ]) assert.equal(domainOf(s), "dishes", `should trigger: ${s}`);
});
test("dishes: existing meal-word phrasings still trigger", () => {
  assert.equal(domainOf("I logged your lunch."), "dishes");
  assert.equal(domainOf("Your lunch has been logged."), "dishes");
});
test("dishes: non-triggers — Q&A, 'logged out', instructional (App Guide)", () => {
  for (const s of [
    "Eggo waffles have about 190 calories.",
    "Logged out of what?",
    "You can log a dish by tapping the Dish Log tab.",
    "To log a dish, tap Dish Log, then take a photo or use voice.",
    "Want me to log your breakfast?",
    "I can log that for you if you tell me what you ate.",
  ]) assert.equal(detectActionClaim(s), null, `should NOT trigger: ${s}`);
});
test("cross-domain: removal + shopping claims route to their own domains", () => {
  assert.equal(domainOf("I removed the cheese from your kitchen."), "kitchen_remove");
  assert.equal(domainOf("Added milk to your shopping list."), "shopping_add");
});
