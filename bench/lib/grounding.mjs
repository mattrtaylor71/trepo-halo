// Programmatic inventory-grounding check for recipe benchmark. An ingredient is
// "grounded" if it matches a kitchen product (shared significant token, singularized)
// or is a common pantry staple. Approximation of the app's matcher — good enough to
// compare models on how well they stick to what's actually in the kitchen.
const PANTRY = new Set(["salt","pepper","oil","olive oil","vegetable oil","water","sugar","flour",
  "butter","garlic","onion","spice","spices","seasoning","black pepper","baking soda","baking powder",
  "vinegar","honey","soy sauce","rice","pasta","stock","broth","egg","eggs","milk","cornstarch",
  "chili powder","cumin","paprika","oregano","basil","cinnamon","vanilla","ketchup","mustard","mayo","mayonnaise"]);
const STOP = new Set(["and","the","with","of","in","a","fresh","dried","ground","chopped","sliced","whole",
  "large","small","medium","organic","raw","cooked","to","taste","for","cup","cups","tbsp","tsp","oz","lb",
  "boneless","skinless","extra","virgin","can","cans","package","piece","pieces"]);

function singular(t){ t=t.toLowerCase(); if(t.length<=3)return t;
  if(t.endsWith("ies")&&t.length>4)return t.slice(0,-3)+"y";
  if(t.endsWith("oes")&&t.length>4)return t.slice(0,-2);
  if(t.endsWith("s")&&!t.endsWith("ss"))return t.slice(0,-1); return t; }
function tokens(s){ return String(s||"").toLowerCase().replace(/\([^)]*\)/g," ").replace(/[^a-z0-9\s]/g," ")
  .split(/\s+/).map(singular).filter(t=>t&&t.length>2&&!STOP.has(t)); }

export function buildKitchenIndex(items){
  const toks = new Set();
  for(const it of items){ for(const t of tokens(it.product_name)) toks.add(t);
    if(it.variant) for(const t of tokens(it.variant)) toks.add(t); }
  return toks;
}
export function isGrounded(ingredient, kitchenToks){
  const low = String(ingredient||"").toLowerCase().trim();
  if(PANTRY.has(low)) return "pantry";
  const its = tokens(ingredient);
  if(its.some(t=>PANTRY.has(t))) return "pantry";
  if(its.some(t=>kitchenToks.has(t))) return "kitchen";
  return "missing";
}
// Score a recipe set: grounding rate over ingredients the recipe claims to USE
// (excluding declared missing_ingredients), plus variety metrics.
export function scoreRecipeSet(set, items){
  const kt = buildKitchenIndex(items);
  const recipes = [...(set.kitchen_only||[]), ...(set.need_grocery||[])];
  let usable=0, grounded=0, phantomInKitchenOnly=0;
  const proteins = new Set(), cats = {};
  const PROTEIN = ["chicken","beef","pork","turkey","salmon","tuna","shrimp","egg","tofu","bean","lentil","cheese","fish","ham","bacon","sausage"];
  for(const r of (set.kitchen_only||[])){
    const miss = new Set((r.missing_ingredients||[]).map(x=>String(x).toLowerCase()));
    for(const ing of (r.ingredients||[])){
      if(miss.has(String(ing).toLowerCase())) continue;
      usable++; const g=isGrounded(ing,kt);
      if(g!=="missing") grounded++; else phantomInKitchenOnly++;
    }
  }
  for(const r of (set.need_grocery||[])){
    const miss=new Set((r.missing_ingredients||[]).map(x=>String(x).toLowerCase()));
    for(const ing of (r.ingredients||[])){ if(miss.has(String(ing).toLowerCase()))continue; usable++; if(isGrounded(ing,kt)!=="missing")grounded++; }
  }
  for(const r of recipes){ cats[r.meal_category]=(cats[r.meal_category]||0)+1;
    const blob=(r.title+" "+(r.ingredients||[]).join(" ")).toLowerCase();
    for(const p of PROTEIN) if(blob.includes(p)) proteins.add(p); }
  return {
    recipeCount: recipes.length,
    kitchenOnlyCount:(set.kitchen_only||[]).length, needGroceryCount:(set.need_grocery||[]).length,
    groundingRate: usable? +(grounded/usable).toFixed(3):null,
    phantomIngredientsInKitchenOnly: phantomInKitchenOnly,
    distinctProteins: proteins.size, mealCategorySpread: cats,
  };
}
