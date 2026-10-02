import fs from "node:fs";
import { Fault, hash } from "../src/core.mjs";
import { catalog } from "../src/capabilities.mjs";
import { prepareAction } from "../src/prepare-actions.mjs";
import { ACTIONS, TrepoGateway } from "../src/gateway.mjs";
export const ACTOR = {
  actor: "fixture-matt",
  household: "fixture-home",
  name: "Matt",
  ctx: {},
};
export class FixtureGateway {
  constructor() {
    this.writes = 0;
    this.failRead = null;
    this.failWrite = null;
    this.actorValue = ACTOR;
    this.data = {
      kitchen: [
        {
          id: "eggs",
          item_name: "Eggs",
          quantity_value: 6,
          quantity_unit: "count",
          storage_location: "fridge",
        },
        {
          id: "spinach",
          item_name: "Spinach",
          quantity_value: null,
          storage_location: "fridge",
        },
        {
          id: "rice",
          item_name: "Rice",
          quantity_value: 2,
          quantity_unit: "cups",
          storage_location: "pantry",
        },
      ],
      shopping: [
        {
          id: "tomatoes",
          item_name: "Tomatoes",
          store: "Groceries",
          action: "OUT",
        },
      ],
      preferences: {
        dietary: {
          allergies: ["peanut"],
          diets: [],
          religious: [],
          health: [],
          custom: [],
        },
        memory: {
          explicit_preferences: {
            cooking_style: { value: ["quick dinners"], scope: "personal" },
          },
          observations: {},
          revision: 1,
        },
      },
      saved_recipes: [],
      suggestions: [],
      calendar: [],
      meal_plan: { status: "ready" },
      dishes: [],
      discards: [],
      recipe_categories:{categories:[],assignments:{}},
    };
  }
  async init() {}
  async actor(actor) {
    if (actor !== this.actorValue.actor)
      throw new Fault("not_enrolled", "Not enrolled.", 403);
    return structuredClone(this.actorValue);
  }
  async check(a) {
    if (
      a.actor !== this.actorValue.actor ||
      a.household !== this.actorValue.household
    )
      throw new Fault("membership", "Household changed.", 403);
    return a;
  }
  async read(a, resource) {
    await this.check(a);
    if (this.failRead === resource) throw new Error("read failure");
    return structuredClone(this.data[resource]);
  }
  async definitions() {
    return catalog(JSON.parse(fs.readFileSync(new URL("../vendor/tool-definitions.json",import.meta.url))),ACTIONS);
  }
  async prepare(a,name,args,before) {return prepareAction(this,a,name,args,before);}
  async readTool(a,name,args) {return TrepoGateway.prototype.readTool.call(this,a,name,args);}
  async mutate(a, action, args, operationId) {
    await this.check(a);
    this.writes++;
    if (this.failWrite) throw new Error("write outcome unknown");
    if (
      action === "add_to_shopping_list" ||
      action === "add_many_to_shopping_list"
    ) {
      for (const t of args.items || [args])
        this.data.shopping.push({
          id: hash([operationId, t.item_name, this.data.shopping.length]).slice(
            0,
            10,
          ),
          item_name: t.item_name,
          quantity: t.quantity,
          quantity_value: t.quantity_value,
          quantity_unit: t.quantity_unit,
          brand: t.brand,
          store: t.store || "Groceries",
          action: "OUT",
        });
      return { ok: true, toolResult: { items: this.data.shopping } };
    }
    if (action === "remove_from_shopping_list") {
      this.data.shopping = this.data.shopping.filter(
        (x) => x.item_name.toLowerCase() !== args.item_name.toLowerCase(),
      );
      return { ok: true };
    }
    if (action === "update_item_quantity") {
      const row = this.data.kitchen.find((x) =>
        args.item_id
          ? x.id === args.item_id
          : x.item_name.toLowerCase() === args.item_name.toLowerCase(),
      );
      Object.assign(row, args);
      return { ok: true, toolResult: row };
    }
    if (action === "save_generated_recipe") {
      const row = {
        id: operationId,
        title: args.title,
        ingredients: args.ingredients,
        instructions: args.steps,
        notes: args.notes || [],
      };
      this.data.saved_recipes.push(row);
      return { ok: true, toolResult: { recipe: row } };
    }
    if (action === "update_food_memory") {
      this.data.preferences.memory.revision++;
      this.data.preferences.memory.explicit_preferences[args.key] = {
        value: args.values || args.minutes,
        scope: args.visibility || "personal",
      };
      return { ok: true };
    }
    throw new Fault(
      "fixture_unsupported",
      "Fixture does not emulate this mutation.",
    );
  }
}
