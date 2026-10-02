import { Fault, hash } from "./core.mjs";
// A save persists the reviewed recipe as-is. No second model gets to rewrite it.
export async function saveCanonical(connection, actor, args, operationId) {
  if (!/^[a-f0-9-]{36}$/i.test(actor.actor))
    throw new Fault("actor", "Invalid account.", 403);
  const digest = hash([actor.actor, operationId]);
  const id = [
    digest.slice(0, 8),
    digest.slice(8, 12),
    digest.slice(12, 16),
    digest.slice(16, 20),
    digest.slice(20, 32),
  ].join("-");
  const url = `trepo-generated:${id}`;
  const table = actor.actor + "_saved_recipes";
  await connection.beginTransaction();
  try {
    const [members] = await connection.execute(
      "SELECT owner_id FROM new_users WHERE user_id=? FOR UPDATE",
      [actor.actor],
    );
    if (
      !members[0] ||
      String(members[0].owner_id || actor.actor) !== actor.household
    )
      throw new Fault("membership", "Your household changed.", 403);
    const [prior] = await connection.execute(
      "SELECT title,ingredients,instructions,notes FROM shared_saved_recipes WHERE owner_id=? AND _id=? FOR UPDATE",
      [actor.actor, id],
    );
    if (prior.length) {
      const row = prior[0],
        decode = (v) => (typeof v === "string" ? JSON.parse(v) : v);
      if (
        row.title !== args.title ||
        hash(decode(row.ingredients)) !== hash(args.ingredients) ||
        hash(decode(row.instructions)) !== hash(args.steps) ||
        hash(decode(row.notes) || []) !== hash(args.notes || [])
      )
        throw new Fault(
          "request_conflict",
          "A different recipe already used this request.",
          409,
        );
      await connection.commit();
      return {
        ok: true,
        toolResult: {
          recipe: {
            id,
            title: args.title,
            ingredients: args.ingredients,
            instructions: args.steps,
          },
        },
      };
    }
    const [columns] = await connection.execute(
      "SELECT COLUMN_NAME FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=?",
      [table],
    );
    const required = [
      "_id",
      "_owner",
      "source_type",
      "source_url",
      "resolved_url",
      "resolved_url_hash",
      "title",
      "ingredients",
      "instructions",
      "notes",
      "status",
    ];
    if (!required.every((n) => columns.some((c) => c.COLUMN_NAME === n)))
      throw new Fault(
        "saved_recipe_schema",
        "Your saved recipe library needs an update before this can be saved.",
        409,
      );
    const values = [
      id,
      actor.actor,
      "text",
      "",
      url,
      hash(url),
      args.title,
      JSON.stringify(args.ingredients),
      JSON.stringify(args.steps),
      JSON.stringify(args.notes || []),
      "ready",
    ];
    const fields = required.map((n) => "`" + n + "`").join(","),
      places = required.map(() => "?").join(",");
    await connection.execute(
      `INSERT INTO shared_saved_recipes (owner_id,${fields}) VALUES (?,${places})`,
      [actor.actor, ...values],
    );
    const hasOwner = columns.some((c) => c.COLUMN_NAME === "owner_id");
    await connection.execute(
      `INSERT INTO \`${table}\` (${hasOwner ? "owner_id," : ""}${fields}) VALUES (${hasOwner ? "?," : ""}${places})`,
      hasOwner ? [actor.actor, ...values] : values,
    );
    await connection.commit();
    return {
      ok: true,
      toolResult: {
        recipe: {
          id,
          title: args.title,
          ingredients: args.ingredients,
          instructions: args.steps,
        },
      },
    };
  } catch (e) {
    await connection.rollback();
    throw e;
  }
}
