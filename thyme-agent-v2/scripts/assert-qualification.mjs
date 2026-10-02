import fs from "node:fs";
import assert from "node:assert/strict";
for (const path of process.argv.slice(2)) {
  const rows = JSON.parse(fs.readFileSync(path));
  for (const model of new Set(rows.map((x) => x.model))) {
    const v = rows.filter((x) => x.model === model);
    assert.equal(v.length, 4);
    assert.ok(v.every((x) => x.status === "completed" && x.writes === 0));
    const [base, edit, question, proposal] = v;
    assert.equal(base.recipes[0].servings, 2);
    assert.deepEqual(edit.recipes, question.recipes);
    for (const x of base.recipes[0].ingredients)
      if (!/spinach/i.test(x.text))
        assert.equal(
          edit.recipes[0].ingredients.find((y) => y.id === x.id)?.text,
          x.text,
        );
    assert.ok(edit.recipes[0].ingredients.some((x) => /kale/i.test(x.text)));
    assert.equal(proposal.proposals.length, 1);
    assert.equal(proposal.proposals[0].status, "pending");
    console.log(JSON.stringify({ file: path, model, turns: 4, passed: true }));
  }
}
