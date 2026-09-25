import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import mysql from 'mysql2/promise';
import { getDbConfig, getDbPool } from '../lib/mysql.mjs';
import { checkInKitchenItem, updateKitchenItemQuantity } from '../lib/data-access.mjs';

after(async () => { if (process.env.TREPO_VOICE_IMAGE_MYSQL) await getDbPool().end(); });
async function fixture(run) {
  assert.equal(process.env.DB_HOST, '127.0.0.1');
  assert.match(process.env.DB_NAME, /^thyme_test_/);
  const id = crypto.randomUUID();
  const context = { ownerId: id, userId: id, tableOwnerId: id, householdMemberIds: [id], responseSurface: 'app' };
  const options = { skipKitchenDependentGeneration: true };
  let connection;
  try {
    const item = await checkInKitchenItem(context, { item_name: 'Ground beef', quantity_value: 25, quantity_unit: 'oz' }, options);
    await run(context, { item_id: item.id }, options);
    connection = await mysql.createConnection(getDbConfig());
    const [rows] = await connection.execute(`SELECT quantity_value,quantity_unit,fill_percent,action FROM \`${id}_prod_kitchen\` WHERE _id=?`, [item.id]);
    assert.equal(rows.length, 1);
    return rows[0];
  } finally {
    if (!connection) connection = await mysql.createConnection(getDbConfig());
    await connection.execute(`DROP TABLE IF EXISTS \`${id}_prod_kitchen\``);
    await connection.end();
  }
}

for (const [input, amount, unit] of [
  ['12.5 oz', 12.5, 'oz'], ['1/2 cup', 0.5, 'cup'], ['1 1/2 cups', 1.5, 'cup'],
  ['½ cup', 0.5, 'cup'], ['half a cup', 0.5, 'cup'], ['125 g', 125, 'g'], ['2 items left', 2, 'item'],
  ['.5 oz', 0.5, 'oz'], ['one ounce', 1, 'oz'], ['half a bushel', 0.5, 'bushel'],
  ['half of 25 oz', 12.5, 'oz'], ['a quarter of 200 g', 50, 'g'],
  ['1/2 of 25 oz', 12.5, 'oz'],
  ['two one-cup portions', 2, 'cup'],
  ['3 1/2-cup portions', 1.5, 'cup'],
  ['two 8-ounce bottles', 16, 'oz'],
  [{ quantity_value: 12.5, quantity_unit: 'oz' }, 12.5, 'oz'],
  [{ quantity_value: 12.5, quantity_unit: 'oz', remaining_quantity: 'half of the ground beef' }, 12.5, 'oz'],
  [{ remaining_quantity: '12.5 oz', quantity_value: null, fill_percent: null }, 12.5, 'oz'],
]) {
  test(`quantity text persists exactly: ${JSON.stringify(input)}`, { skip: !process.env.TREPO_VOICE_IMAGE_MYSQL }, async () => {
    const row = await fixture((context, reference, options) => updateKitchenItemQuantity(context, reference, input, options));
    assert.equal(Number(row.quantity_value), amount);
    assert.equal(row.quantity_unit, unit);
    assert.equal(row.action, 'IN');
    assert.equal(row.fill_percent, null, 'An amount in cups/ounces is not a container percentage');
  });
}

for (const input of ['1/0 cup', '-1 oz', '1/0 of 25 oz', 'half of half of 25 oz', '-1/2 of 25 oz', 'half of the ground beef']) {
  test(`invalid text ${input} leaves the original amount intact`, { skip: !process.env.TREPO_VOICE_IMAGE_MYSQL }, async () => {
    const row = await fixture(async (context, reference, options) => {
      await assert.rejects(updateKitchenItemQuantity(context, reference, input, options), error => error.statusCode === 400);
    });
    assert.equal(Number(row.quantity_value), 25);
    assert.equal(row.quantity_unit, 'oz');
  });
}
