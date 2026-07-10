// Builds identify_images.json (committed: s3_key + deep-path ground-truth label)
// and downloads image bytes to fixtures/images/ (gitignored) for the harness.
import mysql from "mysql2/promise";
import { writeFileSync } from "node:fs";
const c = await mysql.createConnection({host:"database-1.cvig8u6s25dz.us-east-1.rds.amazonaws.com",user:"admin",password:"Nbmqyq17!",database:"mysqlTutorial",connectTimeout:15000});
// 10 varied captures w/ deep-path ground truth; Merlot from archive (the hard case).
const picks = [
  ["shared_archive_kitchen","05afcaf9"], // Merlot Wine — HARD
  ["shared_kitchen","fda6bf54"], // Oven Roasted Turkey Breast
  ["shared_kitchen","9b15da28"], // Marinated Smoked Pork Tenderloin
  ["shared_kitchen","d56d99fb"], // Long Bone Heritage Pork Chop
  ["shared_kitchen","0f566ab3"], // Asian Chopped Salad Kit
  ["shared_kitchen","290acf6e"], // Basil Pesto
  ["shared_kitchen","c45c0cbe"], // Roasted Pine Nut Hummus
  ["shared_kitchen","ea5b519d"], // Provolone Cheese Slices
  ["shared_kitchen","56c41483"], // Mexican Style 4 Cheese Blend
  ["shared_kitchen","291f6fb7"], // Half & Half Blend
];
const manifest = [];
for (const [tbl,frag] of picks) {
  const [r] = await c.execute(`SELECT product_name,brand,category,s3_key FROM \`${tbl}\` WHERE s3_key LIKE ? LIMIT 1`,[`%${frag}%`]);
  if(!r[0]){ console.log("MISS",frag); continue; }
  manifest.push({ id: frag, s3_key: r[0].s3_key, truth_name: r[0].product_name, truth_brand: r[0].brand||null, truth_category: r[0].category, source_table: tbl });
}
writeFileSync("fixtures/identify_images.json", JSON.stringify({bucket:"trepo-grocery-uploads-dev", images:manifest}, null, 2));
console.log(`manifest: ${manifest.length} images`);
manifest.forEach(m=>console.log(`  ${m.id}  ${m.truth_name} [${m.truth_category}]`));
await c.end();
