from pathlib import Path
import zipfile,re,json,shutil
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True);args=parser.parse_args()
P=args.evidence;W=Path(__file__).resolve().parents[1];REL='playground12_voice_ack/trepo-quick-ack/lib/'
NAMES=['trepo-quick-ack-stream-dev','trepo-quick-ack-sam-dev','trepo-quick-ack-async-worker-dev','trepo-thyme-agent-v2-pilot-native','trepo-thyme-agent-v2-pilot-worker','trepo-thyme-agent-v2-pilot-api']
def block(text,name):
 m=re.search(r'^(?:export )?(?:async )?function '+name+r'\(',text,re.M);assert m,name
 end=text.index('\n}\n',m.start())+3;return text[m.start():end]
with zipfile.ZipFile(P/'trepo-quick-ack-stream-dev/before.zip') as z:old_data=z.read(REL+'data-access.mjs').decode();old_assistant=z.read(REL+'device-assistant.mjs').decode()
new_data=(W/REL/'data-access.mjs').read_text();new_assistant=(W/REL/'device-assistant.mjs').read_text()
for name in NAMES:
 d=P/name;updates={REL+'household-food-data.mjs':(W/REL/'household-food-data.mjs').read_text()}
 with zipfile.ZipFile(d/'before.zip') as z:
  text=z.read(REL+'data-access.mjs').decode()
  for f in ['normalizeSavedRecipeItem','normalizeSavedRecipesPayload','getShoppingItems','getSavedRecipes','getSavedRecipeDetail','removeSavedRecipe','getMealPlan','moveRecipeToCategory','resolveSavedRecipeForCalendar','normalizeMealCalendarEntry']:
   old=block(old_data,f)
   if f=='getShoppingItems' and name.startswith('trepo-thyme'): old=old.replace('const limitedItems = items.slice(0, clampLimit(options.limit, 25, 100));','const limitedItems = options.all === true ? items : items.slice(0, clampLimit(options.limit, 25, 100));')
   assert text.count(old)==1,(name,f)
   text=text.replace(old,block(new_data,f))
  text=text.replace('import crypto from "node:crypto";','import crypto from "node:crypto";\nimport { readHouseholdFoodRows, householdSavedRecipes, householdShoppingRows, householdMealPlans } from "./household-food-data.mjs";',1)
  text=text.replace('export async function getSavedRecipes(',block(new_data,'readSavedRecipeCollection')+'\nexport async function getSavedRecipes(',1)
  updates[REL+'data-access.mjs']=text
  if name.startswith('trepo-quick-ack'):
   text=z.read(REL+'device-assistant.mjs').decode()
   for f in ['buildShoppingGroundingMessage','buildRecipeGroundingMessage']:
    old=block(old_assistant,f);assert text.count(old)==1;(text:=text.replace(old,block(new_assistant,f)))
   updates[REL+'device-assistant.mjs']=text
  else:
   assert z.read('thyme-agent-v2/src/gateway.mjs')==(P/'trepo-thyme-agent-v2-pilot-native/source/thyme-agent-v2/src/gateway.mjs').read_bytes()
   updates['thyme-agent-v2/src/gateway.mjs']=(W/'thyme-agent-v2/src/gateway.mjs').read_text()
  for path,content in updates.items():
   target=d/'overlay'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content)
 if name in ['trepo-quick-ack-stream-dev','trepo-thyme-agent-v2-pilot-native']:
  runtime=P/('runtime' if name.startswith('trepo-quick') else 'native-runtime')
  if runtime.is_dir():
   for path in updates:shutil.copy2(d/'overlay'/path,runtime/path)
 print(name,len(updates),'files')

generator=P/'trepo-grocery-backend-dev-MealPlanGeneratorFunctio-4rNstnlDTS1Z'/'overlay'
generator.mkdir(exist_ok=True)
for filename in ['app.py','recipe_generation_guard.py']:
 shutil.copy2(W/'playground6_groceryRec/meal_plan_generator'/filename,generator/filename)

consumer=P/'trepo-household-brain-v2-StreamConsumer-tcofFrGVflHI'/'overlay/household_brain'
consumer.mkdir(parents=True,exist_ok=True)
shutil.copy2(W/'playground6_groceryRec/household_brain/external_queue.py',consumer/'external_queue.py')
