import os, json, pymysql, openai, uuid, re
from datetime import datetime

# ---------- ENV ----------
openai.api_key = os.environ["OPENAI_API_KEY"]
DB_HOST     = os.environ["DB_HOST"]
DB_USER     = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_NAME     = os.environ["DB_NAME"]

# ---------- HANDLER ----------
def lambda_handler(event, context):
    conn = None
    try:
        # ------------ Parse request ------------
        body        = json.loads(event["body"])
        web_id      = body.get("user_id")
        user_prompt = body.get("additional_request", "(no prompt)")
        if not web_id:
            return _err(400, "Missing user_id")

        # ------------ DB look‑ups ------------
        conn = pymysql.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD,
            database=DB_NAME, cursorclass=pymysql.cursors.DictCursor
        )
        with conn.cursor() as cur:
            # Map Auth0 sub → internal id
            cur.execute("SELECT web_id FROM users WHERE auth0_sub=%s", (web_id,))
            row = cur.fetchone()
            if not row:
                return _err(404, "User not found")
            owner_id = row["web_id"]

            # Pull last ~150 consumed items WITH dates
            cur.execute("""
                SELECT title, DATE_FORMAT(_createdDate,'%%Y-%%m-%%d') AS dt
                FROM items
                WHERE _owner=%s
                ORDER BY _createdDate DESC
                LIMIT 150
            """, (owner_id,))
            rows = cur.fetchall()

        # Build helper variables **the prompt expects**
        items               = [r["title"] for r in rows]
        titles_with_dates   = [f"{r['title']} (ate on {r['dt']})" for r in rows]

        # ------------ OpenAI prompt ------------
        prompt = f"""
        🔎 **Context you should learn from**
        • The user discarded **{len(items)}** items recently.  
        • Items + dates → {', '.join(titles_with_dates)}  

        💡 **What to do**
        1. Silently profile the user’s tastes & habits from the items list  
           – favorite cuisines, cravings, nutrition gaps, price sensitivity, cooking skill, etc.  
        2. Blend that insight with the user’s weekly goal: “{user_prompt}”.  
        3. Return **exactly three recipes** that will make the user *feel* they’re winning:
           • 1 × **Description** (1-2 punchy sentences explaining how the three recipes help the user reach **“{user_prompt}”**)  
           • 1 × **Dinner** (balanced, satisfying recipe)  
           • 1 × **Light Lunch / Snack** (quick, portable recipe)  
           • 1 × **Dessert** (sweet but goal‑aligned recipe)

        🍽 **Recipe output format—don’t deviate**
        Return a JSON object with four keys: `"description"`, `"dinner"`, `"lunch_snack"`, `"dessert"`.  
        Each value is an **HTML string** that follows *this* template (use the right title & data):

        <h4>{{Recipe Title}}</h4>
        <strong>Servings:</strong> {{#}} &nbsp;&nbsp;|&nbsp;&nbsp;
        <strong>Prep:</strong> ~{{#}} mins &nbsp;&nbsp;|&nbsp;&nbsp;
        <strong>Cook:</strong> ~{{#}} mins
        <br><hr>
        <strong>Ingredients:</strong>
        <ul>
          <li>{{quantity}} {{unit}} {{**ingredient**}}</li>
          …
        </ul>
        <hr>
        <strong>Directions:</strong>
        <ol>
          <li>{{concise step}}</li>
          …
        </ol>
        <hr>
        <em>Tip: {{pithy dopamine‑boosting tip}}</em>

        🧠 **Voice & style rules**
        • Write to the user like their upbeat performance coach + foodie therapist.  
        • Keep sentences short, vivid, dopamine‑inducing.  
        • Use sensory verbs (“sizzle”, “burst”, “velvety”) to spark craving.  
        • Bold key nouns (recipe titles, hero ingredients, pro tips).  
        • Ingredient lists max 12 items; directions ≤ 6 steps.  
        • Default to common grocery brands unless a discarded‑item insight suggests something niche.  
        • Respect dietary cues implicit in the thrown‑out list (e.g., if lactose‑free products appear, avoid dairy).  
        • Macros matter: flag protein grams or fiber boosts inline where relevant.  
        • Avoid medical claims, guilt, or jargon.

        🚫 **Do NOT** return explanations, markdown fences, or any text outside the JSON.
        """

        chat = openai.ChatCompletion.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        try:
            recipes_json = json.loads(chat.choices[0].message.content)

            # ── POST-PROCESS: turn **foo** into <strong>foo</strong> in every recipe slot
            for slot, html in recipes_json.items():
                # non-greedy, global
                recipes_json[slot] = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

        except Exception as e:
            return _err(502, f"OpenAI returned invalid JSON: {e}")

        _log_request(conn, owner_id, user_prompt, recipes_json)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(recipes_json)
        }

    except Exception as e:
        return _err(500, str(e))

    finally:
        if conn:
            conn.close()

# ---------- HELPERS ----------
def _err(code, msg):
    return {
        "statusCode": code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": msg})
    }

def _log_request(conn, owner_id, req, resp):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ai_request_logs (id,timestamp,_owner,request_text,response_text)
            VALUES (%s,%s,%s,%s,%s)
        """, (
            str(uuid.uuid4()),
            datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            owner_id, req, json.dumps(resp)
        ))
    conn.commit()
