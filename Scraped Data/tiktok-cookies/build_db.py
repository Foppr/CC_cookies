"""Build cookies.db (SQLite) and CSV exports from transcripts/ + extracted/.

extracted/<id>.json format:
  {"recipe_name": str, "cookie_type": str, "is_recipe": bool, "source": "caption|audio|both",
   "ingredients": [{"raw": str, "name": str, "quantity": float|null, "unit": str|null, "note": str|null}]}
"""
import csv
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent
DB = ROOT / "cookies.db"
DB.unlink(missing_ok=True)
con = sqlite3.connect(DB)
con.executescript("""
CREATE TABLE recipes (
    id TEXT PRIMARY KEY, url TEXT, uploader TEXT, recipe_name TEXT, cookie_type TEXT,
    source TEXT, view_count INTEGER, like_count INTEGER, upload_date TEXT,
    description TEXT, transcript TEXT
);
CREATE TABLE ingredients (
    recipe_id TEXT REFERENCES recipes(id), position INTEGER, name TEXT,
    quantity REAL, unit TEXT, note TEXT, raw TEXT
);
CREATE INDEX idx_ing_name ON ingredients(name);
""")

skipped = 0
for ext_path in sorted((ROOT / "extracted").glob("*.json")):
    ext = json.loads(ext_path.read_text())
    if not ext.get("is_recipe") or not ext.get("ingredients"):
        skipped += 1
        continue
    t = json.loads((ROOT / "transcripts" / ext_path.name).read_text())
    con.execute("INSERT INTO recipes VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
        t["id"], t["url"].replace("/@_/", f"/@{t.get('uploader') or '_'}/"), t.get("uploader"),
        ext["recipe_name"], ext.get("cookie_type"), ext.get("source"),
        t.get("view_count"), t.get("like_count"), t.get("upload_date"),
        t.get("description"), t.get("transcript"),
    ))
    for i, ing in enumerate(ext["ingredients"], 1):
        con.execute("INSERT INTO ingredients VALUES (?,?,?,?,?,?,?)", (
            t["id"], i, ing["name"], ing.get("quantity"), ing.get("unit"), ing.get("note"), ing.get("raw"),
        ))
con.commit()

for table, query in {
    "recipes.csv": "SELECT id, url, uploader, recipe_name, cookie_type, source, view_count, like_count, upload_date FROM recipes",
    "ingredients.csv": "SELECT r.recipe_name, r.url, i.* FROM ingredients i JOIN recipes r ON r.id = i.recipe_id ORDER BY r.id, i.position",
}.items():
    cur = con.execute(query)
    with open(ROOT / table, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([d[0] for d in cur.description])
        w.writerows(cur)

n_rec = con.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
n_ing = con.execute("SELECT COUNT(*) FROM ingredients").fetchone()[0]
print(f"{n_rec} recipes, {n_ing} ingredient rows, {skipped} skipped (not a recipe / no ingredients)")
