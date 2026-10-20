#!/usr/bin/env python3
"""Scrape YouTube cookie-recipe videos into a SQLite ingredient database.

Ingredients come from the video description when listed there; otherwise from
schema.org/Recipe JSON-LD on a recipe page linked in the description.

Usage: python3 scrape.py [videos_per_query]
"""
import json
import re
import sqlite3
import subprocess
import sys
import urllib.request
from html import unescape

QUERIES = [
    "chocolate chip cookie recipe",
    "sugar cookie recipe",
    "oatmeal raisin cookie recipe",
    "peanut butter cookie recipe",
    "snickerdoodle recipe",
    "double chocolate cookie recipe",
    "shortbread cookie recipe",
    "gingerbread cookie recipe",
    "white chocolate macadamia cookie recipe",
    "molasses cookie recipe",
    "brown butter cookie recipe",
    "chewy cookie recipe",
    "soft cookie recipe",
    "crinkle cookie recipe",
    "red velvet cookie recipe",
    "lemon cookie recipe",
    "m&m cookie recipe",
    "cookie recipe no chill",
    "thumbprint cookie recipe",
    "butter cookie recipe",
    "nutella cookie recipe",
    "levain style cookie recipe",
    "stuffed cookie recipe",
    "vegan cookie recipe",
    "gluten free cookie recipe",
    "chocolate chip cookies from scratch",
    "cookie recipe easy homemade",
    "christmas cookie recipe",
    "cut out cookie recipe",
    "matcha cookie recipe",
    "coconut cookie recipe",
    "almond cookie recipe",
    "cranberry white chocolate cookie recipe",
    "salted caramel cookie recipe",
    "s'mores cookie recipe",
    "espresso chocolate cookie recipe",
    "pumpkin cookie recipe",
    "chocolate crinkle cookies",
    "oatmeal chocolate chip cookie recipe",
    "sugar cookie icing recipe",
]

UNITS = r"(cups?|c\.|tbsp|tablespoons?|tsp|teaspoons?|g|grams?|kg|ml|oz|ounces?|lbs?|pounds?|sticks?|large|pinch|dash)"
QTY = r"(\d+\s*/\s*\d+|\d+(?:[.,]\d+)?(?:\s+\d/\d)?|[½¼¾⅓⅔⅛])"
ING_LINE = re.compile(rf"^\s*[-•*▢]?\s*{QTY}\s*{UNITS}?\b\.?\s*(?:of\s+)?(.+)$", re.I)
NOT_ING = re.compile(r"^\s*\d+[.)]\s+\S|^\s*\d{1,2}:\d{2}|-ingredient|\d+\s*(mins?|minutes|hours?|°|℃|degrees)\b|\b(bake|preheat|chill|cover|refrigerate)\b", re.I)
SKIP_DOMAINS = ("youtube.com", "youtu.be", "instagram", "tiktok", "facebook", "pinterest",
                "amazon", "twitter", "x.com", "patreon", "bit.ly/sub", "spotify")

KEYWORDS = {  # canonical ingredient -> matching substrings
    "all-purpose flour": ["all-purpose flour", "all purpose flour", "plain flour", "flour"],
    "bread flour": ["bread flour"],
    "butter": ["butter"],
    "brown sugar": ["brown sugar"],
    "granulated sugar": ["granulated sugar", "white sugar", "caster sugar", "sugar"],
    "powdered sugar": ["powdered sugar", "icing sugar", "confectioners"],
    "egg": ["egg"],
    "vanilla extract": ["vanilla"],
    "baking soda": ["baking soda", "bicarbonate", "bicarb"],
    "baking powder": ["baking powder"],
    "salt": ["salt"],
    "chocolate chips": ["chocolate chip", "chocolate chunk", "chopped chocolate", "dark chocolate", "semisweet", "semi-sweet", "milk chocolate"],
    "white chocolate": ["white chocolate"],
    "cocoa powder": ["cocoa"],
    "oats": ["oats", "oatmeal"],
    "raisins": ["raisin"],
    "peanut butter": ["peanut butter"],
    "cinnamon": ["cinnamon"],
    "ginger": ["ginger"],
    "molasses": ["molasses"],
    "cream of tartar": ["cream of tartar"],
    "cornstarch": ["cornstarch", "corn starch", "cornflour"],
    "macadamia nuts": ["macadamia"],
    "nuts": ["walnut", "pecan", "almond", "hazelnut"],
    "milk": ["milk"],
    "honey": ["honey"],
    "nutmeg": ["nutmeg"],
    "cloves": ["clove"],
    "flaky sea salt": ["flaky", "sea salt", "maldon"],
    "oil": ["oil"],
}


def canonical(name):
    n = name.lower()
    # check longer/more specific phrases first
    pairs = sorted(((kw, c) for c, kws in KEYWORDS.items() for kw in kws), key=lambda p: -len(p[0]))
    for kw, c in pairs:
        if kw in n:
            return c
    return None


def run_ytdlp(query, n):
    out = subprocess.run(
        ["yt-dlp", "--skip-download", "--no-warnings", "-j", f"ytsearch{n}:{query}"],
        capture_output=True, text=True, timeout=600,
    ).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def parse_lines(lines):
    out = []
    for line in lines:
        line = unescape(line).strip()
        if len(line) > 120 or "http" in line or NOT_ING.search(line):
            continue
        m = ING_LINE.match(line)
        if m:
            out.append((line, m.group(1), m.group(2) or "", m.group(3).strip()))
    return out


def recipe_links(desc):
    urls = re.findall(r"https?://[^\s)>\]]+", desc)
    return [u for u in urls if not any(d in u.lower() for d in SKIP_DOMAINS)]


def fetch_jsonld_ingredients(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")
    except Exception:
        return None
    for block in re.findall(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', html, re.S):
        try:
            data = json.loads(block)
        except Exception:
            continue
        stack = [data]
        while stack:
            d = stack.pop()
            if isinstance(d, list):
                stack.extend(d)
            elif isinstance(d, dict):
                t = d.get("@type")
                if (t == "Recipe" or (isinstance(t, list) and "Recipe" in t)) and d.get("recipeIngredient"):
                    return d["recipeIngredient"]
                stack.extend(d.get("@graph", []) if isinstance(d.get("@graph"), list) else [])
    return None


def setup(db):
    db.executescript("""
    CREATE TABLE IF NOT EXISTS videos(
        id TEXT PRIMARY KEY, title TEXT, channel TEXT, url TEXT, views INTEGER, likes INTEGER,
        upload_date TEXT, query TEXT, ingredient_source TEXT, recipe_url TEXT);
    CREATE TABLE IF NOT EXISTS ingredients(
        id INTEGER PRIMARY KEY, video_id TEXT REFERENCES videos(id),
        raw TEXT, quantity TEXT, unit TEXT, name TEXT, canonical TEXT);
    CREATE INDEX IF NOT EXISTS idx_ing_canon ON ingredients(canonical);
    """)
    if "likes" not in [r[1] for r in db.execute("PRAGMA table_info(videos)")]:
        db.execute("ALTER TABLE videos ADD COLUMN likes INTEGER")


def main():
    per_query = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    db = sqlite3.connect("cookies.db")
    setup(db)
    for q in QUERIES:
        print(f"== {q}", flush=True)
        for v in run_ytdlp(q, per_query):
            if db.execute("SELECT 1 FROM videos WHERE id=?", (v["id"],)).fetchone():
                continue
            desc = v.get("description") or ""
            ings, source, rurl = parse_lines(desc.splitlines()), "description", None
            if len(ings) < 3:
                ings, source = [], None
                for link in recipe_links(desc)[:3]:
                    got = fetch_jsonld_ingredients(link)
                    if got:
                        parsed = parse_lines(got)
                        # keep unparsed lines too, just without qty/unit
                        seen = {p[0] for p in parsed}
                        parsed += [(g, "", "", g) for g in map(unescape, got) if g not in seen]
                        ings, source, rurl = parsed, "recipe_page", link
                        break
            db.execute("""INSERT INTO videos(id,title,channel,url,views,likes,upload_date,query,
                          ingredient_source,recipe_url) VALUES(?,?,?,?,?,?,?,?,?,?)""", (
                v["id"], v.get("title"), v.get("channel"), v.get("webpage_url"),
                v.get("view_count"), v.get("like_count"), v.get("upload_date"), q, source, rurl))
            db.executemany(
                "INSERT INTO ingredients(video_id,raw,quantity,unit,name,canonical) VALUES(?,?,?,?,?,?)",
                [(v["id"], raw, qty, unit, name, canonical(name)) for raw, qty, unit, name in ings])
            db.commit()
            print(f"  {len(ings):2d} ingredients [{source or 'none'}] {v.get('title')}", flush=True)
    db.close()


if __name__ == "__main__":
    main()
