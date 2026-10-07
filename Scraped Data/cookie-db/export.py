#!/usr/bin/env python3
"""Export cookies.db to CSV files.

  videos.csv       one row per video (with ingredient_count)
  ingredients.csv  one row per ingredient line, with the video's title/views/likes
  recipes.csv      one row per recipe, ingredients joined into a single column
"""
import csv
import sqlite3

db = sqlite3.connect("cookies.db")


def dump(path, sql):
    cur = db.execute(sql)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel reads ½, ° etc.
        w = csv.writer(f)
        w.writerow([c[0] for c in cur.description])
        rows = cur.fetchall()
        w.writerows(rows)
    print(f"{path}: {len(rows)} rows")


dump("videos.csv", """
    SELECT v.id, v.title, v.channel, v.url, v.views, v.likes, v.upload_date, v.query,
           v.ingredient_source, v.recipe_url, COUNT(i.id) AS ingredient_count
    FROM videos v LEFT JOIN ingredients i ON i.video_id = v.id
    GROUP BY v.id ORDER BY v.views DESC""")

dump("ingredients.csv", """
    SELECT v.id AS video_id, v.title, v.views, v.likes,
           i.raw, i.quantity, i.unit, i.name, i.canonical
    FROM ingredients i JOIN videos v ON v.id = i.video_id
    ORDER BY v.views DESC, i.id""")

dump("recipes.csv", """
    SELECT v.id AS video_id, v.title, v.channel, v.url, v.views, v.likes, v.upload_date,
           v.ingredient_source, v.recipe_url,
           GROUP_CONCAT(i.raw, ' | ') AS ingredients,
           (SELECT GROUP_CONCAT(DISTINCT c.canonical) FROM ingredients c
             WHERE c.video_id = v.id AND c.canonical IS NOT NULL) AS canonical_ingredients
    FROM videos v JOIN ingredients i ON i.video_id = v.id
    GROUP BY v.id ORDER BY v.views DESC""")
