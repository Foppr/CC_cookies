"""
Animal Cookie Lab - interface for animal_cookie_ea.py

    pip install deap numpy pandas streamlit
    streamlit run app.py        (or the "Animal Cookie EA (interface)" run option in PyCharm)
"""

import html
import json
import random

import streamlit as st

import animal_cookie_ea as ea

st.set_page_config(page_title="Animal Cookie Lab", page_icon="🍪", layout="wide",
                   initial_sidebar_state="collapsed")

ANIMALS = {
    "dog": {"icon": "🐶", "plural": "dogs", "line": "Hearty biscuits with meat, grains and veg"},
    "cat": {"icon": "🐱", "plural": "cats", "line": "Small bites with fish, poultry and herbs"},
    "bird": {"icon": "🐦", "plural": "birds", "line": "Crunchy cookies full of seeds and greens"},
    "fish": {"icon": "🐟", "plural": "fish", "line": "Flakes of seafood and algae"},
}
SCORE_LABELS = {"structure": "Category balance", "amounts": "Plausible amounts",
                "edible": "Human-edible", "safety": "Safe for the animal",
                "animal": "Animal character"}
TIER_DOT = {"good": ("good", "#3f8f4e"), "ok": ("ok", "#c79a2b"), "toxic": ("toxic", "#c0392b")}

# ----------------------------------------------------------------------------- style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=DM+Sans:wght@400;500;700&display=swap');
:root { --dough:#fbf4ea; --crumb:#f1e2cc; --crust:#b5651d; --cocoa:#3b2a1e; --muted:#8a7461;
        --card:#fffaf3; --line:#e8d6bd; }
.stApp { background: var(--dough); }
html, body, [class*="css"], .stMarkdown, p, li, label { font-family: 'DM Sans', sans-serif; color: var(--cocoa); }
h1, h2, h3, .serif { font-family: 'Fraunces', serif !important; color: var(--cocoa); letter-spacing: -0.01em; }
.block-container { padding-top: 4.2rem; max-width: 1180px; }
.hero h1 { font-size: 3.1rem; margin: 0 0 .2rem 0; line-height: 1.05; }
.hero p { font-size: 1.08rem; color: var(--muted); max-width: 720px; margin: 0; }
.eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .74rem; color: var(--crust);
           font-weight: 700; margin-bottom: .5rem; }
.pick { background: var(--card); border: 2px solid var(--line); border-radius: 22px;
        padding: 26px 22px 18px; text-align: center; transition: all .15s ease; min-height: 232px; }
.pick.on { border-color: var(--crust); box-shadow: 0 10px 28px rgba(181,101,29,.18); background: #fff6ea; }
.pick .emoji { font-size: 4rem; line-height: 1; }
.pick .name { font-family: 'Fraunces', serif; font-size: 1.7rem; font-weight: 700; margin: .5rem 0 .2rem; }
.pick .line { color: var(--muted); font-size: .95rem; min-height: 2.6em; }
.pick .ings { margin-top: .7rem; font-size: .8rem; color: var(--crust); }
div.stButton > button { border-radius: 999px; border: 1.5px solid var(--crust); color: var(--crust);
        background: transparent; font-weight: 700; padding: .55rem 1rem; }
div.stButton > button:hover { background: var(--crust); color: #fff; border-color: var(--crust); }
div.stButton > button[kind="primary"], div.stButton > button[kind="primary"] p { background: var(--crust); color: #fff !important; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(.recipe) { background: var(--card); border: 1px solid var(--line) !important;
     border-radius: 26px; padding: 18px 22px; box-shadow: 0 14px 40px rgba(59,42,30,.07); margin-top: 1.2rem; }
.recipe { padding: 12px 6px 4px; border-bottom: 1px solid var(--line); margin-bottom: 1.2rem; }
.recipe h2 { font-size: 2.4rem; margin: 0 0 .3rem 0; line-height: 1.1; }
.meta { color: var(--muted); font-size: .98rem; }
.badges { margin: .9rem 0 1.4rem; display: flex; flex-wrap: wrap; gap: .5rem; }
.badge { background: var(--crumb); border-radius: 999px; padding: .32rem .8rem; font-size: .84rem;
         font-weight: 500; }
.badge.ok { background: #e3f1e3; color: #2f6b3a; } .badge.bad { background: #f7dcd7; color: #9b2c1f; }
.section { font-family: 'Fraunces', serif; font-size: 1.35rem; font-weight: 700; margin: .2rem 0 .7rem; }
.cat { text-transform: uppercase; letter-spacing: .12em; font-size: .68rem; color: var(--muted);
       margin: 1rem 0 .25rem; font-weight: 700; }
.ing { display: flex; align-items: baseline; gap: .7rem; padding: .42rem 0;
       border-bottom: 1px dashed var(--line); }
.ing .g { font-weight: 700; min-width: 4.4rem; text-align: right; font-variant-numeric: tabular-nums; }
.ing .n { flex: 1; }
.chip { font-size: .7rem; border-radius: 999px; padding: .1rem .5rem; margin-left: .35rem;
        background: var(--crumb); color: var(--crust); white-space: nowrap; }
.dot { display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; margin-right: .3rem; }
.tier { font-size: .74rem; color: var(--muted); white-space: nowrap; }
ol.method { padding-left: 1.2rem; margin: 0; } ol.method li { margin-bottom: .8rem; line-height: 1.5; }
.bar { height: 8px; background: var(--crumb); border-radius: 999px; overflow: hidden; margin: .25rem 0 .7rem; }
.bar > span { display: block; height: 100%; background: var(--crust); border-radius: 999px; }
.score-row { display: flex; justify-content: space-between; font-size: .88rem; }
.big { font-family: 'Fraunces', serif; font-size: 3rem; font-weight: 700; color: var(--crust); line-height: 1; }
.small { color: var(--muted); font-size: .85rem; }
.alt { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 18px 20px; height: 100%; }
.alt h4 { font-family: 'Fraunces', serif; margin: 0 0 .3rem; font-size: 1.15rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading cookie recipes and pet food databases...")
def load():
    return ea.load_data()


db, rdb = load()
available = [a for a in ANIMALS if a in db.pool]
esc = html.escape


def top_ingredients(animal, n=3):
    """Most common human-edible, characteristic ingredients in this animal's food."""
    pool = db.pool[animal]
    cands = [i for i in pool if db.source[i] == "animal food" and db.edible[i] == "yes"
             and db.role[i] in ("Other (Mix-ins)", "flour (structure)", "fat")]
    return sorted(cands, key=pool.get, reverse=True)[:n]


def bake(animal):
    st.session_state.animal = animal
    s = st.session_state
    seed = random.randint(0, 10**6)
    with st.spinner(f"Evolving {ANIMALS[animal]['plural']}' cookies..."):
        st.session_state.result = ea.run(ea.Settings(
            animal=animal, generations=s.get("generations", 150), pop=s.get("pop", 80),
            recipes_in_book=s.get("n_book", 3), seed=seed, weights=s.get("weights")))
    st.session_state.seed = seed


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Settings")
    st.slider("Generations", 20, 500, 150, 10, key="generations")
    st.slider("Population per island", 20, 150, 80, 10, key="pop")
    st.slider("Recipes per batch", 1, 5, 3, key="n_book")
    st.markdown("**Fitness weights**")
    st.session_state.weights = {
        k: st.slider(SCORE_LABELS[k], 0.0, 1.0, v, 0.05, key=f"w_{k}")
        for k, v in ea.FITNESS_WEIGHTS.items()}

# ----------------------------------------------------------------------------- hero
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Animal Cookie Lab · evolutionary baking</div>
  <h1>Cookies you can share with your pet</h1>
  <p>Starting from {len(rdb.recipes)} human cookie recipes, the algorithm swaps in ingredients
  from real pet food until it finds a cookie that is still good to eat for you and safe for them.
  Pick who you are baking for.</p>
</div>""", unsafe_allow_html=True)
st.write("")

current = st.session_state.get("animal")
cols = st.columns(len(available), gap="medium")
for col, animal in zip(cols, available):
    a = ANIMALS[animal]
    ings = " · ".join(top_ingredients(animal))
    with col:
        st.markdown(f"""
        <div class="pick {'on' if animal == current else ''}">
          <div class="emoji">{a['icon']}</div>
          <div class="name">{animal.title()}</div>
          <div class="line">{esc(a['line'])}</div>
          <div class="ings">{esc(ings)}</div>
        </div>""", unsafe_allow_html=True)
        label = "Bake again" if animal == current else f"Bake for {a['plural']}"
        if st.button(label, key=f"bake_{animal}", use_container_width=True,
                     type="primary" if animal == current else "secondary"):
            bake(animal)
            st.rerun()

st.caption("Settings and fitness weights are in the sidebar (› at the top left).")

# ----------------------------------------------------------------------------- recipe
res = st.session_state.get("result")
if res is None:
    st.stop()

a = ANIMALS[res.animal]
top = res.book[0]


def ingredient_rows(rec):
    out, last_cat = [], None
    for i in rec["ingredients"]:
        if i["category"] != last_cat:
            last_cat = i["category"]
            out.append(f'<div class="cat">{esc(last_cat)}</div>')
        word, color = TIER_DOT[i["for_animal"]]
        chip = (f'<span class="chip">🐾 {res.animal} food</span>'
                if i["source"] == "animal food" else "")
        edible = "" if i["human_edible"] == "yes" else \
            f'<span class="chip">human: {esc(i["human_edible"])}</span>'
        grams = f"{i['grams']:.0f} g" if i["grams"] >= 10 else f"{i['grams']:.1f} g"
        out.append(f'<div class="ing"><span class="g">{grams}</span>'
                   f'<span class="n">{esc(i["ingredient"])}{chip}{edible}</span>'
                   f'<span class="tier"><span class="dot" style="background:{color}"></span>'
                   f'{word}</span></div>')
    return "".join(out)


def badges(rec):
    ok_edible = rec["n_inedible"] == 0
    ok_safe = rec["n_toxic"] == 0
    n_animal = sum(1 for i in rec["ingredients"] if i["source"] == "animal food")
    return (f'<div class="badges">'
            f'<span class="badge {"ok" if ok_edible else "bad"}">'
            f'{"✓ Human-edible" if ok_edible else "✗ Contains inedible ingredients"}</span>'
            f'<span class="badge {"ok" if ok_safe else "bad"}">'
            f'{"✓ No toxic ingredients for " + a["plural"] if ok_safe else "✗ Toxic for " + a["plural"]}'
            f'</span><span class="badge">🐾 {n_animal} ingredients from {res.animal} food</span>'
            f'<span class="badge">{len(rec["ingredients"])} ingredients</span></div>')


method = "".join(f"<li>{esc(step)}</li>" for step in top["method"])
n_cookies = top.get("cookies", int(res.batch_grams // ea.COOKIE_GRAMS))
card = st.container(border=True)
card.markdown(f"""
<div class="recipe">
  <div class="eyebrow">Top recipe · for {a['plural']} {a['icon']}</div>
  <h2>{esc(top['name'])}</h2>
  <div class="meta">Makes about {n_cookies} cookies · {res.batch_grams:.0f} g of dough ·
  baked at 175 °C for 10–12 minutes</div>
  {badges(top)}
</div>""", unsafe_allow_html=True)

left, mid, right = card.columns([5, 5, 3], gap="large")
with left:
    st.markdown(f'<div class="section">Ingredients</div>{ingredient_rows(top)}',
                unsafe_allow_html=True)
    st.markdown('<p class="small" style="margin-top:.8rem">Dot = how good the ingredient is for '
                f'{a["plural"]}: <span class="dot" style="background:#3f8f4e"></span>good '
                '<span class="dot" style="background:#c79a2b"></span>ok '
                '<span class="dot" style="background:#c0392b"></span>toxic</p>',
                unsafe_allow_html=True)
with mid:
    st.markdown(f'<div class="section">Method</div><ol class="method">{method}</ol>',
                unsafe_allow_html=True)
with right:
    bars = "".join(
        f'<div class="score-row"><span>{SCORE_LABELS.get(k, k)}</span><span>{v:.2f}</span></div>'
        f'<div class="bar"><span style="width:{min(v, 1) * 100:.0f}%"></span></div>'
        for k, v in top["scores"].items())
    st.markdown(f'<div class="section">Fitness</div><div class="big">{top["fitness"]:.2f}</div>'
                f'<p class="small">out of 1.00</p>{bars}', unsafe_allow_html=True)

if len(res.book) > 1:
    st.markdown('<div class="section" style="margin-top:2rem">More from this batch</div>',
                unsafe_allow_html=True)
    alt_cols = st.columns(len(res.book) - 1, gap="medium")
    for col, rec in zip(alt_cols, res.book[1:]):
        with col:
            st.markdown(f'<div class="alt"><h4>{esc(rec["name"])}</h4>'
                        f'<p class="small">fitness {rec["fitness"]:.2f}</p>'
                        f'{ingredient_rows(rec)}</div>', unsafe_allow_html=True)
            with st.expander("Method"):
                st.markdown("\n".join(f"{n}. {s}" for n, s in enumerate(rec["method"], 1)))

st.write("")
with st.expander("How the evolution went"):
    import pandas as pd
    hist = pd.DataFrame(res.history).groupby("generation")[["best", "mean"]].mean()
    st.line_chart(hist, y_label="fitness", color=["#b5651d", "#d9b48f"])
    tot = sum(res.mutation_log.values()) or 1
    st.caption("Mutations applied: " + ", ".join(
        f"{k} {v / tot:.0%}" for k, v in sorted(res.mutation_log.items(), key=lambda x: -x[1]))
        + f" · seed {st.session_state.get('seed')}")

st.download_button("Download this recipe batch (JSON)",
                   json.dumps({"animal": res.animal, "seed": st.session_state.get("seed"),
                               "recipes": res.book}, indent=2, ensure_ascii=False),
                   file_name=f"recipe_book_{res.animal}.json", mime="application/json")
