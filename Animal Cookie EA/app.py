"""
Animal Cookie Lab - interface for animal_cookie_ea.py

    pip install deap numpy pandas streamlit
    streamlit run app.py
"""

import json
import random

import pandas as pd
import streamlit as st

import animal_cookie_ea as ea

ICONS = {"cat": "🐱", "dog": "🐶", "bird": "🐦", "fish": "🐟"}
EDIBLE = {"yes": "✅ yes", "limited": "⚠️ limited", "no": "❌ no"}
TIER = {"good": "🟢 good", "ok": "🟡 ok", "toxic": "🔴 toxic"}

st.set_page_config(page_title="Animal Cookie Lab", page_icon="🍪", layout="wide")


@st.cache_resource(show_spinner="Loading cookie recipes and animal food databases...")
def data():
    return ea.load_data()


db, rdb = data()
animals = list(db.pool)

st.title("🍪 Animal Cookie Lab")
st.caption(f"Starts from {len(rdb.recipes)} human cookie recipes and mutates them with "
           "ingredients from real pet food. The fitness looks for a good cookie that a "
           "human can still eat and that is safe for the animal.")

animal = st.segmented_control("Who are the cookies for?", animals, default=animals[0],
                              format_func=lambda a: f"{ICONS.get(a, '')} {a.title()}",
                              key="animal") or animals[0]
st.caption(f"{len(db.pool[animal])} {animal}-food ingredients available to the mutation, "
           f"from {db.n_products[animal]} products.")

with st.expander("Settings"):
    c1, c2, c3 = st.columns(3)
    generations = c1.slider("Generations", 20, 500, 150, 10)
    pop = c2.slider("Population per island", 20, 150, 80, 10)
    n_book = c3.slider("Recipes in the book", 1, 5, 3)
    st.markdown("**Fitness weights**")
    cols = st.columns(len(ea.FITNESS_WEIGHTS))
    labels = {"structure": "Category balance", "amounts": "Plausible amounts",
              "edible": "Human-edible",
              "safety": f"Safe for {animal}", "animal": f"{animal.title()}-like"}
    weights = {k: col.slider(labels[k], 0.0, 1.0, v, 0.05)
               for col, (k, v) in zip(cols, ea.FITNESS_WEIGHTS.items())}
    seed_text = st.text_input("Random seed (empty = new every time)", "")

if st.button(f"Bake {animal} cookies", type="primary", use_container_width=True):
    seed = int(seed_text) if seed_text.strip().isdigit() else random.randint(0, 10**6)
    bar = st.progress(0.0, text="Evolving...")
    s = ea.Settings(animal=animal, generations=generations, pop=pop, recipes_in_book=n_book,
                    seed=seed, weights=weights)
    res = ea.run(s, progress=lambda g, t: bar.progress(g / t, text=f"Generation {g}/{t}"))
    bar.empty()
    st.session_state["res"] = (res, seed)


def ingredient_table(rec, animal):
    return pd.DataFrame([{
        "grams": i["grams"], "ingredient": i["ingredient"], "category": i["category"],
        "from": "🐾 animal food" if i["source"] == "animal food" else "🍪 cookie set",
        "human edible": EDIBLE[i["human_edible"]],
        f"for {animal}": f'{TIER[i["for_animal"]]} ({i["toxicity"]})',
        "Dutch name(s)": ", ".join(i["dutch_names"])} for i in rec["ingredients"]])


if "res" in st.session_state:
    res, seed = st.session_state["res"]
    if res.animal != animal:
        st.info(f"Showing the last {res.animal} result - press bake for {animal} cookies.")
    top, rest = res.book[0], res.book[1:]

    st.divider()
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.caption(f"TOP RECIPE · {ICONS.get(res.animal, '')} {res.animal} · seed {seed}")
        st.header(top["name"])
        st.markdown(f"**Makes about {int(res.batch_grams // 25)} cookies** "
                    f"({res.batch_grams:.0f} g dough)")
        st.dataframe(ingredient_table(top, res.animal), hide_index=True,
                     use_container_width=True, height=36 * (len(top["ingredients"]) + 1) + 4)
        st.subheader("Method")
        st.markdown("\n".join(f"{n}. {step}" for n, step in enumerate(top["method"], 1)))
    with right:
        st.metric("Fitness", f"{top['fitness']:.2f}",
                  help="Weighted mix of the scores below, ×0.3 per inedible or toxic ingredient")
        for k, v in top["scores"].items():
            st.progress(min(v, 1.0), text=f"{labels.get(k, k)}: {v:.2f}")
        if top["n_inedible"] or top["n_toxic"]:
            st.error(f"{top['n_inedible']} inedible and {top['n_toxic']} toxic ingredient(s)")
        else:
            st.success(f"Human-edible and no toxic ingredients for a {res.animal}")
        tot = sum(res.mutation_log.values()) or 1
        st.caption("Mutations applied: " + ", ".join(
            f"{k} {v / tot:.0%}" for k, v in sorted(res.mutation_log.items(), key=lambda x: -x[1])))

    if rest:
        st.subheader("More from this recipe book")
        for rec in rest:
            with st.expander(f"{rec['name']} — fitness {rec['fitness']:.2f}"):
                st.dataframe(ingredient_table(rec, res.animal), hide_index=True,
                             use_container_width=True,
                             height=36 * (len(rec["ingredients"]) + 1) + 4)
                st.markdown("\n".join(f"{n}. {s}" for n, s in enumerate(rec["method"], 1)))

    with st.expander("How the evolution went"):
        hist = pd.DataFrame(res.history).groupby("generation")[["best", "mean"]].mean()
        st.line_chart(hist, y_label="fitness")

    st.download_button("Download recipe book (JSON)",
                       json.dumps({"animal": res.animal, "seed": seed, "recipes": res.book},
                                  indent=2, ensure_ascii=False),
                       file_name=f"recipe_book_{res.animal}.json", mime="application/json")
