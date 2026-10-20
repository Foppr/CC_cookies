# Animal Cookie EA

Evolves cookies for cats, dogs or birds that a human can still eat.
Put this folder in the root of the `CC_cookies` repo. It reads the data files from there.

```
pip install deap numpy pandas streamlit
streamlit run app.py                         # interface: pick cat / dog / bird, bake
python animal_cookie_ea.py --animal bird     # command line (--generations, --pop, --seed ...)
```

## What it uses from the repo

| File | Role |
|---|---|
| `recipes_wide_format.csv` | **Starting point of the EA:** 837 categorised cookie recipes (initial population). Likes/views are not used |
| `Mutation (with ingridients)/ingredient_stats_categorized.csv` | The 118 cookie ingredients and their categories |
| `Mutation (with ingridients)/category_stats.csv` | Category targets for the structure fitness |
| `Catfood/`, `Dogfood/`, `Birdfood/` (`ingredient_dictionary.csv`, `products.csv`) | Ingredient pools for the mutation |
| `Scraped Data/Clean ingredients/ingredients_list_pet_toxicity_cooked.csv` | **New:** the toxicity file plus `*_toxicity_cooked` columns (ratings after baking) and a `cooked_note`. Used for good / ok / toxic. Falls back to `ingredients_list_pet_toxicity.csv` if missing |
| `data/animal_ingredient_labels.csv` (this folder) | **New:** translation and labels for the animal ingredients |

A `Fishfood/` folder with the same files is picked up automatically.

### The label table

The pet food ingredients are in Dutch, with many spelling variants, and their categories
were matched by name (chicken and salmon ended up as *flour (structure)*).
`data/animal_ingredient_labels.csv` covers the 516 most frequent ones (≥10 occurrences
in cat or dog food, ≥5 in bird food), which collapse into 221 canonical ingredients. Each row has:

- `ingredient`: English name. If it matches a cookie ingredient (haver → oats,
  honing → honey), both sets share the same gene.
- `category`: corrected cookie category (meat and fish → Other (Mix-ins), oils → fat, ...).
- `human_edible`: `yes` (ordinary food), `limited` (edible but odd, e.g. brewer's yeast,
  krill, inulin) or `no` (by-products, hydrolysates, meals, mineral premixes, cellulose, grit).

Rarer ingredients are skipped. Add a row to include one.

## The algorithm

- **Initial population**: 50% cookie recipes from the inspiring set, 50% random cookies.
- **Crossover**: role-based crossover with BLX-α blending (unchanged).
- **Mutation**: 25% substitute an ingredient with one from the animal set (80% within the
  same category), 25% add one from the animal set, 20% remove an ingredient, 30% change an amount.
  Animal ingredients are drawn by √(occurrences in that animal's food).
- **Selection / output**: tournament (k=3) with 2 elites per island, 3 islands, ring
  migration; recipe book via the team's `pick_book`; `output/recipe_book_<animal>.json` and
  `fitness_history_<animal>.csv`.

## Fitness: a good cookie that humans can eat

| Part | Weight | What it measures |
|---|---|---|
| structure | 0.30 | Team's `CategoryStatsFitness`: category shares vs `category_stats.csv` |
| amounts | 0.15 | Each ingredient's share vs its typical share (within 2.5× = 1 sd) |
| edible | 0.20 | Human edibility (yes 1, limited 0.5, no 0) |
| safety | 0.20 | Good / ok / toxic for the animal (1 / 0.6 / 0) |
| animal | 0.15 | Share of dough from ingredients more typical of the animal's food than of cookies (20% = full score) |

Each inedible ingredient and each toxic ingredient multiplies the total by 0.3.
The like/view ratio is not part of the fitness.

### Good / ok / toxic per animal

- Cookie ingredients use the **baked** ratings (`*_toxicity_cooked`) from the Clean ingredients folder:
  **0 = good, 1-2 = ok, 3-4 = toxic**. Changed from raw: egg, egg yolk and egg white 1 → 0 (salmonella
  and avidin are destroyed by cooking); sourdough discard 3 → 1 (yeast killed); baking soda/powder 2 → 1
  (reacted during baking); extracts and vanilla bean paste one level lower (most alcohol evaporates).
  Theobromine, caffeine, xylitol, salt, lactose and mould toxins are not affected by baking, so those stay.
- Animal-food ingredients have no toxicity score. They count as **good** for the animal whose food
  contains them, and **ok** (unknown) for the other animals. The mutation only draws from the chosen
  animal's own food, so in practice they are good.
- When an animal-food ingredient is also a cookie ingredient (e.g. raisins in bird food), the
  toxicity file wins.

**Two changes to `CategoryStatsFitness`.** Please review them with whoever owns it:
1. The category means add up to about 75%, because each is the share *when the category is present*.
   The fitness therefore rewarded putting the remaining ~25% of the dough in unmeasured
   categories, and the EA filled cookies with 200+ g of lecithin and psyllium. The targets are now rescaled to sum to 100%.
2. Categories seen in fewer than 50 recipes are ignored (`Other` is based on 2 recipes).
