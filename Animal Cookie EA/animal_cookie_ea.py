#!/usr/bin/env python3
"""
animal_cookie_ea.py - evolve human-edible cookies for cats, dogs or birds (DEAP)
===============================================================================

Start            The inspiring set is the categorised cookie recipes
                 (recipes_wide_format.csv). Half of the initial population is
                 taken from them, the rest is random cookies.
Genome           {ingredient: proportion}, proportions sum to 1.
Crossover        Role-based crossover with BLX-alpha blending on shared ingredients.
Mutation         Draws new ingredients from the chosen animal's food database
                 (Catfood/, Dogfood/, Birdfood/ - and Fishfood/ if it is added):
                     25%  substitute an ingredient with one from the animal set
                     25%  add an ingredient from the animal set
                     20%  remove an ingredient from the recipe
                     30%  change an amount
Fitness          Is it a good cookie that a human could still eat?
                     structure   category proportions vs category_stats.csv (team's function)
                     amounts     each ingredient's share vs its typical share in cookies
                     edible      human edibility of the ingredients (yes / limited / no)
                     safety      good / ok / toxic for the chosen animal, from the 0-4 toxicity
                                 scale in Scraped Data/Clean ingredients (0 good, 1-2 ok, 3-4 toxic)
                     animal      share of ingredients typical of the animal's food
                 Every inedible ingredient and every toxic ingredient
                 multiplies the total by 0.3.
Selection        Tournament (k=3) with elitism, 3 islands with ring migration.
Output           Recipe book printed + recipe_book.json + fitness_history.csv;
                 the top recipe is shown in full, with a method.

Usage
    pip install deap numpy
    python animal_cookie_ea.py --animal dog
    python animal_cookie_ea.py --animal bird --generations 300 --seed 3
    streamlit run app.py            # interface
"""

import argparse
import csv
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from deap import algorithms, base, creator, tools

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# ============================================================================
# CONFIGURATION
# ============================================================================

PATHS = {
    "cookie_ingredients": REPO / "Mutation (with ingridients)" / "ingredient_stats_categorized.csv",
    "category_stats": REPO / "Mutation (with ingridients)" / "category_stats.csv",
    "recipes": REPO / "recipes_wide_format.csv",
    # cooked ratings (eggs etc. are safe once baked); falls back to the raw file if missing
    "toxicity": REPO / "Scraped Data" / "Clean ingredients" / "ingredients_list_pet_toxicity_cooked.csv",
    "toxicity_raw": REPO / "Scraped Data" / "Clean ingredients" / "ingredients_list_pet_toxicity.csv",
    "labels": HERE / "data" / "animal_ingredient_labels.csv",
}
ANIMAL_FOLDERS = {"cat": "Catfood", "dog": "Dogfood", "bird": "Birdfood", "fish": "Fishfood"}

ROLES = ["flour (structure)", "fat", "sugar", "egg", "leavening agents",
         "salt/flavorings", "Other (Mix-ins)", "Other (Liquids)", "Other"]
ESSENTIAL_ROLES = ["flour (structure)", "fat", "sugar", "egg"]
MAX_PER_ROLE = {"flour (structure)": 2, "fat": 2, "sugar": 3, "egg": 2, "leavening agents": 2,
                "salt/flavorings": 3, "Other (Mix-ins)": 3, "Other (Liquids)": 2, "Other": 2}
MAX_INGREDIENTS = 14
MIN_PROPORTION = 0.0003

BLX_ALPHA = 0.25
MUTATION_WEIGHTS = {"substitute": 0.25, "add": 0.25, "remove": 0.20, "amount": 0.30}
AMOUNT_SIGMA = 0.15
SAME_CATEGORY_PROB = 0.8      # substitute: chance to stay within the replaced ingredient's category

FITNESS_WEIGHTS = {"structure": 0.30, "amounts": 0.15, "edible": 0.20, "safety": 0.20,
                   "animal": 0.15}
MIN_CATEGORY_RECIPES = 50     # category targets based on fewer recipes are ignored
PENALTY = 0.3                 # multiplier per inedible or highly toxic ingredient
ANIMAL_SHARE_TARGET = 0.20    # 20% of the dough from animal-typical ingredients = full score

EDIBLE_SCORE = {"yes": 1.0, "limited": 0.5, "no": 0.0}
# Division of every ingredient per animal into good / ok / toxic:
#   cookie ingredients -> baked toxicity 0-4 from ingredients_list_pet_toxicity_cooked.csv:
#                         0 = good, 1-2 = ok, 3-4 = toxic
#   animal-food ingredients without a toxicity score -> good for the animal whose food
#                         contains them, ok (unknown) for the other animals
def toxicity_tier(level):
    return "good" if level == 0 else ("ok" if level <= 2 else "toxic")


TIER_SCORE = {"good": 1.0, "ok": 0.6, "toxic": 0.0}
UNKNOWN_TOXICITY = 1

COOKIE_GRAMS = 25            # dough per cookie
OUTPUT_DIVISOR = 4           # the EA evolves an 800 g batch; the output is divided by 4 -> 200 g = 8 cookies

MEAT_FISH = {"chicken", "turkey", "salmon", "tuna", "beef", "lamb", "duck", "pork", "poultry",
             "rabbit", "venison", "white fish", "herring", "sardines", "shrimp", "shellfish",
             "liver", "chicken liver", "poultry liver", "pork liver", "heart", "meat fillets",
             "chicken organ meat"}


def available_animals(repo=REPO):
    return [a for a, folder in ANIMAL_FOLDERS.items()
            if (repo / folder / "ingredient_dictionary.csv").exists()]


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ============================================================================
# INGREDIENT DATABASE: cookie ingredients + animal food ingredients
# ============================================================================

class IngredientDB:
    """One ingredient universe: the 118 cookie ingredients plus the (translated,
    deduplicated) frequent ingredients of every animal food database."""

    def __init__(self, repo=REPO):
        self.names, self.role, self.median, self.popularity = [], {}, {}, {}
        self.edible, self.source, self.dutch = {}, {}, defaultdict(set)

        cat_share = {r["category"]: float(r["pct_of_recipe_mean"]) / 100
                     for r in read_csv(PATHS["category_stats"])}

        cookie_rows = read_csv(PATHS["cookie_ingredients"])
        n_cookie = max(int(r["recipe_count_"]) for r in cookie_rows)
        for r in cookie_rows:
            name = r["ingredient"].strip().lower()
            self._add(name, r["category"].strip(), "yes", "cookie")
            self.median[name] = float(r["pct_of_recipe_mean"] or 0) / 100 or cat_share[self.role[name]]
            self.popularity[name] = int(r["recipe_count_"]) / n_cookie
        self.cookie_names = set(self.names)

        # toxicity of cookie ingredients for each animal (0 safe .. 4 severe)
        self.toxicity = defaultdict(dict)
        tox_file = PATHS["toxicity"] if PATHS["toxicity"].exists() else PATHS["toxicity_raw"]
        self.toxicity_source = tox_file.name
        for r in read_csv(tox_file):
            for animal in ANIMAL_FOLDERS:
                v = (r.get(f"{animal}_toxicity_cooked") or r.get(f"{animal}_toxicity") or "").strip()
                if v:
                    self.toxicity[animal][r["ingredient"].strip().lower()] = int(v)

        # animal food databases, mapped through the label table
        labels = {r["ingredient_nl"]: r for r in read_csv(PATHS["labels"])}
        self.pool, self.pool_share, self.unlabeled, self.n_products = {}, {}, {}, {}
        for animal in available_animals(repo):
            folder = repo / ANIMAL_FOLDERS[animal]
            products = read_csv(folder / "products.csv")
            self.n_products[animal] = max(sum(1 for p in products
                                              if int(p.get("n_ingredients") or 0) > 0), 1)
            occ, unlabeled = Counter(), 0
            for r in read_csv(folder / "ingredient_dictionary.csv"):
                lab = labels.get(r["ingredient"])
                if lab is None:
                    unlabeled += 1
                    continue
                name = lab["ingredient"]
                occ[name] += int(r["occurrences"])
                self.dutch[name].add(r["ingredient"])
                if name not in self.role:
                    self._add(name, lab["category"], lab["human_edible"], "animal food")
                    self.median[name] = (0.01 if lab["category"] == "Other"
                                         else cat_share[lab["category"]] * 0.5)
                    self.popularity[name] = 0.0
            self.pool[animal] = dict(occ)
            self.pool_share[animal] = {i: min(o / self.n_products[animal], 1.0)
                                       for i, o in occ.items()}
            self.unlabeled[animal] = unlabeled

        self.index = {n: i for i, n in enumerate(self.names)}
        self.by_role = defaultdict(list)
        for n in self.names:
            self.by_role[self.role[n]].append(n)
        self.similarity = None

    def _add(self, name, role, edible, source):
        if role not in ROLES:
            raise ValueError(f"unknown category '{role}' for '{name}'")
        self.names.append(name)
        self.role[name], self.edible[name], self.source[name] = role, edible, source

    def tox(self, animal, ing):
        if ing in self.toxicity[animal]:
            return self.toxicity[animal][ing]
        return 0 if ing in self.pool.get(animal, {}) else UNKNOWN_TOXICITY

    def tier(self, animal, ing):
        return toxicity_tier(self.tox(animal, ing))

    def is_typical(self, animal, ing, cookie_share):
        """Is this ingredient more characteristic of the animal's food than of cookies?"""
        return self.pool_share[animal].get(ing, 0) > cookie_share.get(ing, 0)

    def vectorize(self, recipe):
        v = np.zeros(len(self.names))
        for ing, p in recipe.items():
            v[self.index[ing]] = p
        return v

    def build_similarity(self, recipes):
        n = len(self.names)
        C = np.zeros((n, n))
        for rec in recipes:
            idx = [self.index[i] for i in rec]
            for a in idx:
                C[a, idx] += 1
        np.fill_diagonal(C, 0)
        norms = np.linalg.norm(C, axis=1, keepdims=True)
        norms[norms == 0] = 1
        V = C / norms
        self.similarity = V @ V.T

    def sim(self, a, b):
        return 0.0 if self.similarity is None else float(self.similarity[self.index[a], self.index[b]])


# ============================================================================
# RECIPE DATABASE: the categorised cookie recipes (wide format)
# ============================================================================

class RecipeDB:
    """Reads recipes_wide_format.csv (grams_<ingredient> columns). Likes/views are not used."""

    def __init__(self, db, path=PATHS["recipes"]):
        self.db = db
        self.recipes = {}
        for r in read_csv(path):
            grams = {}
            for col, val in r.items():
                if col.startswith("grams_") and val and float(val) > 0:
                    ing = col[6:].strip().lower()
                    if ing in db.index:
                        grams[ing] = grams.get(ing, 0) + float(val)
            total = sum(grams.values())
            if total <= 0:
                continue
            rid = r["recipe_id"]
            self.recipes[rid] = {i: g / total for i, g in grams.items()}
        self.ids = list(self.recipes)
        self.ingredient_share = {i: c / len(self.recipes) for i, c in
                                 Counter(i for rec in self.recipes.values() for i in rec).items()}

        print(f"[RecipeDB] {len(self.recipes)} cookie recipes loaded as the inspiring set")


# ============================================================================
# FITNESS  - is it a good cookie that humans can still eat?
# ============================================================================

def role_totals(recipe, db):
    t = defaultdict(float)
    for ing, p in recipe.items():
        t[db.role[ing]] += p
    return t


def weighted_mean(recipe, score):
    s = w = 0.0
    for ing, p in recipe.items():
        wi = math.sqrt(p)
        s += wi * score(ing)
        w += wi
    return s / w if w else 0.0


class FitnessModel:
    name = "base"

    def evaluate(self, recipe):
        raise NotImplementedError


class CategoryStatsFitness(FitnessModel):
    """(Team's function) category proportions vs the means in category_stats.csv."""
    name = "structure"

    def __init__(self, db, category_stats_csv=PATHS["category_stats"], deviation_scalar=-20,
                 min_recipes=MIN_CATEGORY_RECIPES):
        self.db, self.dev_scalar = db, deviation_scalar
        # categories seen in very few recipes ("Other": 2) would force odd fillers in
        self.targets = {r["category"]: float(r["pct_of_recipe_mean"]) / 100
                        for r in read_csv(category_stats_csv)
                        if int(r["recipe_count"]) >= min_recipes}
        # The means are "share when the category is present" and add up to ~67%, which
        # rewards putting a third of the dough in unmeasured categories. Rescale to 100%.
        total = sum(self.targets.values())
        self.targets = {c: v / total for c, v in self.targets.items()}

    def evaluate(self, recipe):
        t = role_totals(recipe, self.db)
        sse = sum((t.get(c, 0.0) - target) ** 2 for c, target in self.targets.items())
        return math.exp(self.dev_scalar * sse)


class AmountsFitness(FitnessModel):
    """Is every amount plausible? Each ingredient's share is compared with its typical
    share (cookie ingredients: mean % in the cookie recipes; animal ingredients: half the
    category mean). Within 2.5x counts as one standard deviation."""
    name = "amounts"

    def __init__(self, db):
        self.db = db

    def evaluate(self, recipe):
        def score(ing):
            dev = math.log(recipe[ing] / self.db.median[ing]) / math.log(2.5)
            return math.exp(-0.5 * dev * dev)
        return weighted_mean(recipe, score)


class HumanEdibleFitness(FitnessModel):
    """Could a person eat this cookie? yes 1, limited 0.5, no 0 (amount-weighted)."""
    name = "edible"

    def __init__(self, db):
        self.db = db

    def evaluate(self, recipe):
        return weighted_mean(recipe, lambda i: EDIBLE_SCORE[self.db.edible[i]])

    def n_bad(self, recipe):
        return sum(1 for i in recipe if self.db.edible[i] == "no")


class AnimalSafetyFitness(FitnessModel):
    """Is it safe for the animal? good 1.0, ok 0.6, toxic 0.0 (amount-weighted)."""
    name = "safety"

    def __init__(self, db, animal):
        self.db, self.animal = db, animal

    def evaluate(self, recipe):
        return weighted_mean(recipe, lambda i: TIER_SCORE[self.db.tier(self.animal, i)])

    def n_bad(self, recipe):
        return sum(1 for i in recipe if self.db.tier(self.animal, i) == "toxic")


class AnimalMatchFitness(FitnessModel):
    """Is it recognisably a cat/dog/bird cookie? Share of the dough made of
    ingredients that are more typical of the animal's food than of cookies."""
    name = "animal"

    def __init__(self, db, rdb, animal, target=ANIMAL_SHARE_TARGET):
        self.db, self.animal, self.target = db, animal, target
        self.cookie_share = rdb.ingredient_share

    def evaluate(self, recipe):
        share = sum(p for i, p in recipe.items()          # fibres/additives ("Other") don't count
                    if self.db.role[i] != "Other"
                    and self.db.is_typical(self.animal, i, self.cookie_share))
        return min(share / self.target, 1.0)


class AnimalCookieFitness(FitnessModel):
    name = "animal cookie"

    def __init__(self, db, rdb, animal, weights=None):
        self.weights = dict(weights or FITNESS_WEIGHTS)
        self.edible = HumanEdibleFitness(db)
        self.safety = AnimalSafetyFitness(db, animal)
        self.parts = {"structure": CategoryStatsFitness(db), "amounts": AmountsFitness(db),
                      "edible": self.edible, "safety": self.safety,
                      "animal": AnimalMatchFitness(db, rdb, animal)}

    def breakdown(self, recipe):
        scores = {n: m.evaluate(recipe) for n, m in self.parts.items() if self.weights.get(n, 0) > 0}
        total_w = sum(self.weights[n] for n in scores)
        n_inedible, n_toxic = self.edible.n_bad(recipe), self.safety.n_bad(recipe)
        total = (sum(scores[n] * self.weights[n] for n in scores) / total_w
                 * PENALTY ** (n_inedible + n_toxic))
        return total, scores, n_inedible, n_toxic

    def evaluate(self, recipe):
        return self.breakdown(recipe)[0]


# ============================================================================
# RECIPE HELPERS + REPAIR
# ============================================================================

def normalize(recipe):
    total = sum(recipe.values())
    if total > 0:
        for k in recipe:
            recipe[k] /= total


def role_groups(recipe, db):
    g = defaultdict(dict)
    for ing, p in recipe.items():
        g[db.role[ing]][ing] = p
    return g


def weighted_pick(cands, weights):
    return random.choices(cands, weights=weights, k=1)[0]


def pick_popular(db, role, exclude=()):
    cands = [n for n in db.by_role[role] if n not in exclude and n in db.cookie_names]
    return weighted_pick(cands, [db.popularity[n] + 0.002 for n in cands]) if cands else None


def is_removable(recipe, ing, db):
    role = db.role[ing]
    return role not in ESSENTIAL_ROLES or sum(1 for i in recipe if db.role[i] == role) > 1


def repair(recipe, db):
    for k in [k for k, v in recipe.items() if v <= 0]:
        del recipe[k]
    normalize(recipe)
    for k in [k for k, v in recipe.items() if v < MIN_PROPORTION]:
        del recipe[k]
    for role, group in role_groups(recipe, db).items():
        excess = len(group) - MAX_PER_ROLE.get(role, 2)
        for ing in sorted(group, key=group.get)[:max(excess, 0)]:
            del recipe[ing]
    for role in ESSENTIAL_ROLES:
        if not any(db.role[i] == role for i in recipe):
            ing = pick_popular(db, role, exclude=recipe)
            if ing:
                recipe[ing] = db.median[ing]
    while len(recipe) > MAX_INGREDIENTS:
        removable = [i for i in recipe if is_removable(recipe, i, db)]
        del recipe[min(removable, key=recipe.get)]
    normalize(recipe)
    return recipe


def random_recipe(db):
    """A random cookie built from the cookie ingredients (initial population)."""
    plan = {"flour (structure)": (1, 1.0), "fat": (1, 1.0), "sugar": (2, 0.9), "egg": (1, 0.9),
            "leavening agents": (1, 0.8), "salt/flavorings": (2, 0.9), "Other (Mix-ins)": (2, 0.6)}
    rec = {}
    for role, (max_n, prob) in plan.items():
        for _ in range(random.randint(1, max_n)):
            if random.random() < prob:
                ing = pick_popular(db, role, exclude=rec)
                if ing:
                    rec[ing] = db.median[ing] * math.exp(random.gauss(0, 0.3))
    return repair(rec, db)


def recipe_distance(a, b):
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


# ============================================================================
# VARIATION OPERATORS
# ============================================================================

def blx(a, b, alpha=BLX_ALPHA):
    lo, hi = min(a, b), max(a, b)
    d = hi - lo
    return max(random.uniform(lo - alpha * d, hi + alpha * d), 1e-6)


def cx_role_blend(ind1, ind2, db):
    """Per category, child 1 takes the whole ingredient group of one parent and child 2
    the other's; ingredients both parents share get a blended (BLX-alpha) amount."""
    g1, g2 = role_groups(ind1, db), role_groups(ind2, db)
    c1, c2 = {}, {}
    for role in ROLES:
        a, b = g1.get(role, {}), g2.get(role, {})
        if not a and not b:
            continue
        first, second = (a, b) if random.random() < 0.5 else (b, a)
        for child, own, other in ((c1, first, second), (c2, second, first)):
            for ing, amt in own.items():
                child[ing] = blx(amt, other[ing]) if ing in other else amt
    for ind, child in ((ind1, c1), (ind2, c2)):
        ind.clear()
        ind.update(child)
        repair(ind, db)
    return ind1, ind2


class AnimalMutation:
    """The four mutation types; new ingredients come from the animal food database,
    sampled by how often they occur in that animal's food (sqrt-flattened)."""

    def __init__(self, db, animal):
        self.db, self.animal = db, animal
        pool = db.pool[animal]
        self.cands = list(pool)
        self.weights = [math.sqrt(pool[i]) for i in self.cands]
        self.by_role = defaultdict(list)
        for i in self.cands:
            self.by_role[db.role[i]].append(i)
        self.log = Counter()

    def _draw(self, exclude, role=None):
        cands = self.by_role[role] if role else self.cands
        cands = [c for c in cands if c not in exclude]
        if not cands:
            return None
        pool = self.db.pool[self.animal]
        return weighted_pick(cands, [math.sqrt(pool[c]) for c in cands])

    def substitute(self, ind):
        old = random.choice(list(ind))
        role = self.db.role[old] if random.random() < SAME_CATEGORY_PROB else None
        new = self._draw(ind, role) or self._draw(ind)
        if new is None:
            return False
        ind[new] = ind.pop(old)
        return True

    def add(self, ind):                      # repair trims the recipe if it gets too long
        new = self._draw(ind)
        if new is None:
            return False
        ind[new] = self.db.median[new] * math.exp(random.gauss(0, 0.3))
        return True

    def remove(self, ind):
        if len(ind) <= 3:
            return False
        del ind[random.choice(list(ind))]
        return True

    def amount(self, ind):
        ing = random.choice(list(ind))
        ind[ing] *= math.exp(random.gauss(0, AMOUNT_SIGMA))
        return True

    def __call__(self, ind):
        op = random.choices(list(MUTATION_WEIGHTS), weights=list(MUTATION_WEIGHTS.values()))[0]
        if not getattr(self, op)(ind):
            op = "amount"
            self.amount(ind)
        self.log[op] += 1
        repair(ind, self.db)
        return (ind,)


# ============================================================================
# EVOLUTION (island model, tournament selection)
# ============================================================================

@dataclass
class Settings:
    animal: str = "dog"
    generations: int = 200
    pop: int = 80
    islands: int = 3
    recipes_in_book: int = 3
    cxpb: float = 0.7
    mutpb: float = 0.5
    tournament: int = 3
    elite: int = 2
    migration_every: int = 40
    migrants: int = 1
    seed_fraction: float = 0.5
    min_distance: float = 0.4
    batch_grams: float = 800          # batch the EA works with (proportions are unaffected)
    output_divisor: float = OUTPUT_DIVISOR   # final amounts are divided by this (800 g / 4 = 8 cookies)
    seed: int | None = None
    weights: dict = field(default_factory=lambda: dict(FITNESS_WEIGHTS))


@dataclass
class Result:
    animal: str
    book: list
    history: list
    mutation_log: dict
    batch_grams: float
    stats: dict


_CACHE = {}


def load_data():
    """Load ingredient + recipe databases once (the interface reuses them)."""
    if "data" not in _CACHE:
        db = IngredientDB()
        rdb = RecipeDB(db)
        db.build_similarity(rdb.recipes.values())
        _CACHE["data"] = (db, rdb)
    return _CACHE["data"]


def setup_deap(db, rdb, fitness, mutation, s):
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", dict, fitness=creator.FitnessMax)
    seeds = list(rdb.recipes.values())

    def init_individual():
        if random.random() < s.seed_fraction:
            ind = creator.Individual(random.choice(seeds))
        else:
            ind = creator.Individual(random_recipe(db))
        return repair(ind, db)

    tb = base.Toolbox()
    tb.register("individual", init_individual)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", lambda ind: (fitness.evaluate(ind),))
    tb.register("mate", cx_role_blend, db=db)
    tb.register("mutate", mutation)
    tb.register("select", tools.selTournament, tournsize=s.tournament)
    return tb


def evaluate_invalid(pop, tb):
    for ind in pop:
        if not ind.fitness.valid:
            ind.fitness.values = tb.evaluate(ind)


def evolve(tb, s, progress=None):
    islands = [tb.population(n=s.pop) for _ in range(s.islands)]
    for pop in islands:
        evaluate_invalid(pop, tb)
    hofs = [tools.HallOfFame(30) for _ in islands]
    history = []
    for gen in range(1, s.generations + 1):
        for i, pop in enumerate(islands):
            elites = [tb.clone(e) for e in tools.selBest(pop, s.elite)]
            offspring = [tb.clone(o) for o in tb.select(pop, len(pop) - s.elite)]
            offspring = algorithms.varAnd(offspring, tb, s.cxpb, s.mutpb)
            evaluate_invalid(offspring, tb)
            pop[:] = elites + offspring
            hofs[i].update(pop)
            fits = [ind.fitness.values[0] for ind in pop]
            history.append({"generation": gen, "island": i + 1, "best": max(fits),
                            "mean": float(np.mean(fits)),
                            "mean_ingredients": float(np.mean([len(p) for p in pop]))})
        if s.islands > 1 and gen % s.migration_every == 0:
            tools.migRing(islands, s.migrants, tools.selBest)
            for pop in islands:
                pop[:] = [tb.clone(ind) for ind in pop]
        if progress:
            progress(gen, s.generations)
    return islands, hofs, history


def pick_book(islands, hofs, n_recipes, min_distance):
    """(Team's) greedy diverse selection from halls of fame + final populations."""
    pool = [ind for hof in hofs for ind in hof] + [ind for pop in islands for ind in pop]
    pool.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
    book, d = [], min_distance
    while len(book) < n_recipes and d >= 0:
        for cand in pool:
            if len(book) == n_recipes:
                break
            if all(recipe_distance(cand, b) >= max(d, 1e-9) for b in book):
                book.append(cand)
        d -= 0.1
    return book


def run(s: Settings, progress=None) -> Result:
    if s.seed is not None:
        random.seed(s.seed)
        np.random.seed(s.seed)
    db, rdb = load_data()
    if s.animal not in db.pool:
        raise ValueError(f"no food database for '{s.animal}' (found: {', '.join(db.pool)})")
    fitness = AnimalCookieFitness(db, rdb, s.animal, s.weights)
    mutation = AnimalMutation(db, s.animal)
    tb = setup_deap(db, rdb, fitness, mutation, s)
    islands, hofs, history = evolve(tb, s, progress)
    book = pick_book(islands, hofs, s.recipes_in_book, s.min_distance)
    stats = {"cookie_recipes": len(rdb.recipes), "animal_ingredients": len(db.pool[s.animal]),
             "animal_products": db.n_products[s.animal], "unlabeled_variants": db.unlabeled[s.animal]}
    book = [divide_amounts(describe(r, db, fitness, s), s.output_divisor) for r in book]
    return Result(s.animal, book, history, dict(mutation.log),
                  s.batch_grams / s.output_divisor, stats)


def divide_amounts(described, divisor=OUTPUT_DIVISOR):
    """Scale a finished recipe down: every amount is divided by `divisor`.
    Proportions stay exactly the same (800 g batch / 4 = 200 g = 8 cookies of 25 g)."""
    out = dict(described)
    out["ingredients"] = [{**i, "grams": round(i["grams"] / divisor, 1)}
                          for i in described["ingredients"]]
    out["total_grams"] = round(sum(i["grams"] for i in described["ingredients"]) / divisor, 1)
    out["cookies"] = int(round(out["total_grams"] / COOKIE_GRAMS))
    return out


# ============================================================================
# OUTPUT
# ============================================================================

ANIMAL_NAME = {"cat": "Kitty", "dog": "Doggy", "bird": "Birdie", "fish": "Fishy"}


def recipe_name(recipe, db, animal):
    feature = [i for i in recipe if db.role[i] in ("Other (Mix-ins)", "salt/flavorings", "fat")
               and i not in ("salt", "water", "butter", "oil")]
    from_animal = sorted((i for i in feature if db.source[i] == "animal food"),
                         key=recipe.get, reverse=True)
    others = sorted((i for i in feature if i not in from_animal and db.role[i] != "fat"),
                    key=recipe.get, reverse=True)
    words = (from_animal + others)[:2]
    label = " & ".join(w.title() for w in words) if words else "Plain"
    return f"{ANIMAL_NAME.get(animal, animal.title())} {label} Cookies"


def method_steps(recipe, db):
    """A simple cookie method generated from the ingredient categories."""
    by = role_groups(recipe, db)
    names = lambda role: ", ".join(sorted(by.get(role, {}), key=by[role].get, reverse=True))
    steps = ["Preheat the oven to 175 °C (350 °F) and line a baking tray."]
    if any(i in MEAT_FISH for i in recipe):
        steps.append("Cook the " + ", ".join(i for i in recipe if i in MEAT_FISH)
                     + " fully, let it cool and chop it very finely.")
    if by.get("fat") or by.get("sugar"):
        steps.append(f"Cream {names('fat') or 'the fat'} with {names('sugar') or 'the sugar'} "
                     "until light, about 2 minutes.")
    wet = [r for r in ("egg", "Other (Liquids)") if by.get(r)]
    if wet:
        steps.append("Beat in " + " and ".join(names(r) for r in wet) + ".")
    dry = [r for r in ("flour (structure)", "leavening agents", "salt/flavorings", "Other")
           if by.get(r)]
    if dry:
        steps.append("In another bowl whisk together " + ", ".join(names(r) for r in dry)
                     + ", then stir this into the wet mixture until just combined.")
    if by.get("Other (Mix-ins)"):
        steps.append(f"Fold in {names('Other (Mix-ins)')}.")
    steps.append("Scoop 25 g balls onto the trays, 5 cm apart, and bake 10-12 minutes until "
                 "the edges are golden. Cool on the tray for 5 minutes.")
    return steps


def describe(recipe, db, fitness, s):
    total, scores, n_inedible, n_toxic = fitness.breakdown(recipe)
    ings = sorted(recipe.items(), key=lambda x: (ROLES.index(db.role[x[0]]), -x[1]))
    return {
        "name": recipe_name(recipe, db, s.animal),
        "fitness": round(total, 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "n_inedible": n_inedible, "n_toxic": n_toxic,
        "ingredients": [{
            "ingredient": i, "category": db.role[i], "grams": round(p * s.batch_grams, 1),
            "share": round(p, 4), "source": db.source[i],
            "in_animal_food": i in db.pool[s.animal],
            "human_edible": db.edible[i], "toxicity": db.tox(s.animal, i),
            "for_animal": db.tier(s.animal, i),
            "dutch_names": sorted(db.dutch.get(i, []))[:4]} for i, p in ings],
        "method": method_steps(recipe, db),
    }


def fmt_grams(g):
    return f"{g:.0f}" if g >= 10 else f"{g:.1f}"


def print_result(res):
    print("\n" + "=" * 72)
    print(f"{res.animal.upper()} COOKIE BOOK  ({res.batch_grams:.0f} g dough = "
          f"{int(res.batch_grams // COOKIE_GRAMS)} cookies; "
          f"mutations drawn from {res.stats['animal_ingredients']} {res.animal}-food ingredients)")
    print("=" * 72)
    for n, r in enumerate(res.book, 1):
        s = r["scores"]
        print(f"\n{n}. {r['name']}   fitness {r['fitness']:.3f}   "
              + "  ".join(f"{k} {v:.2f}" for k, v in s.items())
              + f"   inedible {r['n_inedible']}  toxic {r['n_toxic']}")
        for i in r["ingredients"]:
            mark = "*" if i["source"] == "animal food" else ("~" if i["in_animal_food"] else " ")
            print(f"   {fmt_grams(i['grams']):>6} g {mark} {i['ingredient']:<28} "
                  f"[{i['category']}]  edible:{i['human_edible']}  {res.animal}:{i['for_animal']}")
        if n == 1:
            print("   Method:")
            for k, step in enumerate(r["method"], 1):
                print(f"     {k}. {step}")
    print("\n   * = from the animal food database   ~ = cookie ingredient also found in it")
    tot = sum(res.mutation_log.values()) or 1
    print("Mutations: " + ", ".join(f"{k} {v / tot:.0%}" for k, v in
                                    sorted(res.mutation_log.items(), key=lambda x: -x[1])))


def save_outputs(res, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"recipe_book_{res.animal}.json", "w", encoding="utf-8") as f:
        json.dump({"animal": res.animal, "batch_grams": res.batch_grams, "recipes": res.book},
                  f, indent=2, ensure_ascii=False)
    with open(out / f"fitness_history_{res.animal}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res.history[0]))
        w.writeheader()
        w.writerows(res.history)
    print(f"Saved {out}/recipe_book_{res.animal}.json and fitness_history_{res.animal}.csv")


def main():
    p = argparse.ArgumentParser(description="Evolve human-edible cookies for an animal.")
    p.add_argument("--animal", default="dog", choices=list(ANIMAL_FOLDERS))
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--pop", type=int, default=80, help="population per island")
    p.add_argument("--islands", type=int, default=3)
    p.add_argument("--recipes-in-book", type=int, default=3)
    p.add_argument("--cxpb", type=float, default=0.7)
    p.add_argument("--mutpb", type=float, default=0.5)
    p.add_argument("--tournament", type=int, default=3)
    p.add_argument("--elite", type=int, default=2)
    p.add_argument("--seed-fraction", type=float, default=0.5)
    p.add_argument("--batch-grams", type=float, default=800, help="batch the EA works with")
    p.add_argument("--divisor", type=float, default=OUTPUT_DIVISOR,
                   help="divide the final amounts by this (default 4: 800 g -> 8 cookies)")
    p.add_argument("--seed", type=int)
    p.add_argument("--out-dir", default="output")
    a = p.parse_args()
    s = Settings(animal=a.animal, generations=a.generations, pop=a.pop, islands=a.islands,
                 recipes_in_book=a.recipes_in_book, cxpb=a.cxpb, mutpb=a.mutpb,
                 tournament=a.tournament, elite=a.elite, seed_fraction=a.seed_fraction,
                 batch_grams=a.batch_grams, output_divisor=a.divisor, seed=a.seed)

    def progress(g, total):
        if g == 1 or g % 25 == 0 or g == total:
            print(f"  generation {g}/{total}")

    res = run(s, progress)
    print_result(res)
    save_outputs(res, a.out_dir)


if __name__ == "__main__":
    main()
