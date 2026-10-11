#!/usr/bin/env python3
"""
cookie_ea.py - Evolutionary cookie recipe generator (DEAP starter framework)
===========================================================================

Genome
    A recipe is a dict {ingredient_name: proportion}, proportions sum to 1.0
    (fraction of total dough weight). Every ingredient has a ROLE, read from
    ingredients.csv (structure, fat, sugar, binder, ...).

Variation operators
    Crossover : role-based crossover with blend (BLX-alpha) on shared ingredients
    Mutation  : amount perturbation 50% / substitution 25% / addition 15% / removal 10%
    Repair    : after every operator (normalise, enforce essential roles and caps)

Selection
    Tournament (k=3) + elitism, inside an island model (3 sub-populations with
    occasional ring migration). The book is picked greedily from all islands:
    best recipes first, each at least --min-distance away from those already chosen.

Fitness  (pluggable - see section "FITNESS MODELS")
    * MockFitness          heuristic ratio score; works with no data at all
    * RatingKNNFitness     predicts a like/view score from the k most similar rated recipes
    * NoveltyFitness       rewards a moderate distance from known recipes
    * CombinedFitness      weighted sum of any of the above
    To plug in your team's fitness function: subclass FitnessModel, implement
    evaluate(recipe) -> float (higher = better) and pass it in build_fitness().

Data (optional, for when the files arrive)
    --recipes  CSV, long format:  recipe_id,ingredient,grams
               ingredient names must match ingredients.csv
    --ratings  CSV:               recipe_id,likes,views
    Test the data pipeline with fake data first:
        python cookie_ea.py --make-demo-data
        python cookie_ea.py --recipes demo_recipes.csv --ratings demo_ratings.csv

Usage
    pip install deap numpy
    python cookie_ea.py                          # mock fitness, default settings
    python cookie_ea.py --generations 200 --pop 120 --seed 7
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from deap import algorithms, base, creator, tools

# ============================================================================
# CONFIGURATION
# ============================================================================

ROLES = ["structure", "fat", "sugar", "binder", "leavener", "salt",
         "liquid", "texturizer", "flavor", "add_in", "decoration"]

ESSENTIAL_ROLES = ["structure", "fat", "sugar"]   # every recipe must contain these

MAX_PER_ROLE = {"structure": 2, "fat": 2, "sugar": 3, "binder": 2, "leavener": 2,
                "salt": 1, "liquid": 2, "texturizer": 2, "flavor": 4,
                "add_in": 3, "decoration": 2}
MAX_INGREDIENTS = 14
MIN_PROPORTION = 0.0003          # below 0.03% of the dough an ingredient is dropped

BLX_ALPHA = 0.25                 # blend crossover: how far outside the parents' range
MUTATION_WEIGHTS = {"amount": 0.50, "substitute": 0.25, "add": 0.15, "remove": 0.10}
AMOUNT_SIGMA = 0.15              # amount mutation: amount *= exp(N(0, sigma))

# How likely each role is picked when the "add" mutation adds an ingredient
ROLE_ADD_WEIGHT = {"flavor": 1.0, "add_in": 1.0, "salt": 0.5, "texturizer": 0.4,
                   "liquid": 0.3, "decoration": 0.3, "binder": 0.3, "sugar": 0.3,
                   "leavener": 0.3, "fat": 0.2, "structure": 0.2}

MUTATION_LOG = Counter()         # counts which mutation types were applied


# ============================================================================
# INGREDIENT DATABASE
# ============================================================================

class IngredientDB:
    """Roles, popularity and typical amounts per ingredient (+ optional similarity)."""

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.names = [r["ingredient"].strip().lower() for r in rows]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.role, self.popularity, raw_median = {}, {}, {}
        for r, name in zip(rows, self.names):
            role = r["role"].strip()
            if role not in ROLES:
                raise ValueError(f"Unknown role '{role}' for ingredient '{name}'")
            self.role[name] = role
            self.popularity[name] = float(r["pct_of_recipes"]) / 100
            v = (r.get("median_pct_of_recipe") or "").strip()
            raw_median[name] = float(v) / 100 if v else None

        role_medians = defaultdict(list)
        for n, m in raw_median.items():
            if m is not None:
                role_medians[self.role[n]].append(m)
        self.median = {n: m if m is not None else float(np.median(role_medians[self.role[n]]))
                       for n, m in raw_median.items()}

        self.by_role = defaultdict(list)
        for n in self.names:
            self.by_role[self.role[n]].append(n)

        self.similarity = None   # (n x n) cosine similarity, built from recipes

    def build_similarity(self, recipes):
        """Co-occurrence based ingredient similarity: ingredients used in similar
        company (e.g. pecans / walnuts) get a high score."""
        n = len(self.names)
        C = np.zeros((n, n))
        for rec in recipes.values():
            idx = [self.index[i] for i in rec]
            for a in idx:
                C[a, idx] += 1
        np.fill_diagonal(C, 0)
        norms = np.linalg.norm(C, axis=1, keepdims=True)
        norms[norms == 0] = 1
        V = C / norms
        self.similarity = V @ V.T

    def sim(self, a, b):
        if self.similarity is None:
            return 0.0
        return float(self.similarity[self.index[a], self.index[b]])

    def vectorize(self, recipe):
        v = np.zeros(len(self.names))
        for ing, p in recipe.items():
            v[self.index[ing]] = p
        return v


# ============================================================================
# RECIPE DATABASE (existing recipes + like/view ratings)
# ============================================================================

class RecipeDB:
    """Loads existing recipes (long format) and optional ratings.

    Ratings are turned into a 0-1 score:
      1. Bayesian smoothing of likes/views, so a recipe with 3 views and 2 likes
         does not beat a recipe with 50k views and 8k likes:
             smoothed = (likes + m * global_ratio) / (views + m),  m = median views
      2. Percentile rank of the smoothed ratio (robust to outliers).
    """

    def __init__(self, recipes_csv, ingredient_db, ratings_csv=None):
        self.db = ingredient_db
        grams = defaultdict(lambda: defaultdict(float))
        unknown = Counter()
        with open(recipes_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ing = r["ingredient"].strip().lower()
                g = (r.get("grams") or "").strip()
                if not g:
                    continue
                if ing not in ingredient_db.index:
                    unknown[ing] += 1
                    continue
                grams[r["recipe_id"].strip()][ing] += float(g)

        self.recipes = {}
        for rid, ings in grams.items():
            total = sum(ings.values())
            if total > 0:
                self.recipes[rid] = {i: g / total for i, g in ings.items() if g > 0}
        if unknown:
            print(f"[RecipeDB] skipped {sum(unknown.values())} rows with unknown ingredients, "
                  f"e.g. {', '.join(n for n, _ in unknown.most_common(5))}")
        print(f"[RecipeDB] loaded {len(self.recipes)} recipes")

        self.ids = list(self.recipes)
        self.matrix = np.array([ingredient_db.vectorize(self.recipes[i]) for i in self.ids])

        self.rated_ids, self.rated_matrix, self.rated_scores = [], None, None
        if ratings_csv:
            self._load_ratings(ratings_csv)

    def _load_ratings(self, path):
        likes, views = {}, {}
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rid = r["recipe_id"].strip()
                v = float(r["views"] or 0)
                if rid in self.recipes and v > 0:
                    likes[rid], views[rid] = float(r["likes"] or 0), v
        if not views:
            print("[RecipeDB] no usable ratings found")
            return
        global_ratio = sum(likes.values()) / sum(views.values())
        m = float(np.median(list(views.values())))
        smoothed = {rid: (likes[rid] + m * global_ratio) / (views[rid] + m) for rid in views}

        self.rated_ids = list(smoothed)
        values = np.array([smoothed[r] for r in self.rated_ids])
        ranks = values.argsort().argsort()
        self.rated_scores = ranks / max(len(values) - 1, 1)
        self.rated_matrix = np.array([self.db.vectorize(self.recipes[r]) for r in self.rated_ids])
        print(f"[RecipeDB] {len(self.rated_ids)} recipes with ratings "
              f"(global like/view ratio {global_ratio:.4f})")

    def nearest_distance(self, recipe):
        v = self.db.vectorize(recipe)
        return float(np.abs(self.matrix - v).sum(axis=1).min())   # L1, range 0..2


# ============================================================================
# FITNESS MODELS  (swap in your team's function here)
# ============================================================================

def role_totals(recipe, db):
    t = defaultdict(float)
    for ing, p in recipe.items():
        t[db.role[ing]] += p
    return t


def gauss(x, target, sigma):
    return math.exp(-0.5 * ((x - target) / sigma) ** 2)


class FitnessModel:
    """Interface: evaluate(recipe) -> float, higher is better (ideally 0..1)."""
    name = "base"

    def evaluate(self, recipe):
        raise NotImplementedError


class MockFitness(FitnessModel):
    """Stand-in plausibility score from role ratios (targets from ingredient medians).
    Replace or combine with the real fitness function."""
    name = "mock"
    # role: (target share of dough, tolerance, weight)
    TARGETS = {"structure": (0.28, 0.07, 2), "fat": (0.19, 0.06, 2),
               "sugar": (0.26, 0.07, 2), "binder": (0.08, 0.04, 1),
               "leavener": (0.006, 0.005, 1), "salt": (0.004, 0.003, 1),
               "add_in": (0.15, 0.12, 1)}

    def __init__(self, db):
        self.db = db

    def evaluate(self, recipe):
        t = role_totals(recipe, self.db)
        total_w = sum(w for _, _, w in self.TARGETS.values())
        score = sum(w * gauss(t[r], tg, s) for r, (tg, s, w) in self.TARGETS.items()) / total_w

        if t["liquid"] > 0.08:                                   # too wet = batter
            score *= gauss(t["liquid"], 0.08, 0.04)
        n_flavor = sum(1 for i in recipe if self.db.role[i] == "flavor")
        if n_flavor == 0:
            score *= 0.8
        elif n_flavor > 3:
            score *= 0.85 ** (n_flavor - 3)
        if len(recipe) > 12:
            score *= 0.9 ** (len(recipe) - 12)
        return score


class RatingKNNFitness(FitnessModel):
    """Predicts the (0-1) like/view score of a new recipe as the distance-weighted
    mean of its k nearest rated recipes."""
    name = "ratings_knn"

    def __init__(self, recipe_db, k=15):
        if recipe_db.rated_matrix is None:
            raise ValueError("RatingKNNFitness needs a ratings file")
        self.rdb, self.k = recipe_db, min(k, len(recipe_db.rated_ids))

    def evaluate(self, recipe):
        v = self.rdb.db.vectorize(recipe)
        d = np.abs(self.rdb.rated_matrix - v).sum(axis=1)
        idx = np.argpartition(d, self.k - 1)[:self.k]
        w = 1.0 / (d[idx] + 0.05)
        return float((w * self.rdb.rated_scores[idx]).sum() / w.sum())


class NoveltyFitness(FitnessModel):
    """Rewards a *moderate* distance to the nearest known recipe (hump-shaped):
    copies score low, and so do recipes that are unrecognisable."""
    name = "novelty"

    def __init__(self, recipe_db, sweet_spot=0.5, width=0.25):
        self.rdb, self.d0, self.w = recipe_db, sweet_spot, width

    def evaluate(self, recipe):
        return gauss(self.rdb.nearest_distance(recipe), self.d0, self.w)


class CombinedFitness(FitnessModel):
    name = "combined"

    def __init__(self, components):            # list of (FitnessModel, weight)
        self.components = components
        self.total = sum(w for _, w in components)

    def evaluate(self, recipe):
        return sum(m.evaluate(recipe) * w for m, w in self.components) / self.total


def build_fitness(mode, db, recipe_db):
    """Central place to choose the fitness function."""
    if mode == "auto":
        mode = "ratings" if recipe_db is not None and recipe_db.rated_matrix is not None else "mock"
    if mode == "mock":
        return MockFitness(db)
    if mode == "ratings":
        # Placeholder weights - to be replaced by your teammate's fitness design.
        return CombinedFitness([(RatingKNNFitness(recipe_db), 0.6),
                                (MockFitness(db), 0.3),
                                (NoveltyFitness(recipe_db), 0.1)])
    raise ValueError(f"unknown fitness mode {mode}")


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


def weighted_pick(candidates, weights):
    return random.choices(candidates, weights=weights, k=1)[0]


def pick_popular(db, role, exclude=()):
    cands = [n for n in db.by_role[role] if n not in exclude]
    return weighted_pick(cands, [db.popularity[n] + 0.002 for n in cands])


def is_removable(recipe, ing, db):
    role = db.role[ing]
    return role not in ESSENTIAL_ROLES or sum(1 for i in recipe if db.role[i] == role) > 1


def repair(recipe, db):
    """Make any recipe valid again. Modifies in place."""
    for k in [k for k, v in recipe.items() if v <= 0]:
        del recipe[k]
    normalize(recipe)
    for k in [k for k, v in recipe.items() if v < MIN_PROPORTION]:
        del recipe[k]

    for role, group in role_groups(recipe, db).items():           # caps per role
        excess = len(group) - MAX_PER_ROLE.get(role, 3)
        for ing in sorted(group, key=group.get)[:max(excess, 0)]:
            del recipe[ing]

    for role in ESSENTIAL_ROLES:                                  # essentials present
        if not any(db.role[i] == role for i in recipe):
            ing = pick_popular(db, role, exclude=recipe)
            recipe[ing] = db.median[ing]

    while len(recipe) > MAX_INGREDIENTS:                          # total cap
        removable = [i for i in recipe if is_removable(recipe, i, db)]
        del recipe[min(removable, key=recipe.get)]
    normalize(recipe)
    return recipe


def random_recipe(db):
    """A plausible random recipe, built role by role from typical amounts."""
    plan = {"structure": (1, 1.0), "fat": (1, 1.0), "sugar": (2, 0.9), "binder": (1, 0.85),
            "leavener": (1, 0.8), "salt": (1, 0.8), "flavor": (2, 0.9), "add_in": (2, 0.7),
            "liquid": (1, 0.15), "texturizer": (1, 0.15), "decoration": (1, 0.1)}
    rec = {}
    for role, (max_n, prob) in plan.items():
        for _ in range(random.randint(1, max_n)):
            if random.random() < prob:
                ing = pick_popular(db, role, exclude=rec)
                rec[ing] = db.median[ing] * math.exp(random.gauss(0, 0.3))
    return repair(rec, db)


# ============================================================================
# VARIATION OPERATORS
# ============================================================================

def blx(a, b, alpha=BLX_ALPHA):
    lo, hi = min(a, b), max(a, b)
    d = hi - lo
    return max(random.uniform(lo - alpha * d, hi + alpha * d), 1e-6)


def cx_role_blend(ind1, ind2, db):
    """Role-based crossover: per role, child 1 inherits the whole ingredient group
    of one parent and child 2 the group of the other. Ingredients present in BOTH
    parents' group get a blended amount (BLX-alpha)."""
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


def _mut_amount(ind, db):
    ing = random.choice(list(ind))
    ind[ing] *= math.exp(random.gauss(0, AMOUNT_SIGMA))
    return True


def _mut_substitute(ind, db):
    old = random.choice(list(ind))
    role = db.role[old]
    cands = [n for n in db.by_role[role] if n not in ind]
    if not cands:
        return False
    if db.similarity is not None:   # similar ingredients first, popularity as tie-breaker
        w = [(max(db.sim(old, c), 0) + 0.05) * math.sqrt(db.popularity[c] + 0.002) for c in cands]
    else:
        w = [math.sqrt(db.popularity[c] + 0.002) for c in cands]
    new = weighted_pick(cands, w)
    ind[new] = ind.pop(old) * db.median[new] / db.median[old]   # rescale to typical amount
    return True


def _mut_add(ind, db):
    if len(ind) >= MAX_INGREDIENTS:
        return False
    counts = Counter(db.role[i] for i in ind)
    cands = [n for n in db.names
             if n not in ind and counts[db.role[n]] < MAX_PER_ROLE.get(db.role[n], 3)]
    if not cands:
        return False
    w = []
    for c in cands:
        fit = (np.mean([max(db.sim(c, i), 0) for i in ind]) + 0.05) if db.similarity is not None else 1.0
        w.append(ROLE_ADD_WEIGHT.get(db.role[c], 0.3) * fit * math.sqrt(db.popularity[c] + 0.002))
    new = weighted_pick(cands, w)
    ind[new] = db.median[new] * math.exp(random.gauss(0, 0.3))
    return True


def _mut_remove(ind, db):
    removable = [i for i in ind if is_removable(ind, i, db)]
    if not removable:
        return False
    del ind[random.choice(removable)]
    return True


MUTATIONS = {"amount": _mut_amount, "substitute": _mut_substitute,
             "add": _mut_add, "remove": _mut_remove}


def mutate(ind, db):
    op = random.choices(list(MUTATION_WEIGHTS), weights=list(MUTATION_WEIGHTS.values()))[0]
    if not MUTATIONS[op](ind, db):          # operator not possible -> fall back
        op = "amount"
        _mut_amount(ind, db)
    MUTATION_LOG[op] += 1
    repair(ind, db)
    return (ind,)


def recipe_distance(a, b):
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))   # L1, 0..2


# ============================================================================
# EVOLUTION (island model)
# ============================================================================

def setup_deap(db, recipe_db, fitness, args):
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", dict, fitness=creator.FitnessMax)

    seeds = list(recipe_db.recipes.values()) if recipe_db else []

    def init_individual():
        if seeds and random.random() < args.seed_fraction:
            ind = creator.Individual(random.choice(seeds))
        else:
            ind = creator.Individual(random_recipe(db))
        repair(ind, db)
        return ind

    tb = base.Toolbox()
    tb.register("individual", init_individual)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", lambda ind: (fitness.evaluate(ind),))
    tb.register("mate", cx_role_blend, db=db)
    tb.register("mutate", mutate, db=db)
    tb.register("select", tools.selTournament, tournsize=args.tournament)
    return tb


def evaluate_invalid(pop, tb):
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(tb.evaluate, invalid)):
        ind.fitness.values = fit
    return len(invalid)


def evolve(tb, args):
    islands = [tb.population(n=args.pop) for _ in range(args.islands)]
    for pop in islands:
        evaluate_invalid(pop, tb)
    hofs = [tools.HallOfFame(30) for _ in islands]
    history = []

    for gen in range(1, args.generations + 1):
        for i, pop in enumerate(islands):
            elites = [tb.clone(e) for e in tools.selBest(pop, args.elite)]
            offspring = [tb.clone(o) for o in tb.select(pop, len(pop) - args.elite)]
            offspring = algorithms.varAnd(offspring, tb, args.cxpb, args.mutpb)
            evaluate_invalid(offspring, tb)
            pop[:] = elites + offspring
            hofs[i].update(pop)

            fits = [ind.fitness.values[0] for ind in pop]
            history.append({"generation": gen, "island": i + 1, "best": max(fits),
                            "mean": float(np.mean(fits)),
                            "mean_ingredients": float(np.mean([len(p) for p in pop]))})

        if args.islands > 1 and gen % args.migration_every == 0:
            tools.migRing(islands, args.migrants, tools.selBest)
            for pop in islands:                      # migrants become independent copies
                pop[:] = [tb.clone(ind) for ind in pop]

        if gen == 1 or gen % args.log_every == 0 or gen == args.generations:
            row = [h for h in history if h["generation"] == gen]
            best = "  ".join(f"I{h['island']} {h['best']:.3f}/{h['mean']:.3f}" for h in row)
            print(f"gen {gen:4d} | best/mean per island: {best}")

    return islands, hofs, history


def pick_book(islands, hofs, n_recipes, min_distance):
    """Greedy diverse selection: pool every island's hall of fame + final population,
    sort by fitness, and take the best recipe that is at least `min_distance` away
    from all recipes already in the book. If too few candidates qualify, the
    distance requirement is relaxed step by step."""
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


# ============================================================================
# OUTPUT
# ============================================================================

def recipe_name(recipe, db):
    for role in ("add_in", "flavor"):
        items = [i for i in recipe if db.role[i] == role]
        if items:
            return f"{max(items, key=recipe.get).title()} cookie"
    return "Plain cookie"


def fmt_grams(g):
    return f"{g:.0f}" if g >= 10 else f"{g:.1f}"


def format_recipe(recipe, db, batch):
    lines = []
    for role in ROLES:
        items = sorted(((i, p) for i, p in recipe.items() if db.role[i] == role), key=lambda x: -x[1])
        for ing, p in items:
            lines.append(f"    {fmt_grams(p * batch):>6} g   {ing:<26} [{role}]")
    return "\n".join(lines)


def save_outputs(book, history, db, args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = []
    print("\n" + "=" * 64 + f"\nRECIPE BOOK  (batch of {args.batch_grams} g dough)\n" + "=" * 64)
    for n, rec in enumerate(book, 1):
        name = recipe_name(rec, db)
        print(f"\n{n}. {name}   fitness {rec.fitness.values[0]:.3f}")
        print(format_recipe(rec, db, args.batch_grams))
        data.append({"name": name, "fitness": rec.fitness.values[0],
                     "proportions": dict(sorted(rec.items(), key=lambda x: -x[1])),
                     "grams": {i: round(p * args.batch_grams, 1) for i, p in rec.items()}})
    with open(out / "recipe_book.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    with open(out / "fitness_history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    total = sum(MUTATION_LOG.values()) or 1
    print("\nMutations applied: " + ", ".join(f"{k} {v / total:.0%}" for k, v in MUTATION_LOG.most_common()))
    print(f"Saved {out / 'recipe_book.json'} and {out / 'fitness_history.csv'}")


# ============================================================================
# DEMO DATA (tests the recipe/ratings pipeline before the real files arrive)
# ============================================================================

def make_demo_data(db, n=300):
    mock = MockFitness(db)
    with open("demo_recipes.csv", "w", newline="", encoding="utf-8") as fr, \
         open("demo_ratings.csv", "w", newline="", encoding="utf-8") as ft:
        wr, wt = csv.writer(fr), csv.writer(ft)
        wr.writerow(["recipe_id", "ingredient", "grams"])
        wt.writerow(["recipe_id", "likes", "views"])
        for k in range(n):
            rid = f"demo{k:04d}"
            rec = random_recipe(db)
            for ing, p in rec.items():
                wr.writerow([rid, ing, round(p * 800, 1)])
            views = int(math.exp(random.uniform(6, 13)))
            ratio = 0.02 + 0.10 * mock.evaluate(rec) * random.uniform(0.6, 1.4)
            wt.writerow([rid, int(views * ratio), views])
    print(f"Wrote demo_recipes.csv and demo_ratings.csv ({n} fake recipes)")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evolve a cookie recipe book with DEAP.")
    p.add_argument("--ingredients", default="ingredients.csv")
    p.add_argument("--recipes", help="existing recipes CSV: recipe_id,ingredient,grams")
    p.add_argument("--ratings", help="ratings CSV: recipe_id,likes,views")
    p.add_argument("--fitness", default="auto", choices=["auto", "mock", "ratings"])
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--pop", type=int, default=80, help="population size per island")
    p.add_argument("--islands", type=int, default=3, help="number of sub-populations")
    p.add_argument("--recipes-in-book", type=int, default=3)
    p.add_argument("--cxpb", type=float, default=0.7)
    p.add_argument("--mutpb", type=float, default=0.3)
    p.add_argument("--tournament", type=int, default=3)
    p.add_argument("--elite", type=int, default=2)
    p.add_argument("--migration-every", type=int, default=40)
    p.add_argument("--migrants", type=int, default=1)
    p.add_argument("--seed-fraction", type=float, default=0.5,
                   help="share of the initial population taken from existing recipes")
    p.add_argument("--min-distance", type=float, default=0.4,
                   help="minimum L1 distance between recipes in the book (0..2)")
    p.add_argument("--batch-grams", type=float, default=800)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--make-demo-data", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    db = IngredientDB(args.ingredients)
    print(f"[IngredientDB] {len(db.names)} ingredients in {len(db.by_role)} roles")
    if args.make_demo_data:
        make_demo_data(db)
        return

    recipe_db = None
    if args.recipes:
        recipe_db = RecipeDB(args.recipes, db, args.ratings)
        db.build_similarity(recipe_db.recipes)

    fitness = build_fitness(args.fitness, db, recipe_db)
    print(f"[Fitness] using '{fitness.name}'")

    tb = setup_deap(db, recipe_db, fitness, args)
    islands, hofs, history = evolve(tb, args)
    book = pick_book(islands, hofs, args.recipes_in_book, args.min_distance)
    save_outputs(book, history, db, args)


if __name__ == "__main__":
    main()
