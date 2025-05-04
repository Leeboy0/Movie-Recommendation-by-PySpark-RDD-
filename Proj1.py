#!/usr/bin/env python3
import os, sys, math
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import pandas as pd

# ── 0) Force Spark to use this python for driver & workers ────────────────
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYSPARK_PYTHON"]        = sys.executable

# ── 1) Start Spark ────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("MovieLensRecommender")
    .config("spark.pyspark.driver.python", sys.executable)
    .config("spark.pyspark.python",        sys.executable)
    .getOrCreate()
)
sc = spark.sparkContext

# ── 2) Resolve script‑relative paths ────────────────────────────────────────
base      = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base, "u.data").replace("\\","/")
item_path = os.path.join(base, "u.item").replace("\\","/")

# ── 3) Load & parse ratings ────────────────────────────────────────────────
def parse_rating(line):
    u, m, r, _ = line.split("\t")
    return int(u), int(m), float(r)

ratings = (
    sc.textFile(f"file:///{data_path}")
      .map(parse_rating)
      .cache()
)

# ── 4) Build and broadcast user → [(movie, rating)] dict ───────────────────
user_ratings_rdd = (
    ratings
    .map(lambda x: (x[0], (x[1], x[2])))
    .groupByKey()
    .mapValues(list)
)
user_ratings_dict = user_ratings_rdd.collectAsMap()
b_user_ratings    = sc.broadcast(user_ratings_dict)

# ── 5) Load & broadcast movie titles ────────────────────────────────────────
movies = (
    sc.textFile(f"file:///{item_path}")
      .map(lambda line: line.split("|"))
      .map(lambda p: (int(p[0]), p[1]))
      .collectAsMap()
)
b_movies = sc.broadcast(movies)

# ── 6) Cosine similarity ────────────────────────────────────────────────────
def cosine(r1, r2):
    d1, d2 = dict(r1), dict(r2)
    common = set(d1) & set(d2)
    if not common: return 0.0
    num = sum(d1[m]*d2[m] for m in common)
    den = math.sqrt(sum(d1[m]**2 for m in common)) * math.sqrt(sum(d2[m]**2 for m in common))
    return num/den if den else 0.0

# ── 7) Build all user‑pairs & compute similarity via broadcast dict ────────
users = list(user_ratings_dict.keys())
pairs = [(u1,u2) for i,u1 in enumerate(users) for u2 in users[i+1:]]
user_pairs = sc.parallelize(pairs).map(
    lambda pair: (
       pair,
       cosine(b_user_ratings.value[pair[0]], b_user_ratings.value[pair[1]])
    )
)
sims = user_pairs.flatMap(lambda x: [
    (x[0][0], (x[0][1], x[1])),
    (x[0][1], (x[0][0], x[1]))
]).cache()

# ── 8) Plot similarity distribution ─────────────────────────────────────────
sim_scores = sims.map(lambda x: x[1][1]).collect()
plt.figure(figsize=(6,4))
plt.hist(sim_scores, bins=50)
plt.title("User–User Cosine Similarity")
plt.xlabel("Similarity"); plt.ylabel("Count")
plt.tight_layout(); plt.show()

# ── 9) Generate top‑10 recommendations for user=1 ───────────────────────────
target_user = 1
N = 10
top_neighbors = (
    sims.filter(lambda x: x[0]==target_user)
        .map(lambda x: x[1])     # (neighbor_id, sim)
        .sortBy(lambda x: -x[1])
        .take(N)
)
sim_dict    = dict(top_neighbors)
seen_movies = {m for m,_ in b_user_ratings.value[target_user]}

# gather neighbor ratings for unseen movies
neighbor_ratings = []
for uid, movelist in b_user_ratings.value.items():
    if uid in sim_dict:
        for (mov, r) in movelist:
            if mov not in seen_movies:
                neighbor_ratings.append((uid, mov, r))

# compute weighted scores
movie_scores = {}
for uid, mov, r in neighbor_ratings:
    w = sim_dict[uid]
    score, weight = movie_scores.get(mov, (0,0))
    movie_scores[mov] = (score + w*r, weight + abs(w))

# finalize predictions
preds = [(b_movies.value[m], score/weight if weight else 0.0)
         for m,(score,weight) in movie_scores.items()]
top_recs = sorted(preds, key=lambda x: -x[1])[:10]

# ── 10) Plot recommendations ────────────────────────────────────────────────
df = pd.DataFrame(top_recs, columns=["title","predicted_score"])
ax = df.plot.barh(x="title", y="predicted_score", legend=False, figsize=(6,5))
ax.invert_yaxis()
plt.xlabel("Predicted Rating"); plt.title(f"Top 10 for User {target_user}")
plt.tight_layout(); plt.show()

# ── 11) Plot overall rating distribution ────────────────────────────────────
all_r = ratings.map(lambda x: x[2]).collect()
plt.figure(figsize=(5,3))
plt.hist(all_r, bins=5, edgecolor='k')
plt.title("Overall Rating Distribution")
plt.xlabel("Rating"); plt.ylabel("Count")
plt.tight_layout(); plt.show()

spark.stop()
