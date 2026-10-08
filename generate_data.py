import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)  # fixed seed: same data every run
TOPICS = ["food", "history", "architecture", "museum", "art", "nature", "shopping", "nightlife", "activities", "kids"]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# POIs: 1-2 topic tags + optional style tag ("local-gem" / "touristy"); popularity depends on style
pois = []
for i in range(200):
    style = rng.choice(["local-gem", "touristy", ""], p=[0.35, 0.3, 0.35])
    tags = list(rng.choice(TOPICS, rng.integers(1, 3), replace=False)) + ([style] if style else [])
    pop = rng.beta(6, 2) if style == "touristy" else rng.beta(2, 6) if style == "local-gem" else rng.beta(2, 2)
    days = sorted(rng.choice(7, rng.integers(3, 8), replace=False))  # open weekdays, 0 = Monday
    pois.append(dict(poi_id=i, tags=" ".join(tags), price=int(rng.integers(1, 4)),
                     popularity=round(pop, 3), open_days=" ".join(map(str, days)),
                     remote=int(rng.random() < 0.25)))  # remote = hard to reach without a car
pois = pd.DataFrame(pois)

# Travelers: hidden taste (4 words), 2 trips each, 20 shown POIs per trip
trips, interactions = [], []
for u in range(100):
    avoids = rng.random() < 0.4  # hidden: avoids touristy places
    taste = list(rng.choice(TOPICS, 3 if avoids else 4, replace=False)) + (["local-gem"] if avoids else [])
    for k, day in enumerate(sorted(rng.choice(365, 2, replace=False))):
        trip_id, budget, trip_days = 2 * u + k, int(rng.integers(1, 4)), int(rng.integers(2, 6))
        start = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(day))
        mobility, party = rng.choice(["walk", "transit", "car"]), rng.choice(["solo", "couple", "family"])
        # stated interests: only 2 of the 4 taste words (avoiders always state local-gem)
        interests = ["local-gem", rng.choice(taste[:3])] if avoids else list(rng.choice(taste, 2, replace=False))
        trips.append(dict(trip_id=trip_id, traveler_id=u, interests=" ".join(interests),
                          budget=budget, trip_days=trip_days, start_date=start.date(), mobility=mobility, party=party))
        for p in rng.choice(200, 20, replace=False):
            tags, price, pop = pois.tags[p].split(), pois.price[p], pois.popularity[p]
            z = (-3.5 + 2 * len(set(taste) & set(tags)) - 2.5 * avoids * ("touristy" in tags)
                 + 2.0 * (1 - avoids) * pop - 1.0 * max(0, price - budget)
                 - 1.5 * pois.remote[p] * (mobility != "car") - 2 * (party == "family") * ("nightlife" in tags))  # hidden truth
            liked = rng.random() < sigmoid(z)
            action = rng.choice(["save", "visit", "booking"] if liked else ["view", "click", "dismiss"])
            stamp = start + pd.Timedelta(days=int(rng.integers(0, trip_days)))
            interactions.append(dict(trip_id=trip_id, poi_id=int(p), action=action, timestamp=stamp.date()))

os.makedirs("data", exist_ok=True)  # create the data folder if missing
pois.to_csv("data/pois.csv", index=False)
pd.DataFrame(trips).to_csv("data/trips.csv", index=False)
pd.DataFrame(interactions).to_csv("data/interactions.csv", index=False)
print(len(pois), "POIs,", len(trips), "trips,", len(interactions), "interactions")