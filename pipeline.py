import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import json

N = 30  # size of each candidate list
K = 5   # size of the final list


#---------------------------------------------------------------------------------
#step1: loading the data


pois=pd.read_csv("data/pois.csv", index_col="poi_id")
trips=pd.read_csv("data/trips.csv", index_col="trip_id", parse_dates=["start_date"])
interactions=pd.read_csv("data/interactions.csv")


def to_set(text):
    return set(text.split())


def to_int_set(text):
    result = set()
    for word in str(text).split():  # str() in case pandas read a single number
        result.add(int(word))
    return result


def action_to_label(action):
    if action in ["save", "visit", "booking"]:
        return 1
    return 0


pois["tag_set"] = pois["tags"].apply(to_set)
# print(pois)
pois["open_set"] = pois["open_days"].apply(to_int_set)
# print(pois)
trips["interest_set"] = trips["interests"].apply(to_set)

# print(interactions)
interactions["liked"] = interactions["action"].apply(action_to_label)
# print(interactions)
labels = interactions[["trip_id", "poi_id", "liked"]] #This will be the sub domain for our predictor

# print(pois, trips, interactions)




#---------------------------------------------------------------------------
#step2:calculating weekdays from start date and number of days
def trip_weekdays(trip_id):
    start = trips.loc[trip_id, "start_date"]
    days = trips.loc[trip_id, "trip_days"]
    weekdays = set()
    for i in range(days):
        day = start + pd.Timedelta(days=i)
        weekdays.add(day.weekday())
    return weekdays

#note: {0: monday, 1: tuesday, ... , 6: sunday}
# print(trip_weekdays(0))


# the POIs that are open at least one day of q trip
def open_pois(trip_id):
    weekdays = trip_weekdays(trip_id)
    keep=[]
    for poi_id in pois.index:
        if len(pois.loc[poi_id, "open_set"] & weekdays) > 0:
            keep.append(poi_id)
    return pois.loc[keep]

# print(pois.loc[1, "open_set"])
# print(pois.index)
# print(open_pois(1))




#------------------------------------------------------------
#step3: features for each (trip, POI) pair
def jaccard(set_a, set_b):
    union = set_a | set_b
    if len(union) == 0:
        return 0.0
    return len(set_a & set_b) / len(union)


def history_tags(trip_id):
    traveler = trips.loc[trip_id, "traveler_id"]
    start = trips.loc[trip_id, "start_date"]
    earlier_trips = trips[(trips["traveler_id"] == traveler) & (trips["start_date"] < start)]
    tags = set()
    for other_trip in earlier_trips.index:
        liked_rows = labels[(labels["trip_id"] == other_trip) & (labels["liked"] == 1)]
        for poi_id in liked_rows["poi_id"]:
            tags = tags | pois.loc[poi_id, "tag_set"]
    return tags


# print(history_tags(1))

def make_features(trip_id, poi_ids):
    interests = trips.loc[trip_id, "interest_set"]
    budget = trips.loc[trip_id, "budget"]
    history = history_tags(trip_id)
    rows = []
    for poi_id in poi_ids:
        tags = pois.loc[poi_id, "tag_set"]
        over_budget = pois.loc[poi_id, "price"] - budget
        if over_budget < 0:
            over_budget = 0
        rows.append({"poi_id": poi_id,
                     "interest_match": jaccard(interests, tags),
                     "history_match": jaccard(history, tags),
                     "popularity": pois.loc[poi_id, "popularity"],
                     "over_budget": over_budget})
    features = pd.DataFrame(rows)
    return features.set_index("poi_id")



# print(make_features(0,[1,2,3,4,5,6,7,8,9,10]))






# # ---------- Step 4: training rows and the date split ----------
def shown_rows(trip_id):
    open_ids = open_pois(trip_id).index
    shown = labels[labels["trip_id"] == trip_id]
    poi_ids = []
    y_values = []
    for poi_id, liked in zip(shown["poi_id"], shown["liked"]):
        if poi_id in open_ids:
            poi_ids.append(poi_id)
            y_values.append(liked)
    rows = make_features(trip_id, poi_ids)
    rows["y"] = y_values
    rows["trip_id"] = trip_id
    return rows



# print(shown_rows(0))

def collect_rows(trip_ids):
    tables = []
    for trip_id in trip_ids:
        tables.append(shown_rows(trip_id))
    return pd.concat(tables)


order = trips.sort_values("start_date").index
cut = int(0.7 * len(order))  # earlier 70% of trips train the model, later 30% test it
train = collect_rows(order[:cut])
test = collect_rows(order[cut:])





# -------------------------------------------------------------
#step5: train the model


FEATURES = ["interest_match", "history_match", "popularity"]  # over_budget is used later, in compatibility

model = LogisticRegression()
model.fit(train[FEATURES], train["y"])




#------------------------------------------
# step6: final score = preference x compatibility

def final_score(trip_id, features):
    mobility = trips.loc[trip_id, "mobility"]
    party = trips.loc[trip_id, "party"]
    preferences = model.predict_proba(features[FEATURES])[:, 1]

    rows = []
    for poi_id, preference in zip(features.index, preferences):
        compatibility = 1 - features.loc[poi_id, "over_budget"] / 2
        if mobility != "car" and pois.loc[poi_id, "remote"] == 1:
            compatibility = compatibility * 0.5
        if party == "family" and "nightlife" in pois.loc[poi_id, "tag_set"]:
            compatibility = compatibility * 0.3
        rows.append({"poi_id": poi_id,
                     "preference": preference,
                     "compatibility": compatibility,
                     "score": preference * compatibility})
    return pd.DataFrame(rows).set_index("poi_id")






# -------------------------------------------------------------
#step7: candidates and the top-5 list

def get_candidates(trip_id):
    features = make_features(trip_id, open_pois(trip_id).index)
    by_match = features.sort_values(["interest_match", "history_match"], ascending=False).head(N)
    by_popularity = features.sort_values("popularity", ascending=False).head(N)
    ids = list(by_match.index)
    for poi_id in by_popularity.index:
        if poi_id not in ids:  # a POI can be in both lists
            ids.append(poi_id)
    return features.loc[ids]


def recommend(trip_id):
    candidates = get_candidates(trip_id)
    table = candidates.join(final_score(trip_id, candidates))
    top = table.sort_values("score", ascending=False).head(K)
    total = top["score"].sum()
    has_history = table["history_match"].max() > 0

    results = []
    for rank, poi_id in enumerate(top.index, start=1):
        row = top.loc[poi_id]
        confidence = 0.4 + 0.3 * row["popularity"]
        if has_history:
            confidence = confidence + 0.3
        results.append({"poi_id": int(poi_id),
                        "rank": rank,
                        "score": round(float(row["score"]), 3),
                        "preference": round(float(row["preference"]), 3),
                        "context_compatibility": round(float(row["compatibility"]), 3),
                        "weight": round(float(row["score"] / total), 3),
                        "confidence": round(float(confidence), 3)})
    return results





#------------------------------------------
# step8: evaluation metrics
def metrics(y_ranked):
    top = y_ranked[:K]
    hits = sum(top)
    precision = hits / K
    recall = hits / sum(y_ranked)
    return precision, recall


def evaluate(score_function):
    # rank each test trip's shown POIs, then average the two metrics over the trips
    results = []
    for trip_id in test["trip_id"].unique():
        rows = test[test["trip_id"] == trip_id].copy()
        if rows["y"].sum() == 0:
            continue  # nothing was liked on this trip, so there is nothing to find
        rows["s"] = score_function(trip_id, rows)
        rows = rows.sort_values("s", ascending=False)
        results.append(metrics(rows["y"].tolist()))
    return np.mean(results, axis=0).round(3)


def model_scores(trip_id, rows):
    return final_score(trip_id, rows)["score"].values


def popularity_scores(trip_id, rows):
    return rows["popularity"].values


# =====================================================================
# RESULTS
# =====================================================================
print("=" * 70)
print("1. OUTPUT FORMAT: top 5 POIs for trip 171, as JSON")
result = recommend(171)
print(json.dumps(result, indent=2))
print("number of POIs:", len(result), "| weights add up to:", round(sum(item["weight"] for item in result), 3))

print()
print("=" * 70)
print("2. EVALUATION on the test trips (shown POIs only)")
print("model      [precision@5, recall@5]:", evaluate(model_scores))
print("popularity [precision@5, recall@5]:", evaluate(popularity_scores))

print()
print("=" * 70)
print("3. THREE EXAMPLE TRIPS")
for trip_id in [2, 195, 196]:
    print()
    print("Trip", trip_id, "| interests:", trips.loc[trip_id, "interests"], "| budget:", trips.loc[trip_id, "budget"],
          "|", trips.loc[trip_id, "mobility"], "|", trips.loc[trip_id, "party"])
    for item in recommend(trip_id):
        poi_id = item["poi_id"]
        print(item["rank"], "POI", poi_id, "|", pois.loc[poi_id, "tags"], "| price", pois.loc[poi_id, "price"],
              "| score", item["score"], "| compatibility", item["context_compatibility"])