A) The problem:  Construct a function that takes in a tuple of (traveller/ trip, POI) and gives out a number in [0,1] which suggests how compatible the POI is.

We have three tables of data with us.
POIs table - The columns are POI ids, tags, price, popularity, open days, remote
Trips table - The columns are trip ids, traveller ids, interests, budget, trip days, start date, mobility, party
Interactions table - The columns are - trip ids, POI ids, action, timestamp

Where,
tags is a subset of {activities, architecture, art, food, history, kids, local-gem, museum, nature, nightlife, shopping, touristy}
price ={1,2,3},
popularity in [0,1],
	open days ={o,1,2,3,4,5,6},
	remote ={0,1}
	traveller id ={0,1,...,99},
	budget ={1,2,3},
	trip days ={2,3,4,5},
	start date is a date in 2025,
	mobility ={walk, transit, car},
	party ={solo, couple, family},
	action ={view, click, dismiss, save, visit, booking},

The known output is either 1 or 0 for a pair of (trip, poi) depending on actions. 1 if action belongs to {booking, save, visit} and 0 if action belongs to {click, view, dismiss}. We could have had a rating on a scale of 1 to 6 for these actions, but I wanted to keep it simple and hence it is just a binary classification.


So the domain of our function is Trips * POIs. The values that we already know are in the interaction table, let us call that as function y. Let S be the set of tuples (trip, poi) in the interactions table. Now y: S -> {0,1}. Hence it forms as a sub-domain.

Let T be the set of trips and P be the set of POIs. We need to construct a function f: T*P -> [0,1] such that f agrees on y in the sub-domain S.
A POI that is closed every day of the trip cannot be recommended, and hence we drop it.
We do not simply want to construct a function from T*P by converting the columns into appropriate numbers, as it will be a very weak one and not capture anything.

The idea: Naturally, we want the some kind of tags of trips intersection with tags of POIs. We would also want to take into consideration the intersection of travellers tags from previous trips and POIs. We consider popularity of the POI as the third indicator. And finally, we do not want the budget to overshoot the price, and hence we subtract budget of the trip from price of the POI, and consider the max (0, price-budget).

The intersection measures are calculated using the Jaccard similarity. For 2 sets A and B, it is given by,
J(A,B)= |A intersection B|/ |A union B| if union is non empty and 0 if union is empty.

So the 4 feature functions are,
phi_1(t,p) = J(I_t, tau_p) (interest match)
phi_2(t,p) = J(H_t, tau_p) (history match)
phi_3(t,p) = popularity of p
phi_4(t,p) = max(0, price_p - budget_t)

where, I_t is tags of trips,
       tau_p is tags of POI,
       H_t is union of tags of previous trips

Next, out we use these 3 numbers as 3 variables of input to a logistic regression, as we get a value in (0,1).

So we have broken our function f as a composition of 2 functions:
f:T*P -> R^3 -> [0,1]
f= Phi composition g
where,
phi(t,p)= (phi_1(t,p), phi_2(t,p), phi_3(t,p))
     g(x,y,z)= sigmoid(w_0+ w_1*x+ w_2*y +w_3*z)

we will make use of budget in the next part.

The function f takes care of interests. We want another function which says if the POI is practical.
For this, we use another function c:T*P -> [0,1] which takes into account budget, mobility and party.
c(t, p) = c_budget * c_mobility * c_party.

where,
       c_budget(t,p)= 1− (phi_4(t,p)/2), so 1, 0.5, 0 for 0,1,2 tiers over.
       c_mobility(t,p) = 0.5 if mobility is not “car” and the POI is remote. Else gives 1.
c_party(t,p)= 0.3 if party is “family” and “nightlife” is among the tags. Else outputs 1.

Considering both f and compatibility function c, we will call our new function s.

So finally, s(t,p)= f(t,p)*c(t,p)

We train f, and compute c.

The trips are sorted by start date. The earliest 140 trips (2,727 rows) train the model and the latest 60 trips (1,164 rows) test it, so the model is always tested on trips that come after the ones it learnt from. Only the 4 numbers w_0 to w_3 of g are learnt, by minimising the log-loss, and they come out as w_0 = -2.043 and w = (2.462, 1.305, 0.250). On the test trips the log-loss is 0.413, against 0.440 for always predicting the base rate.

B) The code:

The code runs in 9 steps.

1. Load the data. Input: the three csv files. Output: the three tables, with tags, interests and open days turned into sets, and each action turned into a label y (1 or 0). This puts the data in a form we can compute with, and gives us y on S.

2. Open POIs. Input: a trip. Output: the weekdays the trip covers, and the POIs open on at least one of them. This removes the pairs that cannot be recommended (the set D*).

3. Features. Input: a trip and a list of POIs. Output: a table with one row per POI and the numbers phi_1 (interest match), phi_2 (history match), phi_3 (popularity) and phi_4 (over budget). This turns each (trip, POI) pair into numbers that measure how well they match.

4. Training rows and split. Input: the features and y for the POIs each trip was shown. Output: a train table (the earliest 140 trips) and a test table (the latest 60 trips). The model learns from the earlier trips and is checked on later ones it has not seen.

5. Train the model. Input: phi_1, phi_2, phi_3 and y of the train table. Output: the fitted logistic regression, which is our f (the preference). This learns how much each feature matters for a POI being liked.

6. Final score. Input: a trip and the feature table of its POIs. Output: for each POI the preference f, the compatibility c, and the score s = f * c. This adds the practical side (budget, mobility, party) to the preference.

7. Candidates and top 5. Input: a trip. Output: the 5 best POIs as JSON, each with score, weight and confidence. It builds a shortlist (30 by interest match plus 30 by popularity), scores it with step 6 and keeps the top 5. This is the answer the system gives.

8. Evaluation metrics. Input: the liked/not-liked values of one test trip's POIs, in the order the system ranked them. Output: Precision@5 and Recall@5. This measures how well the liked POIs are placed near the top.

9. Results. Input: the trained system and the test trips. Output: the JSON for trip 171, the metrics of the model against the popularity baseline, and the lists for three example trips (2, 195 and 196). This shows that the system works and gives different lists for different trips.


C) Cold start:

New traveller: H_t is empty, so phi_2 = 0 and the ranking uses the stated interests and popularity.


D) Scoring:

score is f(t,p)*c(t,p)
preference is f= g(phi(t,p))
compatibility is c(t,p)
weights is just score divided by sum of scores


E) Evaluation:

Precision@5 is the fraction of the top 5 recommended POIs that were liked, and Recall@5 is the fraction of all the POIs the traveller liked that appear in the top 5, both averaged over the 57 test trips that have at least one liked POI.
On the test trips, the model gets Precision@5 = 0.344 and Recall@5 = 0.558, against 0.151 and 0.199 for the popularity baseline.


F) Data Preparation:

I used an AI assistant to write the script 'generate_data.py' (fixed seed), which creates the three tables: 200 POIs, 200 trips (100 travellers with 2 trips each) and 4000 interactions (20 shown POIs per trip, 18% liked). In the script, the chance of a like goes up when the tags match, and goes down when the POI is over budget, remote for a traveller without a car, or nightlife for a family. A coin flip then decides the action. So the data is synthetic and the results show that the method works, not how it would perform on real data. In the code (step 1), the tags, interests and open days are turned into sets, and each action is turned into the label y.
The main code 'pipeline.py', was also written with the help of an AI assistant. I designed the approach and the mathematical formulation, and I went through the code step by step.


G) Running the code:


(Python 3 with pandas, numpy and scikit-learn is used)
clone the repo.
create a virtual env.

pip install -r requirements.txt
python generate_data.py (writes pois.csv, trips.csv and interactions.csv into data/, with a fixed seed so the data is the same every time)
python pipeline.py (trains the model and prints the JSON for trip 171, the evaluation, and the lists for three example trips; it takes a few seconds)

Run both commands from the project folder, since the code reads the files from 'data/'