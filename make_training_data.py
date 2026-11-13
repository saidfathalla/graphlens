#!/usr/bin/env python3
"""Build the GraphLens layout-classifier training corpus and validate it with
leave-one-DOMAIN-out cross-validation (LODO-CV).

Columns of training_data.csv
  columns_text  split column names (camelCase / snake_case -> words)
  descriptor    domain-agnostic *shape* tokens, computed at run time by describe_view()
  target_chart  Line | Bar | Geomap | Scatter | Boxplot | Violin | Heatmap
  domain        grouping key for LODO-CV only (never a model feature)

IMPORTANT: this corpus is SYNTHETIC. Use it to bootstrap and to unit-test the pipeline.
For the paper, replace/extend it with real, independently labelled result tables
(see review_report.md, Section 'Layout classifier') and report accuracy on THAT corpus.
"""
import re, random, argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report

DOMAINS = {
 "scholarly": dict(time=["year","startYear","trackingYear","editionYear"], count=["submitted","accepted","papers","submissions"],
                   ratio=["acceptanceRate","acceptance_ratio"], cat=["field","researchField","subdomain","venueTier"],
                   geo=["country","hostingCountry","heldInCountry","nation"], ent=["acronym","seriesAcronym","venueName","series"],
                   clus=["clusterProfile","clusterLabel","agnosticProfile"]),
 "retail":    dict(time=["month","orderYear","fiscalYear","date"], count=["units","revenue","orders","sales"],
                   ratio=["conversionRate","returnRatio","margin"], cat=["category","productLine","segment","channel"],
                   geo=["country","region","storeCountry","market"], ent=["storeName","sku","brand","customerId"],
                   clus=["clusterProfile","segmentCluster","customerBand"]),
 "health":    dict(time=["year","admissionYear","reportYear","week"], count=["cases","admissions","patients","tests"],
                   ratio=["mortalityRate","positivityRatio","occupancy"], cat=["diagnosisGroup","ward","specialty","ageBand"],
                   geo=["country","province","district","nation"], ent=["hospitalName","clinicId","trialAcronym","provider"],
                   clus=["clusterProfile","riskBand","cohortCluster"]),
 "energy":    dict(time=["year","hour","commissionYear","period"], count=["capacityMW","generation","plants","consumption"],
                   ratio=["loadFactor","efficiency","shareRatio"], cat=["fuelType","sector","gridZone","operator"],
                   geo=["country","state","hostCountry","region"], ent=["plantName","utilityId","stationCode","operatorName"],
                   clus=["clusterProfile","loadBand","plantCluster"]),
 "education": dict(time=["year","academicYear","enrolYear","term"], count=["students","enrolments","graduates","applications"],
                   ratio=["admitRate","dropoutRatio","passRate"], cat=["faculty","programType","level","discipline"],
                   geo=["country","city","homeCountry","region"], ent=["universityName","courseCode","schoolId","institution"],
                   clus=["clusterProfile","performanceBand","schoolCluster"]),
 "mobility":  dict(time=["year","tripDate","surveyYear","day"], count=["trips","passengers","vehicles","distanceKm"],
                   ratio=["modalShare","delayRatio","occupancyRate"], cat=["mode","operatorType","lineType","purpose"],
                   geo=["country","city","origin","zone"], ent=["routeId","stopName","lineCode","operator"],
                   clus=["clusterProfile","demandBand","lineCluster"]),
}

# chart -> list of (column roles, mandatory shape tokens, [(optional token, probability)])
RECIPES = {
 "Line":    [(["time","count"],        ["__temporal","__numeric_1"],                               [("__skewed",.5)]),
             (["time","cat","count"],  ["__temporal","__numeric_1","__lowcard","__many_rows_per_key"], [("__skewed",.5)]),
             (["time","clus","count"], ["__temporal","__numeric_1","__lowcard","__many_rows_per_key"], [("__skewed",.5)])],
 "Bar":     [(["ent","count"],         ["__highcard","__numeric_1","__one_row_per_key"],           [("__skewed",.6)]),
             (["cat","count"],         ["__lowcard","__numeric_1","__one_row_per_key"],            [("__skewed",.4)])],
 "Geomap":  [(["geo","count"],         ["__geo","__numeric_1","__one_row_per_key"],                [("__skewed",.7)])],
 "Scatter": [(["count","ratio"],       ["__numeric_2"],                                            [("__bounded",.7),("__skewed",.7)]),
             (["ent","count","ratio"], ["__numeric_2","__highcard","__one_row_per_key"],           [("__bounded",.7),("__skewed",.7)]),
             (["clus","count","ratio"],["__numeric_2","__lowcard","__many_rows_per_key"],          [("__bounded",.7),("__skewed",.7)])],
 "Boxplot": [(["cat","ratio"],         ["__lowcard","__numeric_1","__many_rows_per_key","__bounded"], [])],
 "Violin":  [(["clus","count"],        ["__lowcard","__numeric_1","__many_rows_per_key","__skewed"],  [])],
 "Heatmap": [(["time","count","count","ratio"], ["__numeric_3plus"],                               [("__temporal",.7),("__skewed",.6),("__bounded",.6)])],
}
ALL_SHAPE = ["__temporal","__numeric_1","__numeric_2","__numeric_3plus","__lowcard","__highcard","__geo",
             "__one_row_per_key","__many_rows_per_key","__bounded","__skewed"]

def split_identifier(s):
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s).replace("_", " ").lower()

def build(seed=42, variants=8, noise=0.04):
    rnd = random.Random(seed); rows = []
    for dom, voc in DOMAINS.items():
        for chart, recs in RECIPES.items():
            for roles, must, opts in recs:
                for _ in range(variants):
                    cols = [rnd.choice(voc[r]) for r in roles]
                    if rnd.random() < .25: cols.append(rnd.choice(voc["cat"]))       # distractor name
                    rnd.shuffle(cols)
                    toks = list(must) + [t for t, p in opts if rnd.random() < p]
                    # label-noise on shape tokens: simulates mis-detected dtype / cardinality
                    toks = [t for t in toks if rnd.random() > noise]
                    if rnd.random() < noise: toks.append(rnd.choice(ALL_SHAPE))
                    rows.append(dict(columns_text=" ".join(split_identifier(c) for c in cols),
                                     descriptor=" ".join(sorted(set(toks))),
                                     target_chart=chart, domain=dom))
    return pd.DataFrame(rows).drop_duplicates(["columns_text","descriptor","target_chart"]).reset_index(drop=True)

def lodo(df, use_names, use_shape, depth=8):
    X = ((df.columns_text if use_names else "") + " " + (df.descriptor if use_shape else "")).str.strip()
    Xv = CountVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(X)
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=depth, min_samples_leaf=2, random_state=42)
    pred = cross_val_predict(clf, Xv, df.target_chart, groups=df.domain, cv=LeaveOneGroupOut())
    return accuracy_score(df.target_chart, pred), f1_score(df.target_chart, pred, average="macro"), pred

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("-o","--out",default="training_data.csv"); a = ap.parse_args()
    df = build(); df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} rows | {df.domain.nunique()} domains | {df.target_chart.nunique()} classes")
    print("majority-class baseline acc = %.3f" % df.target_chart.value_counts(normalize=True).iloc[0])
    print("\nLeave-one-domain-out CV (entropy decision tree):")
    for name, (n, s) in {"column names only (v1 behaviour)": (True, False),
                         "shape descriptor only":            (False, True),
                         "names + shape descriptor":         (True, True)}.items():
        acc, f1, pred = lodo(df, n, s); print(f"  {name:36s} acc={acc:.3f}  macro-F1={f1:.3f}")
    print(); print(classification_report(df.target_chart, pred, digits=3))
