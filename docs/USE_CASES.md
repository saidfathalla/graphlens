# Running the use cases

GraphLens ships with three use cases on three unrelated knowledge graphs. The code is the same for all of them; each case only has its own SPARQL query (`queries/`) and profile (`profiles/`). Every case can be run **offline** (from a local copy of the data) and, where the data is published online, **live** (straight from the web).

| Use case | Knowledge graph | Rows analysed | Offline | Live |
|---|---|---|---|---|
| [1. Scholarly events](#1-eventskg-scholarly-events) | EVENTSKG | one per conference edition | local `.ttl` file | RDF file downloaded from GitHub |
| [2. Films](#2-dbpedia-films) | DBpedia | one per film | local `.ttl` file | DBpedia SPARQL endpoint |
| [3. Olympic Games](#3-olympic-games) | Olympics KG (wallscope) | one per national delegation and gender | local SPARQL server or `.ttl` file | not available (data is published as a zip) |

## Before you start

Run everything from the repository root.

```bash
pip install -r requirements.txt
```

`data/training_data.csv` (layout classifier) and `data/world_countries.geojson` (world map) must be in `data/`. Every run writes its figures and tables to the folder given with `-o`. Run times below were measured on a laptop and depend on your hardware.

---

## 1. EVENTSKG: scholarly events

**Question:** how have submissions and acceptance rates of top computer-science conferences developed, by research field and host country?

| Role | Variable |
|---|---|
| Volume / success / ratio | submitted papers / accepted papers / acceptance rate |
| Category / location / time | research field / host country / start date |
| Entity | event acronym (grouped into event series) |

**Offline** (local copy of the graph):

```bash
python3 graphlens.py -q queries/eventskg-query.rq -f data/eventskg.ttl \
    -p profiles/eventskg-profile.yaml -o output/eventskg
```

**Live** (downloads the current release from GitHub, about 10 s):

```bash
python3 graphlens.py -q queries/eventskg-query.rq \
    -u https://raw.githubusercontent.com/saidfathalla/EVENTSKG-Dataset/master/EVENTSKG.ttl \
    -p profiles/eventskg-profile.yaml -o output/eventskg-online
```

Notes:

- The online release (3.0, 39,480 triples) is newer than older local copies and uses `seo:ResearchField` where older releases use `seo:field`. The query reads both, so either file works, but the numbers differ between releases. State which release you used when you report results.
- The profile keeps editions from 2000 to 2026. On the online release this gives 1,764 analysed editions, 8 research fields and 59 host countries; submissions grow by about 9.6% per year.

---

## 2. DBpedia: films

**Question:** how do film budgets and box-office recovery relate, by original language and country of production?

| Role | Variable |
|---|---|
| Volume / success / ratio | budget / recouped budget (gross, capped at the budget) / recovery rate |
| Category / location / time | original language / country / release year |
| Entity | film |

**Live, sample** (DBpedia public SPARQL endpoint, about 2,300 films, 3-6 minutes):

```bash
python3 graphlens.py -q queries/dbpedia-query.rq -e https://dbpedia.org/sparql \
    -p profiles/dbpedia-profile.yaml -o output/dbpedia-live \
    --page-size 10000 --timeout 300 --max-rows 2000 --no-count-check
```

`--max-rows` fetches whole slices of films until 2,000 rows are reached. The slices are buckets of an MD5 hash of the film IRI, so the sample is random with respect to budget, language and year, and the same films are chosen in every run. The log line `Sample: k of 16 slices` and `sample_partitions` in `run_report.json` record the sample size.

Each slice (about 780 films) fits in one page, so `--stable-order` is not needed, and `--no-count-check` saves one request per slice; keep the check for runs you report.

**Live, all films** (about 12,500 films, roughly 30 minutes):

```bash
python3 graphlens.py -q queries/dbpedia-query.rq -e https://dbpedia.org/sparql \
    -p profiles/dbpedia-profile.yaml -o output/dbpedia-live-all --page-size 10000 --timeout 300
```

**Offline** (local copy of the same 2,300-film sample, `data/dbpedia_films.nt`):

```bash
python3 graphlens.py -q queries/dbpedia-query.rq -f data/dbpedia_films.nt \
    -p profiles/dbpedia-profile.yaml -o output/dbpedia
```

The copy holds only the triples the query reads. It is made with `queries/dbpedia-snapshot.rq`, one small request per 1/256 slice of films (DBpedia limits the size of each answer); to refresh it:

```bash
rm -f data/dbpedia_films.nt
for a in 0 1 2; do for b in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
  sed "s/__SLICE__/$a$b/" queries/dbpedia-snapshot.rq |
    curl -sS --retry 3 https://dbpedia.org/sparql --data-urlencode query@- \
         --data-urlencode format=text/ntriples >> data/dbpedia_films.nt
done; done
```

DBpedia changes between releases, so a refreshed copy can give slightly different numbers.

Notes:

- The query returns one row per film. The release year comes from `dbo:releaseDate`, else the infobox `dbp:released`, else a "YYYY films" category; films without any of them are dropped and listed in `normalization_dropped.csv`. No default year, country or language is ever filled in. When a film has several values (e.g. two languages), the query takes the first in alphabetical order (`MIN`), so every run gives the same result.
- In endpoint mode the query runs as 16 slices of films (the `# GRAPHLENS-PARTITION ?film` line), so each slice stays within DBpedia's time limit. The line sits in a small sub-query of its own: in the large aggregated block DBpedia ignores the slice filter. If a slice needs several pages, add `--stable-order`, otherwise DBpedia may change the row order between pages. The run is complete when the log shows no `INCOMPLETE RESULT` or "identical rows" warning and `run_report.json` shows `"complete": true` under `ingest`. If slices time out, add `--partitions 256`.
- Data quality rules in the query: only budgets and grosses typed in US dollars are used (DBpedia mixes currencies and untyped numbers), budgets under $10,000 are excluded as unit errors (e.g. "15" for $15 million), and budgets over $500 million as currency errors (e.g. 8 billion Indonesian rupiah typed as dollars).
- Two bands (Low and High Recovery): about 60% of films recover exactly 100% because gross is capped at the budget, so a third cluster would only split the recovered films by budget size.
- Original language is used as the category rather than genre: DBpedia's genre values for films are missing for about 90% of films and are mostly music genres. Language links (`dbo:language`) and infobox text (`dbp:language`) are merged, so "English_language" and "English" count as one.
- The charts show the 10 most frequent languages and group the rest as "Other" (profile option `analysis: {top_categories: 10}`); this affects the figures only, not the clusters or trends.

---

## 3. Olympic Games

**Question:** how have national delegations at the Summer Olympics grown since 1896, how efficiently do they win medals, and how has women's participation changed?

| Role | Variable |
|---|---|
| Volume / success / ratio | athlete entries / medal-winning entries / medal rate |
| Category / location / time | gender / country / Games year |
| Entity | National Olympic Committee (NOC) |

Data: [wallscope/olympics-rdf](https://github.com/wallscope/olympics-rdf), 1.78 million triples, 1896-2016. Get it once:

```bash
mkdir -p data
curl -L -o data/olympics-ttl-nodup.zip \
    https://raw.githubusercontent.com/wallscope/olympics-rdf/master/data/olympics-ttl-nodup.zip
unzip -o data/olympics-ttl-nodup.zip -d data/          # creates data/olympics.ttl (63 MB)
```

**Offline, recommended** (local SPARQL server; about 30 s per run, low memory):

```bash
# once: install the server and load the data (the olympics_db folder keeps it)
pip install oxigraph
oxigraph load --location olympics_db --file data/olympics.ttl

# every run, in one terminal: start the server in the background, run GraphLens, stop the server
oxigraph serve-read-only --location olympics_db --bind localhost:7878 > oxigraph.log 2>&1 &
SERVER=$!
sleep 2
python3 graphlens.py -q queries/olympics-query.rq -e http://localhost:7878/query \
    -p profiles/olympics-profile.yaml -o output/olympics
kill $SERVER
```

The server keeps running until it is stopped; "Listening for requests" means it is ready.

**Offline, file mode** (no server, but rdflib needs more than 8 GB of RAM and several minutes for this graph):

```bash
python3 graphlens.py -q queries/olympics-query.rq -f data/olympics.ttl \
    -p profiles/olympics-profile.yaml -o output/olympics
```

**Live:** not available. The graph is published only as a zip file and there is no public SPARQL endpoint for it.

Notes:

- 4,822 rows (NOC x Games x gender) from 29 Summer Games and 229 NOCs, 213,951 athlete entries. Women's share of entries rises from 1.6% (1900) to 17.9% (1960) and 45.7% (2016).
- About 2% of athletes are linked to more than one NOC; the graph does not record which NOC they represented at each Games, so they are excluded.
- For the map only, historical teams (e.g. Soviet Union, Yugoslavia) are drawn on the territory of their main successor state; these lines are marked in the profile and can be deleted. Six entries cannot be placed on a map (e.g. Refugee Olympic Athletes) and are reported as a warning.
- For the Winter Games, replace `Summer` with `Winter` in `queries/olympics-query.rq`.

---

## Refreshing the figures in the README

The README shows each chart type once, from `docs/figures/`. After running the cases above, copy the new figures there:

```bash
mkdir -p docs/figures/eventskg docs/figures/dbpedia docs/figures/olympics
cp output/eventskg-online/fig{3_field_growth,4_geo_distribution,6_ratio_by_category}.png docs/figures/eventskg/
cp output/dbpedia-live/fig{1_cluster_scatter,5_correlation,7_top_entities}.png docs/figures/dbpedia/
cp output/olympics/fig{2_cluster_trend_ols,8_cluster_density}.png docs/figures/olympics/
```
