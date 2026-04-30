# Dev Guide

## Prereqs

1. git
2. docker 

## Project Structure 

+ `src/` - source code 
+ `dataset/` - folder for the dataset provided by the professor 
+ `pyrpoject.toml` and `uv.lock` - defines python dependencies 
+ `docker-compose.yml` - connects local code to Docker container for real-time changes 

## Getting Started 

#### 1. Clone the Repo

#### 2. Start the Environment 

```
docker compose up -d --build
```

this command:
1. builds the docker image
2. installs all dependencies 
3. keeps the container running in the background

_Note: You only need to use the `--build` flag the first time you build the Docker image or when someone adds new dependencies_

#### 3. To Run the Code

```
docker compose exec ir-dev uv run <file_name>
```

this command executes through the container 

## Adding New Packages

you can add new packages using `uv` through Docker

```
docker compose exec ir-dev uv add <pkg_name>
```

this command updates `pyproject.toml` and `uv.lock` locally 

after commiting these files, a team member can `git pull` these new added dependencies and then rebuild their container with:

```
docker compose up -d build
```

## Stopping the Environment 

```
docker compose down
```
# Using Search Models
## Processing Collection
Initialize DataProcessing during Indexing Processes.
```
p = DataProcessing(DATA_DIR)
```
`DATA_DIR` includes documents, queries, and relevance scores.
### Precomputed Values
Assuming `p` is the initialize variable for `DataProcessing`.
- `p.docs` DataFrame of `documents.json`
- `p.queries` DataFrame of test queries provided in `queries.json`
- `p.qrels` DataFrame of Query_id, Document_id, and relevance (1 means relevant)
## Search Engines
After initializing data processing, use the given class variables to initialize a search engine. 
```
bm25 = BM25Search(p.docs, p.qrels)
vsm = VSMSearch(p.docs, p.qrels, p.vectorizer, p.tfidf)
bim = BIMSearch(p.docs, p.qrels)
unigram = LanguageModel(docs=p.docs, relevance=p.qrels, model='unigram', lambda_=0.3)
bigram = LanguageModel(docs=p.docs, relevance=p.qrels, model='bigram', lambda_=0.3)
```
## Searching
To search using a search engine, call `getAPScores()` and define the query.
```
vsm.getAPScores(p.queries)
```
this returns a sorted list of documents (descending) that match the query.
## Adding Live Queries
For the demo, we need a way to pass in a live query. 
