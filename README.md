# Dev Guide

## Prereqs

1. git
2. docker 

## Project Structure 

+ `src/` - source code 
+ `data/` - untracked git folder for the dataset provided by the professor 
+ `pyrpoject.toml` and `uv.lock` - defines python dependencies 
+ `docker-compose.yml` - connects local code to Docker container for real-time changes 

## Getting Started 

#### 1. Clone the Repo

#### 3. Start the Environment 

```
docker compose up -d --build
```

this command:
1. builds the docker image
2. installs all dependencies 
3. keeps the container running in the background

_Note: You only need to use the `--build` flag the first time you build the Docker image or when someone adds new dependencies_

#### 4. To Run the Code

```
docker compose exec ir-dev python main.py
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

