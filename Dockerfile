FROM python:3.14

# copy the uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# set working dir inside container
WORKDIR /app 

# copy dependency files 
COPY pyproject.toml uv.lock ./ 

# install dependencies in container 
RUN uv sync --frozen --no-cache 

COPY . . 

CMD ["python", "main.py"]
