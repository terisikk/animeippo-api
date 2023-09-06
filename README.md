
# Bugs

Cannot process people with no scores added

# Dependencies for development

* Poetry >= 1.4.0

# How to run locally

`poetry install` or `make install`

For testing with web-api version:

`poetry shell && flask --app app run --host 0.0.0.0`

For testing without web front:

`potery shell && python animeippo/main.py``

# VSCode devcontainers

To allow caching (and probably to allow starting the container at all), create a docker network called animeippo-network

`docker network create animeippo-network`

and run redis with 

`docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest`

and add the container to the correct network with

`docker network connect animeippo-network [redis-stack-server]`.
