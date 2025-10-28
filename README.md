
# Bugs

Cannot process people with no scores added

# Dependencies for development

* uv >= 0.6.7

# How to run locally

`make install`

For testing with web-api version:

`uv run flask --app app run --host 0.0.0.0`

For testing without web front:

`uv run python animeippo/main.py`

# VSCode devcontainers

To allow caching (and probably to allow starting the container at all), create a docker network called animeippo-network

`docker network create animeippo-network`

and run redis with 

`docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest`

and add the container to the correct network with

`docker network connect animeippo-network [redis-stack-server]`.

# Cache Pre-loading (Production)

For optimal performance, the cache should be pre-loaded with upcoming anime data. This is especially important in production to ensure users get instant responses.

## Pre-loading Script

The pre-loading script fetches and caches:
- Full year of anime for previous, current, and next year
- Genre and tag collections from AniList API

### Running in Docker Container

Since production runs in a Docker container, execute the script inside the container:

```bash
# One-time manual execution
docker exec -it <container-name> .venv/bin/animeippo-preload-cache

# Skip genre/tag collections (only pre-load yearly anime)
docker exec -it <container-name> .venv/bin/animeippo-preload-cache --skip-static
```

### Automated Nightly Pre-loading

Set up a cron job on the Docker host to run the script nightly:

```bash
# Edit crontab on the host machine
crontab -e

# Add this line to run at 2 AM daily
0 2 * * * docker exec <container-name> .venv/bin/animeippo-preload-cache >> /var/log/animeippo-preload.log 2>&1
```

Replace `<container-name>` with your actual container name.

**Note**: The script requires Redis to be running and will exit with error code 1 if Redis is not available.
