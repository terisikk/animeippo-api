"""Clear the Redis cache."""

import redis


def cli():
    r = redis.Redis(host="redis-stack-server", port=6379)
    r.flushall()
    print("Cache cleared")


if __name__ == "__main__":
    cli()
