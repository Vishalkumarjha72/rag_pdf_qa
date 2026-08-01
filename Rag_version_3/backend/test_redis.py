"""
Isolated smoke test for the Redis connection — no FastAPI, no LangGraph.

Requires a Redis server reachable at localhost:6379. For local testing
without the full docker-compose stack:
    docker run -d -p 6379:6379 --name local-redis redis:7-alpine

Run from inside backend/: python test_redis.py
"""

from app.redis_client import get_redis_client, ping_redis, RedisConnectionError


def main():
    print("Pinging Redis...")
    try:
        ping_redis()
    except RedisConnectionError as e:
        print(f"  FAILED: {e}")
        print("  Is a Redis container running on localhost:6379?")
        return

    print("  Connected.")

    client = get_redis_client()

    print("\nTesting basic set/get...")
    client.set("test_key", "hello from test_redis.py")
    value = client.get("test_key")
    print(f"  Wrote and read back: {value!r}")
    assert value == "hello from test_redis.py", "Value mismatch!"

    print("\nTesting TTL (expiry)...")
    client.setex("test_key_with_ttl", 60, "expires in 60s")
    ttl = client.ttl("test_key_with_ttl")
    print(f"  TTL on key: {ttl} seconds (should be close to 60)")

    print("\nCleaning up test keys...")
    client.delete("test_key", "test_key_with_ttl")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
