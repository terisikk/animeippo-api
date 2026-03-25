# Rate Limiting & Request Queue

## Status: Partially implemented

### Done

- [x] 429 handling with `Retry-After` in AniList connection (`rate_limited` decorator)
- [x] Rate limit header tracking (`X-RateLimit-Remaining`, `X-RateLimit-Limit`)
- [x] Warning logging when remaining requests are low
- [x] Sequential pagination (replaced parallel `asyncio.gather` with sequential fetches)

### Remaining

#### 2. Redis job queue + status endpoint

Create `src/animeippo/queue/jobs.py`:
- Job submission: derive idempotent key from params (`job:recommend:{user}:{year}:{season}`)
- If job already exists (queued/processing), return existing job ID and position
- Store jobs in Redis: list `animeippo:queue` (FIFO), hash `animeippo:job:{id}` with status/params/result
- Job statuses: `queued` → `processing` → `completed` / `failed`
- Completed jobs stay in Redis with configurable TTL for re-requests

Add to `app.py`:
- Modify `/recommend` and `/analyse` endpoints:
  - If cached and fresh → serve immediately (200)
  - If stale/missing → submit job to Redis queue, return 202 with job ID and position
  - `?refresh=true` → always queue, ignore cache
- New `GET /status/{job_id}` endpoint → returns status, position, result when done
- Return 503 with `Retry-After` when queue exceeds `QUEUE_MAX_SIZE`

Response formats:
```json
{"status": "queued", "job_id": "abc123", "position": 3}
{"status": "completed", "data": {...}}
{"status": "queue_full", "retry_after": 30}
```

#### 3. Worker container

Create `src/animeippo/queue/worker.py`:
- Separate Python process that pops jobs from Redis queue
- Processes one job at a time (sequential)
- Uses existing `recommender.recommend_seasonal_anime()` / `profiler.analyse()`
- Tracks AniList rate limit state in Redis (`animeippo:rate_remaining`)
- Throttles when remaining is low
- Stores JSON results in Redis with TTL
- Yields to user-submitted jobs over background refresh

Add to Docker deployment:
- New `worker` service in docker-compose sharing Redis network
- Same image as API, different entrypoint (`python -m animeippo.queue.worker`)

#### 4. Cache refresh scheduler

Create `src/animeippo/queue/scheduler.py`:
- Runs inside worker container
- Periodically checks which users have stale cache (> `CACHE_REFRESH_TTL_HOURS`)
- During non-peak window (`CACHE_REFRESH_WINDOW`, e.g. 02:00-06:00), submits refresh jobs
- Low priority — pauses if user-submitted jobs are in queue

#### 5. Frontend integration

Frontend changes (separate repo):
- On 202 response: store job ID, show queue position, poll `/status/{job_id}` every 3-5 seconds
- On 200: display results immediately (cached hit)
- On 503: show "Server busy" with retry countdown
- On page refresh: re-submit request, backend returns existing job (idempotent)

### Configuration

```env
QUEUE_MAX_SIZE=10              # max queued jobs before 503
CACHE_REFRESH_TTL_HOURS=24     # refresh stale cache after this
CACHE_REFRESH_WINDOW=02:00-06:00  # non-peak refresh window
RATE_LIMIT_THROTTLE=10         # start throttling when remaining < this
```
