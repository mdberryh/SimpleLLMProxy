"""GPU Wakelock Manager — auto-start/stop GPU containers on demand.

================================================================================
[AUTOWAKE-TEMP] TEMPORARY FEATURE — REMOVAL GUIDE
================================================================================
This entire module exists only to work around current AMD GPU drivers, which
cannot spin idle GPUs down on their own. We start/stop the GPU containers and
wait for warmup. Once the drivers handle idle power management natively, this
machinery is no longer needed and should be deleted.

The feature is already neutralized at runtime by `autoWakeEnabled: false` in
app.yml — when disabled, route() collapses to a simple "forward to the
highest-priority local model" passthrough (see the `if not self.enabled:`
branch in route()). That passthrough IS the intended post-removal behavior.

To PHYSICALLY remove the feature when the time comes, grep the whole repo for
the tag `[AUTOWAKE-TEMP]` and delete/simplify each hit:

  1. src/gpu_managers/wakelock.py  — delete this module entirely, OR replace
     GpuWakelockManager with a ~10-line passthrough that keeps only route()'s
     disabled-mode logic (pick lowest-priority-number localModel + request_path).
  2. docker/proxy/app.yml          — remove the tagged auto-wake config keys
     (autoWakeEnabled, dockerNames, warmupSeconds, idleTimeoutMinutes,
     startupGroup, coldStartWaitSeconds). Keep localModels (routing still needs
     them) and the externalModels/externalFallbackEnabled block if you still
     want cloud fallback.
  3. src/proxy/server.py           — at each tagged call site, drop the wake-only
     calls (`_record_request`, `release`, the `used_external`/`external_auth`
     block) and replace `WAKELOCK.route(...)` with the passthrough lookup. The
     `/api/status` and `/metrics` wake fields can be removed too.

Nothing outside these tagged sites depends on the wake machinery.
================================================================================
"""

# Change Log
# ==============================================
# 2026-06-15 - Queue dequeues only to ready GPUs; restart stopped GPU; start retry-once
# 2026-06-15 - route() no longer holds self._lock during _classify_gpus (deadlock fix); added coldStartWaitSeconds wait before Venice fallback
# 2026-06-15 - Added externalFallbackEnabled toggle to disable Venice fallback (queue locally instead)
# 2026-06-15 - Added [AUTOWAKE-TEMP] removal guide/markers so the temporary auto-wake feature can be cleanly excised later
# 2026-06-15 - Queue overflow now offloads to external (Venice) per spec instead of always 503; falls back to 503 only when externalFallbackEnabled is false
# 2026-07-06 - release() no longer calls Docker while holding self._lock (container check moved outside the lock)
# 2026-07-06 - Fixed self-deadlock in _wait_for_gpu: _dequeue/_check_queue_overflow/_restart_stopped were called while holding the non-reentrant _queue_lock via the condition
# 2026-07-06 - release() now frees the slot and notifies the queue before the Docker container check so a hung Docker call cannot starve waiters
# ==============================================

import logging
import os
import threading
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

logger = logging.getLogger("proxy.wakelock")


class GpuWakelockManager:
    """
    Manages GPU container lifecycle:
      - Starts containers from startupGroup on first request
      - Health-probes /health until GPU is ready
      - Round-robins between ready GPUs
      - Queues requests when all GPUs busy
      - Shuts down all containers after idle timeout
    """

    def __init__(self, config: dict):
        self.config = config
        self.enabled = False
        self.docker = None
        self.gpus = {}
        self._idle_thread = None
        self._lock = threading.Lock()
        self._last_request_at = time.time()
        self._request_timestamps = []
        self._last_used_gpu = None
        self._queue_entries = []
        self._queue_lock = threading.Lock()
        self._queue_event = threading.Condition(self._queue_lock)
        self._overflow_active = False
        self._probe_thread = None
        self._probe_stop = threading.Event()
        self._health_cache = {}  # Track health status for load balancing

        # Always initialize GPU list for health checking and load balancing
        for gm in config.get("localModels", []):
            name = gm["name"]
            self.gpus[name] = {
                "url": gm["url"],
                "dockerName": gm.get("dockerName", name),
                "priority": gm.get("priority", 99),
                "capabilities": gm.get("capabilities", ["text"]),
                "state": "unknown",
                "busy": False,
                "last_request": 0,
                "started_at": 0,
            }

        # Support both old key name and new key name
        wake_enabled = config.get("autoWakeEnabled", config.get("dockerShutdown", False))
        if not wake_enabled:
            logger.info("Wakelock: auto-wake disabled, but health checking enabled for %d GPUs", len(self.gpus))
            # Start health probe thread even when auto-wake is disabled
            if self.gpus:
                self._probe_thread = threading.Thread(
                    target=self._health_probe_loop,
                    daemon=True,
                    name="gpu-health-probe",
                )
                self._probe_thread.start()
                logger.info("Wakelock: health probing enabled (auto-wake disabled)")
            return

        try:
            import docker
            self.docker = docker.from_env()
        except Exception as e:
            logger.warning("Wakelock: docker unavailable (%s), disabled", e)
            self.enabled = False
            return

        self.enabled = True

        for gm in config.get("localModels", []):
            name = gm["name"]
            self.gpus[name] = {
                "url": gm["url"],
                "dockerName": gm["dockerName"],
                "priority": gm.get("priority", 99),
                "capabilities": gm.get("capabilities", ["text"]),
                "state": "stopped",
                "busy": False,
                "last_request": 0,
                "started_at": 0,
            }

        self.startup_group = config.get("startupGroup", [])

        self._idle_thread = threading.Thread(target=self._idle_loop, daemon=True)
        self._idle_thread.start()

        if self.gpus:
            self._probe_thread = threading.Thread(
                target=self._health_probe_loop,
                daemon=True,
                name="gpu-health-probe",
            )
            self._probe_thread.start()
            logger.info("Wakelock: active health probing enabled")

        logger.info("Wakelock: enabled, managing %d GPUs", len(self.gpus))

    # ------------------------------------------------------------------
    # Docker helpers
    # ------------------------------------------------------------------

    def _running(self, name: str) -> bool:
        try:
            c = self.docker.containers.get(name)
            return c.status == "running"
        except Exception:
            return False

    def _start(self, name: str) -> bool:
        # Per NK-09 edge cases: retry container start once before giving up.
        for attempt in (1, 2):
            try:
                c = self.docker.containers.get(name)
                c.start()
                logger.info("Wakelock: started %s (attempt %d)", name, attempt)
                return True
            except Exception as e:
                logger.error(
                    "Wakelock: failed to start %s (attempt %d): %s", name, attempt, e
                )
                if attempt == 1:
                    time.sleep(2)
        return False

    def _stop(self, name: str) -> None:
        try:
            c = self.docker.containers.get(name)
            if c.status == "running":
                c.stop(timeout=30)
                logger.info("Wakelock: stopped %s", name)
        except Exception as e:
            logger.error("Wakelock: failed to stop %s: %s", name, e)

    # ------------------------------------------------------------------
    # Health probing
    # ------------------------------------------------------------------

    def _health_probe_loop(self) -> None:
        poll_interval = 2.0
        warmup_hard = self.config.get("warmupSeconds", 180)

        while not self._probe_stop.is_set():
            with self._lock:
                for name, gpu in self.gpus.items():
                    # When auto-wake is disabled, GPUs start in "unknown" state
                    # When auto-wake is enabled, GPUs start in "starting" state
                    if gpu["state"] not in ("starting", "unknown"):
                        continue

                    # Only apply warmup timeout for "starting" state (auto-wake enabled)
                    # Skip timeout if started_at is 0 (container started externally)
                    if gpu["state"] == "starting" and gpu["started_at"] > 0:
                        elapsed = time.time() - gpu["started_at"]
                        if elapsed >= warmup_hard:
                            logger.warning(
                                "Wakelock: %s hard timeout after %.0fs (>%ds), marking stopped",
                                name, elapsed, warmup_hard,
                            )
                            gpu["state"] = "stopped"
                            self._health_cache[name] = False
                            continue

                    health_url = gpu["url"].rstrip("/") + "/health"
                    is_healthy = self._probe_url(health_url)
                    
                    # Update health cache
                    self._health_cache[name] = is_healthy
                    
                    if is_healthy and gpu["state"] in ("starting", "unknown"):
                        gpu["state"] = "ready"
                        logger.info(
                            "Wakelock: %s ready via health probe",
                            name,
                        )
                    elif not is_healthy and gpu["state"] == "ready":
                        # GPU was ready but now unhealthy
                        gpu["state"] = "unknown"
                        logger.warning(
                            "Wakelock: %s became unhealthy, marking unknown",
                            name,
                        )

            self._probe_stop.wait(timeout=poll_interval)

    @staticmethod
    def _probe_url(url: str, timeout: int = 3) -> bool:
        try:
            resp = urllib_request.urlopen(url, timeout=timeout)
            return 200 <= resp.status < 300
        except (urllib_error.URLError, urllib_error.HTTPError, OSError, ConnectionError, TimeoutError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Queue overflow
    # ------------------------------------------------------------------

    def _check_queue_overflow(self) -> bool:
        with self._queue_lock:
            if not self._overflow_active:
                return False
            if len(self._queue_entries) <= 3:
                self._overflow_active = False
                logger.info(
                    "Wakelock: queue overflow cleared (queue=%d <= 3)",
                    len(self._queue_entries),
                )
                return False
        return True

    def _enqueue(self) -> None:
        now = time.time()
        with self._queue_lock:
            self._queue_entries.append(now)
            queue_len = len(self._queue_entries)
            if queue_len > 3:
                oldest_wait = now - self._queue_entries[0]
                if oldest_wait > 120 and not self._overflow_active:
                    self._overflow_active = True
                    logger.warning(
                        "Wakelock: QUEUE OVERFLOW — queue=%d (>3), oldest wait=%.0fs (>120s)",
                        queue_len, oldest_wait,
                    )
            elif self._overflow_active:
                self._overflow_active = False
                logger.info(
                    "Wakelock: queue overflow cleared (queue=%d <= 3)", queue_len
                )

    def _dequeue(self) -> None:
        with self._queue_lock:
            if self._queue_entries:
                self._queue_entries.pop(0)
            if len(self._queue_entries) <= 3:
                if self._overflow_active:
                    self._overflow_active = False
                    logger.info(
                        "Wakelock: queue overflow cleared on dequeue (queue=%d)",
                        len(self._queue_entries),
                    )

    # ------------------------------------------------------------------
    # Idle shutdown
    # ------------------------------------------------------------------

    def _idle_loop(self) -> None:
        timeout = self.config.get("idleTimeoutMinutes", 15) * 60
        busy_timeout = 300

        while True:
            time.sleep(60)
            now = time.time()

            with self._lock:
                for name, gpu in self.gpus.items():
                    if gpu["busy"] and now - gpu["last_request"] > busy_timeout:
                        logger.warning(
                            "Wakelock: %s stuck busy for %.0fs, force-releasing",
                            name, now - gpu["last_request"],
                        )
                        gpu["busy"] = False
                        gpu["state"] = "ready"

                should_check_running = False
                if now - self._last_request_at > timeout:
                    if not any(g["busy"] for g in self.gpus.values()):
                        should_check_running = True

            if not should_check_running:
                continue

            running_status = {
                name: self._running(name)
                for name in self.startup_group
            }
            containers_to_stop = [
                name for name, running in running_status.items() if running
            ]

            if containers_to_stop:
                logger.info(
                    "Wakelock: group idle %.0fs, stopping %d containers",
                    now - self._last_request_at, len(containers_to_stop),
                )
                for name in containers_to_stop:
                    self._stop(name)
                with self._lock:
                    for gpu in self.gpus.values():
                        gpu["state"] = "stopped"

    # ------------------------------------------------------------------
    # GPU classification & selection
    # ------------------------------------------------------------------

    def _classify_gpus(self, required_capabilities: set = None) -> tuple:
        """
        Classify GPUs by checking Docker container state.

        Args:
            required_capabilities: Set of capabilities required (e.g., {"vision"} for multimodal)

        Splits Docker I/O from lock-protected state mutations:
        1. First, check each container's running status via Docker API (no lock)
        2. Then acquire lock and apply state transitions based on results
        """
        if required_capabilities is None:
            required_capabilities = set()

        # Phase 1: Docker API calls OUTSIDE the lock
        running_status = {}
        for name, gpu in sorted(self.gpus.items(), key=lambda x: x[1]["priority"]):
            running_status[name] = self._running(gpu["dockerName"])

        # Phase 2: State mutations INSIDE the lock
        ready_gpus = []
        starting_gpus = []
        with self._lock:
            for name, gpu in sorted(self.gpus.items(), key=lambda x: x[1]["priority"]):
                # Skip GPUs that don't have required capabilities
                if not required_capabilities.issubset(set(gpu.get("capabilities", ["text"]))):
                    continue

                running = running_status[name]
                if not running:
                    gpu["state"] = "stopped"
                elif gpu["busy"]:
                    gpu["state"] = "busy"
                elif gpu["state"] == "busy":
                    # Stale busy flag — GPU is actually idle, recover to ready
                    gpu["state"] = "ready"
                elif gpu["state"] not in ("ready", "starting"):
                    gpu["state"] = "starting"
                # else: stays "ready" or "starting"

                if gpu["state"] == "ready":
                    ready_gpus.append((name, gpu))
                elif gpu["state"] == "starting":
                    starting_gpus.append((name, gpu))

        return ready_gpus, starting_gpus

    def _pick_gpu(self, ready_gpus: list):
        if not ready_gpus:
            return None
        if len(ready_gpus) == 1:
            name, gpu = ready_gpus[0]
        else:
            candidates = [g for g in ready_gpus if g[0] != self._last_used_gpu]
            if not candidates:
                candidates = ready_gpus
            name, gpu = candidates[0]

        self._last_used_gpu = name
        gpu["busy"] = True
        gpu["last_request"] = time.time()
        return (name, gpu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _start_all(self) -> None:
        for name in self.startup_group:
            if not self._running(name):
                self._start(name)

    def route(self, request_path: str, multimodal: bool = False) -> tuple:
        """
        Route a request to the best available GPU.

        Args:
            request_path: The request path (e.g., /v1/chat/completions)
            multimodal: If True, only route to GPUs with 'vision' capability

        Returns (url, used_external, gpu_name).

        On queue overflow the request is offloaded to the external (Venice)
        provider when externalFallbackEnabled; otherwise RuntimeError
        ("QUEUE_OVERFLOW") is raised so server.py can return a clean 503.

        IMPORTANT: _classify_gpus() manages self._lock internally (Phase 1 is
        lock-free Docker I/O; Phase 2 acquires self._lock for state mutations).
        route() must NOT hold self._lock while calling _classify_gpus() — doing
        so would deadlock because Python's threading.Lock is not reentrant.
        Only _pick_gpu(), which mutates gpu state, is called under the lock.
        """
        self._last_request_at = time.time()

        # Determine required capabilities based on request type
        required_capabilities = {"vision"} if multimodal else set()

        if not self.enabled:
            # Health-aware load balancing when auto-wake is disabled
            # Find healthy GPUs from the health cache
            with self._lock:
                healthy_gpus = [
                    (name, gpu) for name, gpu in sorted(self.gpus.items(), key=lambda x: x[1]["priority"])
                    if self._health_cache.get(name, False)
                    and required_capabilities.issubset(set(gpu.get("capabilities", ["text"])))
                ]
            
            if healthy_gpus:
                # Load balance between healthy GPUs (round-robin)
                if len(healthy_gpus) == 1:
                    name, gpu = healthy_gpus[0]
                else:
                    # Pick the next GPU that wasn't last used
                    candidates = [g for g in healthy_gpus if g[0] != self._last_used_gpu]
                    if not candidates:
                        candidates = healthy_gpus
                    name, gpu = candidates[0]
                
                self._last_used_gpu = name
                logger.info("Wakelock: routing to healthy GPU %s (health-aware)", name)
                return gpu["url"].rstrip("/") + request_path, False, name
            
            # No healthy GPUs available - try external fallback
            if self._external_enabled():
                logger.warning("Wakelock: no healthy GPUs, routing to external fallback")
                return self._external_route(request_path)
            
            # Fall back to highest priority GPU even if unhealthy
            local = self.config.get("localModels", [])
            if local:
                # Filter by capability if multimodal
                if multimodal:
                    local = [m for m in local if "vision" in m.get("capabilities", ["text"])]
                if local:
                    preferred = sorted(local, key=lambda x: x.get("priority", 99))[0]
                    logger.warning("Wakelock: no healthy GPUs, falling back to highest priority %s", preferred["name"])
                    return preferred["url"].rstrip("/") + request_path, False, None
            
            logger.warning("No localModels configured")
            return "", False, None

        # How long to poll locally before falling back to Venice on cold start.
        # Health probes run every 2s, so waiting 10-15s costs almost nothing
        # while avoiding an unnecessary cloud round-trip for a nearly-ready GPU.
        cold_wait = self.config.get("coldStartWaitSeconds", 15)

        # Phase 1: classify without holding self._lock (_classify_gpus acquires
        # its own lock internally for state mutations).
        ready_gpus, starting_gpus = self._classify_gpus(required_capabilities)

        if not ready_gpus and not starting_gpus:
            # Check if any GPUs are busy before assuming they are all stopped
            with self._lock:
                any_busy = any(g["state"] == "busy" for g in self.gpus.values()
                             if required_capabilities.issubset(set(g.get("capabilities", ["text"]))))
            
            if not any_busy:
                # All stopped — kick off the startup group, then re-classify.
                self._start_all()
                with self._lock:
                    for name, gpu in sorted(self.gpus.items(), key=lambda x: x[1]["priority"]):
                        gpu["state"] = "starting"
                        gpu["started_at"] = time.time()
                        gpu["last_request"] = time.time()
                logger.info("Wakelock: all GPUs starting")
                ready_gpus, starting_gpus = self._classify_gpus(required_capabilities)

        # Phase 2: pick a ready GPU (mutates busy/last_request, needs the lock).
        picked = n_ready = n_busy = None
        with self._lock:
            picked = self._pick_gpu(ready_gpus)
            if picked:
                n_ready = len(ready_gpus)
                n_busy = sum(1 for g in self.gpus.values() if g["busy"])
        if picked:
            name, gpu = picked
            logger.info(
                "Wakelock: routing to %s (ready=%d, busy=%d)", name, n_ready, n_busy,
            )
            return gpu["url"].rstrip("/") + request_path, False, name

        if starting_gpus:
            # GPUs are booting. Since the health probe fires every 2s we can
            # afford to wait up to coldStartWaitSeconds before paying the
            # Venice round-trip cost.
            poll = 2.0
            waited = 0.0
            while waited < cold_wait:
                time.sleep(poll)
                waited += poll
                ready_gpus, _ = self._classify_gpus(required_capabilities)
                picked = None
                with self._lock:
                    picked = self._pick_gpu(ready_gpus)
                if picked:
                    name, gpu = picked
                    logger.info(
                        "Wakelock: GPU %s ready after %.0fs cold-start wait — routing locally",
                        name, waited,
                    )
                    return gpu["url"].rstrip("/") + request_path, False, name

            if self._external_enabled():
                logger.info(
                    "Wakelock: GPUs still warming after %.0fs — routing to external fallback",
                    cold_wait,
                )
                return self._external_route(request_path)
            # External fallback disabled (or none configured): keep the request
            # local by falling through to the queue, which waits for a warming
            # GPU to become ready instead of touching the cloud.
            logger.info(
                "Wakelock: external fallback disabled — queuing for local GPU warmup",
            )

        # Queue overflow: per NK-09, offload to the cloud (Venice) to prevent
        # request timeouts. If external fallback is disabled, surface a clean
        # 503 to the client instead (server.py maps QUEUE_OVERFLOW -> 503).
        if self._check_queue_overflow():
            if self._external_enabled():
                logger.warning(
                    "Wakelock: queue overflow — offloading to external fallback",
                )
                return self._external_route(request_path)
            logger.warning("Wakelock: queue overflow, no external fallback — rejecting")
            raise RuntimeError("QUEUE_OVERFLOW")

        try:
            free_name = self._wait_for_gpu()
        except RuntimeError as e:
            if str(e) == "QUEUE_OVERFLOW":
                if self._external_enabled():
                    logger.warning(
                        "Wakelock: queue overflow during wait — offloading to external fallback",
                    )
                    return self._external_route(request_path)
                raise
            raise

        with self._lock:
            gpu = self.gpus[free_name]
            logger.info("Wakelock: dequeued → routing to %s", free_name)
            return gpu["url"].rstrip("/") + request_path, False, free_name

    def release(self, gpu_name: str) -> None:
        if not gpu_name or not self.enabled:
            return
        if gpu_name not in self.gpus:
            return
        # Free the slot and wake the queue FIRST — a slow/hung Docker API call
        # must never delay another request from grabbing this GPU.
        with self._lock:
            self.gpus[gpu_name]["busy"] = False
            self.gpus[gpu_name]["state"] = "ready"
        self._process_queue()
        # Docker I/O OUTSIDE the lock: verify the container is still running
        # and downgrade the state if it is gone (forwarding to a dead
        # container just fails fast with a 502; the health probe also
        # re-detects this within 2s).
        if not self._running(self.gpus[gpu_name]["dockerName"]):
            with self._lock:
                self.gpus[gpu_name]["state"] = "stopped"
            logger.warning(
                "Wakelock: %s container gone on release, marking stopped",
                gpu_name,
            )

    def _external_enabled(self) -> bool:
        """
        True only when external (Venice) fallback is both configured AND not
        explicitly disabled via externalFallbackEnabled. When False, cold-start
        and overflow paths stay local instead of routing to the cloud.
        """
        if not self.config.get("externalFallbackEnabled", True):
            return False
        return bool(self.config.get("externalModels", []))

    def _external_route(self, request_path: str) -> tuple:
        """
        Build the external (Venice) offload route tuple. Callers MUST check
        _external_enabled() first; this assumes at least one externalModel.
        """
        ext = self.config.get("externalModels", [])[0]
        return ext["url"].rstrip("/") + request_path, True, None

    def external_auth(self) -> str:
        if not self.enabled or not self._external_enabled():
            return ""
        external = self.config.get("externalModels", [])
        if external:
            key = os.environ.get(external[0].get("apiKeyEnv", ""), "")
            return f"Bearer {key}" if key else ""
        return ""

    def _record_request(self) -> None:
        now = time.time()
        self._last_request_at = now
        self._request_timestamps.append(now)
        cutoff = now - 600
        self._request_timestamps = [t for t in self._request_timestamps if t > cutoff]

    def _restart_stopped(self, name: str) -> None:
        """Restart a queued GPU whose container died, so the queue can drain."""
        gpu = self.gpus.get(name)
        if not gpu:
            return
        logger.info("Wakelock: queued GPU %s is stopped, attempting restart", name)
        if self._start(gpu["dockerName"]):
            with self._lock:
                gpu["state"] = "starting"
                gpu["started_at"] = time.time()

    def _wait_for_gpu(self) -> str:
        # DEADLOCK WARNING: _queue_event is a Condition built on _queue_lock,
        # which is NOT reentrant. _enqueue/_dequeue/_check_queue_overflow all
        # acquire _queue_lock, so they must NEVER be called while holding the
        # condition — that self-deadlocks and permanently wedges every request
        # behind "waiting for GPU". Only the wait() itself holds the condition.
        self._enqueue()
        while True:
            free_name = None
            restart_name = None
            with self._lock:
                free_name = next(
                    (
                        n
                        for n, g in self.gpus.items()
                        if not g["busy"] and g["state"] == "ready"
                    ),
                    None,
                )
                if free_name:
                    self.gpus[free_name]["busy"] = True
                    self.gpus[free_name]["last_request"] = time.time()
                    self.gpus[free_name]["state"] = "busy"
                else:
                    restart_name = next(
                        (
                            n
                            for n, g in self.gpus.items()
                            if not g["busy"] and g["state"] == "stopped"
                        ),
                        None,
                    )
            if free_name:
                self._dequeue()
                return free_name
            if restart_name:
                self._restart_stopped(restart_name)
            logger.info("Wakelock: request queued, waiting for GPU...")
            with self._queue_event:
                self._queue_event.wait(timeout=2)
            if self._check_queue_overflow():
                self._dequeue()
                raise RuntimeError("QUEUE_OVERFLOW")

    def _process_queue(self) -> None:
        with self._queue_event:
            self._queue_event.notify_all()

    @property
    def status(self) -> dict:
        if not self.enabled:
            # Health-aware status when auto-wake is disabled
            with self._lock:
                gpu_health = {
                    name: {
                        "healthy": self._health_cache.get(name, False),
                        "state": gpu["state"],
                    }
                    for name, gpu in self.gpus.items()
                }
            return {
                "gpu_busy": False,
                "active_requests": 0,
                "last_request_at": (
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._last_request_at))
                    if self._last_request_at
                    else None
                ),
                "wakelock_enabled": False,
                "health_checking_enabled": True,
                "gpus": gpu_health,
            }
        # Read queue stats under _queue_lock first, then GPU stats under _lock
        # to avoid ABBA deadlock with _wait_for_gpu() which acquires _queue_lock
        # then _lock in the opposite order.
        with self._queue_lock:
            queue_len = len(self._queue_entries)
            oldest_wait = (
                time.time() - self._queue_entries[0] if self._queue_entries else 0
            )
        with self._lock:
            return {
                "gpu_busy": any(g["busy"] for g in self.gpus.values()),
                "active_requests": sum(1 for g in self.gpus.values() if g["busy"]),
                "last_request_at": (
                    time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._last_request_at)
                    )
                    if self._last_request_at
                    else None
                ),
                "wakelock_enabled": True,
                "gpus": {
                    n: {
                        "state": g["state"],
                        "busy": g["busy"],
                        "healthy": self._health_cache.get(n, False),
                    }
                    for n, g in self.gpus.items()
                },
                "queue_length": queue_len,
                "queue_oldest_wait_sec": round(oldest_wait, 1),
                "queue_overflow_active": self._overflow_active,
            }
