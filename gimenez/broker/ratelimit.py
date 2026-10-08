"""Token-bucket rate limiter. TradeLocker publishes per-route limits; we use a fraction of them,
so the live trader never gets throttled and history downloads never crowd out trading calls."""
from __future__ import annotations

import threading
import time


class TokenBucket:
    def __init__(self, rate_per_second: float, burst: int = 2, clock=time.monotonic, sleep=time.sleep) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate_per_second
        self.burst = max(1, burst)
        self.tokens = float(self.burst)
        self.clock, self.sleep = clock, sleep
        self.last = clock()
        self.lock = threading.Lock()
        self.waited = 0.0

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
        self.last = now

    def acquire(self) -> None:
        with self.lock:
            while True:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
                self.waited += wait
                self.sleep(wait)

    def set_rate(self, rate_per_second: float) -> None:
        with self.lock:
            self._refill()
            self.rate = max(0.01, rate_per_second)
