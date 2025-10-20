"""
cache.py — 공용 LRU+TTL 캐시
- 기본 정책: LRU + TTL
- 기본 크기: 64개 (ENV: CACHE_MAX_ITEMS)
- 기본 TTL: 900초 = 15분 (ENV: CACHE_TTL_SEC)
- 스레드 세이프
"""

import os
import time
from collections import OrderedDict
from threading import RLock
from typing import Any, Optional, Tuple, Dict, Iterable, Callable

# ===== 환경설정 (config.py가 없어도 동작하도록 ENV 우선) =====
DEFAULT_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "64"))
DEFAULT_TTL_SEC  = int(os.getenv("CACHE_TTL_SEC",  "900"))  # 15분


class _LRUTTLCache:
    """내부용: LRU + TTL 구현체 (OrderedDict 사용)"""
    def __init__(self, max_items: int = DEFAULT_MAX_ITEMS, ttl_sec: int = DEFAULT_TTL_SEC):
        self.max_items = int(max_items)
        self.ttl_sec   = int(ttl_sec)
        self._lock = RLock()
        # key -> (value, set_time)
        self._store: "OrderedDict[Any, Tuple[Any, float]]" = OrderedDict()

    # --- 내부 유틸 ---
    def _expired(self, set_time: float, ttl_override_sec: Optional[int] = None) -> bool:
        """ttl_override_sec가 주어지면 더 짧은 TTL을 우선 적용"""
        eff_ttl = self.ttl_sec
        if ttl_override_sec is not None and ttl_override_sec > 0:
            eff_ttl = min(eff_ttl if eff_ttl > 0 else ttl_override_sec, ttl_override_sec)
        if eff_ttl <= 0:
            return False
        return (time.time() - set_time) >= eff_ttl

    def _evict_if_needed(self) -> int:
        """용량 초과 시 LRU 순서대로 제거"""
        evicted = 0
        while self.max_items > 0 and len(self._store) > self.max_items:
            self._store.popitem(last=False)  # LRU 제거
            evicted += 1
        return evicted

    # --- 공개 API ---
    def get(self, key: Any, ttl_override_sec: Optional[int] = None) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            value, set_time = item
            if self._expired(set_time, ttl_override_sec):
                try:
                    del self._store[key]
                except KeyError:
                    pass
                return None
            # LRU 갱신
            self._store.move_to_end(key, last=True)
            return value

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            self._store[key] = (value, time.time())
            self._store.move_to_end(key, last=True)
            self._evict_if_needed()

    def delete(self, key: Any) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def prune(self) -> Dict[str, int]:
        """만료/용량 정리 수행. returns: {"expired": n1, "lru": n2}"""
        with self._lock:
            now = time.time()
            expired_keys: Iterable[Any] = (k for k, (_, t) in list(self._store.items())
                                           if self.ttl_sec > 0 and (now - t) >= self.ttl_sec)
            expired_cnt = 0
            for k in list(expired_keys):
                if k in self._store:
                    del self._store[k]
                    expired_cnt += 1
            lru_evict = self._evict_if_needed()
        return {"expired": expired_cnt, "lru": lru_evict}

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "size": len(self._store),
                "max_items": self.max_items,
                "ttl_sec": self.ttl_sec,
            }


# ===== 싱글톤 매니저: 프로젝트 전역에서 import 하여 사용 =====
class CacheManager:
    """
    전역 캐시 매니저
      - get(key, ttl_override_sec=None), set(key, value), delete(key), clear(), prune(), stats()
      - TTL/크기는 환경변수 또는 set_policy()로 조정 가능
    """
    _instance: Optional[_LRUTTLCache] = None
    _lock = RLock()

    @classmethod
    def _ensure(cls) -> _LRUTTLCache:
        with cls._lock:
            if cls._instance is None:
                cls._instance = _LRUTTLCache()
            return cls._instance

    # ---- 기본 연산 ----
    @classmethod
    def get(cls, key: Any, ttl_override_sec: Optional[int] = None) -> Optional[Any]:
        return cls._ensure().get(key, ttl_override_sec)

    @classmethod
    def set(cls, key: Any, value: Any) -> None:
        cls._ensure().set(key, value)

    @classmethod
    def delete(cls, key: Any) -> bool:
        return cls._ensure().delete(key)

    @classmethod
    def clear(cls) -> None:
        cls._ensure().clear()

    @classmethod
    def prune(cls) -> Dict[str, int]:
        return cls._ensure().prune()

    @classmethod
    def stats(cls) -> Dict[str, int]:
        return cls._ensure().stats()

    @classmethod
    def set_policy(cls, *, max_items: Optional[int] = None, ttl_sec: Optional[int] = None) -> None:
        """정책 변경 (필요 시)"""
        with cls._lock:
            inst = cls._ensure()
            if max_items is not None:
                inst.max_items = int(max_items)
            if ttl_sec is not None:
                inst.ttl_sec = int(ttl_sec)


# ===== 편의: 메모이즈 데코레이터 (옵션) =====
def memoize_ttl(key_func: Optional[Callable[..., Any]] = None):
    """
    사용법:
        @memoize_ttl()
        def fetch(x): ...
    또는
        @memoize_ttl(lambda a,b: f"{a}-{b}")
    """
    def decorator(func: Callable[..., Any]):
        def wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs) if key_func else (func.__name__, args, tuple(sorted(kwargs.items())))
            cached = CacheManager.get(key)
            if cached is not None:
                return cached
            value = func(*args, **kwargs)
            CacheManager.set(key, value)
            return value
        return wrapper
    return decorator


# ===== 백워드 호환 별칭(옵션) =====
# 향후 모듈 수정 시 CacheManager를 직접 사용하되,
# 과거 코드가 참조할 수 있는 별칭을 제공한다.
feature_cache = CacheManager
global_cache = CacheManager
_feature_cache = CacheManager
