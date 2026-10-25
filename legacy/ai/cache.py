# ai/cache.py

_DIST_CACHE: dict = {}
_route_cache: dict = {}

# 1 局を通してクリアされない使い方（game.py の ai_move 等）でも
# メモリが際限なく増えないよう、上限を超えたら丸ごと捨てる。
_MAX_ENTRIES = 500_000

def get_dist(key):
    return _DIST_CACHE.get(key)

def set_dist(key, my_dist, enemy_dist):
    if len(_DIST_CACHE) >= _MAX_ENTRIES:
        _DIST_CACHE.clear()
    _DIST_CACHE[key] = (my_dist, enemy_dist)

def get_route(key, turn):
    return _route_cache.get((key, turn))

def set_route(key, turn, route):
    if len(_route_cache) >= _MAX_ENTRIES:
        _route_cache.clear()
    _route_cache[(key, turn)] = route

def clear_dist_cache():
    _DIST_CACHE.clear()
    _route_cache.clear()