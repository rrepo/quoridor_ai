# ai/cache.py

_DIST_CACHE: dict = {}
_route_cache: dict = {}

def get_dist(key):
    return _DIST_CACHE.get(key)

def set_dist(key, my_dist, enemy_dist):
    _DIST_CACHE[key] = (my_dist, enemy_dist)

def get_route(key, turn):
    return _route_cache.get((key, turn))

def set_route(key, turn, route):
    _route_cache[(key, turn)] = route

def clear_dist_cache():
    _DIST_CACHE.clear()
    _route_cache.clear()