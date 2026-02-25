#!/usr/bin/env python3
"""2x2 Rubik's Cube solver in a single Python file.

Usage:
    python rubiks2x2_solver.py "R U R' F2"

Moves supported in scrambles and solutions:
    U D L R F B with optional suffix 2 or '
"""

from __future__ import annotations

import sys
from collections import deque
from math import factorial

# Corner ordering:
# 0: URF, 1: UFL, 2: ULB, 3: UBR, 4: DFR, 5: DLF, 6: DBL, 7: DRB

BASE_MOVES = {
    "U": {
        "cp": [3, 0, 1, 2, 4, 5, 6, 7],
        "co": [0, 0, 0, 0, 0, 0, 0, 0],
    },
    "R": {
        "cp": [4, 1, 2, 0, 7, 5, 6, 3],
        "co": [2, 0, 0, 1, 1, 0, 0, 2],
    },
    "F": {
        "cp": [1, 5, 2, 3, 0, 4, 6, 7],
        "co": [1, 2, 0, 0, 2, 1, 0, 0],
    },
    "D": {
        "cp": [0, 1, 2, 3, 5, 6, 7, 4],
        "co": [0, 0, 0, 0, 0, 0, 0, 0],
    },
    "L": {
        "cp": [0, 2, 6, 3, 4, 1, 5, 7],
        "co": [0, 1, 2, 0, 0, 2, 1, 0],
    },
    "B": {
        "cp": [0, 1, 3, 7, 4, 5, 2, 6],
        "co": [0, 0, 1, 2, 0, 0, 2, 1],
    },
}

MOVE_NAMES = [
    "U", "U2", "U'",
    "R", "R2", "R'",
    "F", "F2", "F'",
    "D", "D2", "D'",
    "L", "L2", "L'",
    "B", "B2", "B'",
]


def apply_move(cp, co, mv):
    ncp = [cp[mv["cp"][i]] for i in range(8)]
    nco = [(co[mv["cp"][i]] + mv["co"][i]) % 3 for i in range(8)]
    return ncp, nco


def compose_moves(a, b):
    # Apply a then b
    cp = [a["cp"][b["cp"][i]] for i in range(8)]
    co = [(a["co"][b["cp"][i]] + b["co"][i]) % 3 for i in range(8)]
    return {"cp": cp, "co": co}


def build_moves():
    moves = []
    for face in ("U", "R", "F", "D", "L", "B"):
        m1 = BASE_MOVES[face]
        m2 = compose_moves(m1, m1)
        m3 = compose_moves(m2, m1)
        moves.extend([m1, m2, m3])
    return moves


MOVES = build_moves()


def ori_to_index(co):
    idx = 0
    for i in range(7):
        idx = idx * 3 + co[i]
    return idx


def index_to_ori(idx):
    co = [0] * 8
    s = 0
    for i in range(6, -1, -1):
        co[i] = idx % 3
        s += co[i]
        idx //= 3
    co[7] = (-s) % 3
    return co


def perm_to_index(cp):
    idx = 0
    vals = list(cp)
    for i in range(8):
        smaller = 0
        for j in range(i + 1, 8):
            if vals[j] < vals[i]:
                smaller += 1
        idx += smaller * factorial(7 - i)
    return idx


def index_to_perm(idx):
    elems = list(range(8))
    cp = [0] * 8
    for i in range(8):
        f = factorial(7 - i)
        pos = idx // f
        idx %= f
        cp[i] = elems.pop(pos)
    return cp


def build_orientation_dist():
    dist = [-1] * (3 ** 7)
    start = ori_to_index([0] * 8)
    dist[start] = 0
    q = deque([start])

    while q:
        cur = q.popleft()
        d = dist[cur]
        co = index_to_ori(cur)
        cp = list(range(8))
        for mv in MOVES:
            _, nco = apply_move(cp, co, mv)
            ni = ori_to_index(nco)
            if dist[ni] == -1:
                dist[ni] = d + 1
                q.append(ni)
    return dist


def build_permutation_dist():
    dist = [-1] * factorial(8)
    start = perm_to_index(list(range(8)))
    dist[start] = 0
    q = deque([start])
    zero_co = [0] * 8

    while q:
        cur = q.popleft()
        d = dist[cur]
        cp = index_to_perm(cur)
        for mv in MOVES:
            ncp, _ = apply_move(cp, zero_co, mv)
            ni = perm_to_index(ncp)
            if dist[ni] == -1:
                dist[ni] = d + 1
                q.append(ni)
    return dist


def parse_scramble(scramble):
    if not scramble.strip():
        return []
    tokens = scramble.split()
    result = []
    valid_faces = "URFDLB"
    for tok in tokens:
        if tok[0] not in valid_faces:
            raise ValueError(f"Invalid move: {tok}")
        suffix = tok[1:]
        if suffix not in ("", "2", "'"):
            raise ValueError(f"Invalid move suffix: {tok}")
        base = "URFDLB".index(tok[0]) * 3
        if suffix == "":
            result.append(base)
        elif suffix == "2":
            result.append(base + 1)
        else:
            result.append(base + 2)
    return result


def heuristic(cp, co, ori_dist, perm_dist):
    return max(ori_dist[ori_to_index(co)], perm_dist[perm_to_index(cp)])


def ida_search(cp, co, g, bound, last_face, path, ori_dist, perm_dist):
    h = heuristic(cp, co, ori_dist, perm_dist)
    f = g + h
    if f > bound:
        return f
    if cp == list(range(8)) and co == [0] * 8:
        return True

    min_next = 1_000_000
    for mi, mv in enumerate(MOVES):
        face = mi // 3
        if face == last_face:
            continue
        ncp, nco = apply_move(cp, co, mv)
        path.append(mi)
        res = ida_search(ncp, nco, g + 1, bound, face, path, ori_dist, perm_dist)
        if res is True:
            return True
        if res < min_next:
            min_next = res
        path.pop()
    return min_next


def solve(scramble):
    cp = list(range(8))
    co = [0] * 8
    for mi in parse_scramble(scramble):
        cp, co = apply_move(cp, co, MOVES[mi])

    ori_dist = build_orientation_dist()
    perm_dist = build_permutation_dist()

    bound = heuristic(cp, co, ori_dist, perm_dist)
    path = []
    while True:
        res = ida_search(cp, co, 0, bound, -1, path, ori_dist, perm_dist)
        if res is True:
            return " ".join(MOVE_NAMES[m] for m in path)
        bound = res


def main():
    if len(sys.argv) < 2:
        print("Usage: python rubiks2x2_solver.py \"R U R' F2\"")
        sys.exit(1)

    scramble = " ".join(sys.argv[1:])
    try:
        solution = solve(scramble)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)

    print(solution if solution else "Cube is already solved.")


if __name__ == "__main__":
    main()
