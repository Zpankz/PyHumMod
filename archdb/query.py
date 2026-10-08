"""Python query API over the PyHumMod architecture database.

    from archdb import open_db
    db = open_db()                       # builds build/hummod_arch.db if missing or stale
    db.variable("ADHPool.conc_ADH")      # role, defining equations, readers
    db.upstream("ADHPool.conc_ADH", 2)   # what it depends on, by distance
    db.downstream("BetaBlockade.Block_percent")  # what a change to it entails
    db.path("ADHPool.conc_ADH", "CD_H2O.Outflow")
    db.sql("SELECT * FROM state_variables LIMIT 5")
"""

import bisect
import hashlib
import json
import os
import sqlite3
from collections import defaultdict, deque

from . import extract


def _rows(cursor):
    cols = [c[0] for c in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def _source_sha(source):
    with open(source, encoding="utf-8") as f:
        return hashlib.sha256(f.read().encode("utf-8")).hexdigest()


def open_db(path=extract.DEFAULT_DB, source=extract.DEFAULT_SOURCE, rebuild=False):
    """Open the database, (re)building it when missing, stale, or from an older extractor."""
    stale = rebuild or not os.path.exists(path)
    if not stale:
        con = sqlite3.connect(path)
        try:
            meta = dict(con.execute("SELECT key, value FROM meta"))
        except sqlite3.DatabaseError:
            meta = {}
        con.close()
        stale = (meta.get("extractor_version") != extract.EXTRACTOR_VERSION
                 or (os.path.exists(source) and meta.get("source_sha256") != _source_sha(source)))
    if stale:
        extract.build(source, path)
    return ArchDB(path)


class ArchDB:
    def __init__(self, path=extract.DEFAULT_DB):
        self.db_path = path
        self.con = sqlite3.connect(path)
        self._fwd = None
        self._rev = None

    # ------------------------------------------------------------- basics

    def sql(self, query, params=()):
        return _rows(self.con.execute(query, params))

    def stats(self):
        out = dict(self.con.execute("SELECT key, value FROM meta"))
        out["roles"] = dict(self.con.execute(
            "SELECT role, COUNT(*) FROM variables GROUP BY role ORDER BY 2 DESC"))
        out["equation_kinds"] = dict(self.con.execute(
            "SELECT kind, COUNT(*) FROM equations GROUP BY kind ORDER BY 2 DESC"))
        out["issue_kinds"] = dict(self.con.execute(
            "SELECT kind, COUNT(*) FROM issues GROUP BY kind ORDER BY 2 DESC"))
        out["phases"] = dict(self.con.execute(
            "SELECT phase, COUNT(*) FROM execution WHERE item='equation' GROUP BY phase ORDER BY MIN(ord)"))
        return out

    def resolve(self, name):
        """Exact qualname for a variable, function or module; else candidates matching the name."""
        for table, col in (("variables", "qualname"), ("functions", "qualname"), ("modules", "name")):
            if self.con.execute("SELECT 1 FROM %s WHERE %s = ?" % (table, col), (name,)).fetchone():
                return table[:-1] if table != "modules" else "module", name
        return None, self.search(name, limit=20)

    def search(self, pattern, limit=50):
        """Substring match on names; `*` is the only wildcard (`_` and `%` match literally)."""
        escaped = pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = "%" + escaped.replace("*", "%") + "%"
        rows = self.sql(
            "SELECT 'module' AS type, name AS qualname FROM modules WHERE name LIKE ? ESCAPE '\\' "
            "UNION ALL SELECT 'function', qualname FROM functions WHERE qualname LIKE ? ESCAPE '\\' "
            "UNION ALL SELECT 'variable', qualname FROM variables "
            "WHERE qualname LIKE ? ESCAPE '\\' AND scope='attribute' "
            "LIMIT ?", (like, like, like, limit))
        return rows

    # ------------------------------------------------------------ entities

    def module(self, name):
        mod = self.sql("SELECT * FROM modules WHERE name = ?", (name,))
        if not mod:
            return None
        out = mod[0]
        out["functions"] = self.sql(
            "SELECT qualname, kind, phase, scheduled, line_start FROM functions WHERE module = ? ORDER BY line_start",
            (name,))
        out["variables"] = self.sql(
            "SELECT qualname, role, dtype, initial_expr, reader_count FROM variables "
            "WHERE module = ? AND scope='attribute' ORDER BY name", (name,))
        out["called_by"] = self.sql(
            "SELECT DISTINCT caller FROM calls WHERE callee LIKE ? ORDER BY caller", (name + ".%",))
        return out

    def variable(self, qualname):
        var = self.sql("SELECT * FROM variables WHERE qualname = ?", (qualname,))
        if not var:
            return None
        out = var[0]
        out["defined_by"] = self.sql(
            "SELECT id, function, kind, expression, derivative, condition, line, first_exec "
            "FROM equations WHERE target = ? ORDER BY first_exec IS NULL, first_exec, line", (qualname,))
        for eq in out["defined_by"]:
            eq["inputs"] = self.sql(
                "SELECT variable, via, lagged FROM equation_inputs WHERE equation_id = ? ORDER BY variable",
                (eq["id"],))
        out["read_by"] = self.sql(
            "SELECT DISTINCT e.target, e.function, i.via, i.lagged FROM equation_inputs i "
            "JOIN equations e ON e.id = i.equation_id WHERE i.variable = ? AND e.target <> ? ORDER BY e.target",
            (qualname, qualname))
        return out

    def function(self, qualname):
        fn = self.sql("SELECT * FROM functions WHERE qualname = ?", (qualname,))
        if not fn:
            return None
        out = fn[0]
        out["equations"] = self.sql(
            "SELECT id, seq, target, kind, expression, condition, line FROM equations "
            "WHERE function = ? ORDER BY seq", (qualname,))
        out["calls"] = self.sql(
            "SELECT callee, seq, condition, line FROM calls WHERE caller = ? ORDER BY seq", (qualname,))
        out["called_by"] = self.sql(
            "SELECT caller, condition, line FROM calls WHERE callee = ? ORDER BY caller", (qualname,))
        if out["kind"] == "curve":
            out["curve"] = self.curve(qualname)
        return out

    def curve(self, qualname):
        rows = self.sql("SELECT * FROM curves WHERE function = ?", (qualname,))
        if not rows:
            return None
        c = rows[0]
        for k in ("x_json", "y_json", "slope_json"):
            c[k[:-5]] = json.loads(c.pop(k))
        c["used_by"] = self.sql(
            "SELECT e.target, u.argument FROM curve_uses u JOIN equations e ON e.id = u.equation_id "
            "WHERE u.curve = ?", (qualname,))
        return c

    def eval_curve(self, qualname, x):
        """Evaluate a HumMod curve the way special_functions.cubic_hermite_spline does (clamped ends)."""
        c = self.curve(qualname)
        xs, ys, ms = c["x"], c["y"], c["slope"]
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = bisect.bisect_right(xs, x) - 1
        h = xs[i + 1] - xs[i]
        t = (x - xs[i]) / h
        h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
        h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        return h00 * ys[i] + h10 * h * ms[i] + h01 * ys[i + 1] + h11 * h * ms[i + 1]

    # --------------------------------------------------------- dependencies

    CONDITION_VIAS = ("condition", "call_condition")

    def _graph(self):
        """Adjacency maps; each edge keeps every (via, lagged) kind it occurs with."""
        if self._fwd is None:
            fwd, rev = defaultdict(dict), defaultdict(dict)
            for src, dst, via, lagged in self.con.execute(
                    "SELECT i.variable, e.target, i.via, i.lagged FROM equation_inputs i "
                    "JOIN equations e ON e.id = i.equation_id WHERE i.via <> 'integrator_state'"):
                if src == dst:
                    continue
                kinds = fwd[src].setdefault(dst, set())
                kinds.add((via, lagged))
                rev[dst][src] = kinds
            self._fwd, self._rev = fwd, rev
        return self._fwd, self._rev

    def _edge_allowed(self, kinds, include_conditions, include_lagged):
        return any((include_conditions or via not in self.CONDITION_VIAS) and (include_lagged or not lagged)
                   for via, lagged in kinds)

    def _bfs(self, graph, start, max_depth, include_conditions, include_lagged):
        seen = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            depth = seen[node]
            if max_depth is not None and depth >= max_depth:
                continue
            for nxt, kinds in graph.get(node, {}).items():
                if nxt in seen or not self._edge_allowed(kinds, include_conditions, include_lagged):
                    continue
                seen[nxt] = depth + 1
                queue.append(nxt)
        seen.pop(start)
        return seen

    def _annotate(self, found):
        roles = dict(self.con.execute("SELECT qualname, role FROM variables"))
        return [dict(variable=v, depth=d, role=roles.get(v)) for v, d in
                sorted(found.items(), key=lambda kv: (kv[1], kv[0]))]

    def inputs(self, qualname):
        """Direct dependencies: variables read to compute this one."""
        return self.upstream(qualname, max_depth=1)

    def outputs(self, qualname):
        """Direct dependents: variables computed from this one."""
        return self.downstream(qualname, max_depth=1)

    def upstream(self, qualname, max_depth=None, include_conditions=True, include_lagged=True):
        """Everything this variable depends on (transitively), with shortest distance."""
        _, rev = self._graph()
        return self._annotate(self._bfs(rev, qualname, max_depth, include_conditions, include_lagged))

    def downstream(self, qualname, max_depth=None, include_conditions=True, include_lagged=True):
        """Everything this variable entails: all variables whose value it can change."""
        fwd, _ = self._graph()
        return self._annotate(self._bfs(fwd, qualname, max_depth, include_conditions, include_lagged))

    def entailments(self, qualname, max_depth=None):
        """Summary of what a change to `qualname` propagates to."""
        down = self.downstream(qualname, max_depth)
        modules = defaultdict(int)
        for row in down:
            modules[row["variable"].split(".")[0]] += 1
        states = [r["variable"] for r in down if r["role"] == "state"]
        same_step = self.downstream(qualname, max_depth, include_lagged=False)
        return dict(variable=qualname, affected=len(down), affected_same_step=len(same_step),
                    affected_states=len(states), states=states,
                    modules=dict(sorted(modules.items(), key=lambda kv: -kv[1])),
                    max_depth=max((r["depth"] for r in down), default=0))

    def path(self, src, dst, include_conditions=True):
        """Shortest dependency chain src -> ... -> dst, or None."""
        fwd, _ = self._graph()
        prev = {src: None}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                chain = []
                while node is not None:
                    chain.append(node)
                    node = prev[node]
                return chain[::-1]
            for nxt, kinds in fwd.get(node, {}).items():
                if nxt not in prev and self._edge_allowed(kinds, include_conditions, True):
                    prev[nxt] = node
                    queue.append(nxt)
        return None

    # ------------------------------------------------------------ call graph

    def callees(self, qualname, transitive=False):
        if not transitive:
            return self.sql("SELECT callee, condition, line FROM calls WHERE caller = ? ORDER BY seq", (qualname,))
        return self.sql(
            "WITH RECURSIVE r(fn, depth) AS (SELECT ?, 0 UNION "
            "SELECT c.callee, r.depth + 1 FROM calls c JOIN r ON c.caller = r.fn WHERE r.depth < 50) "
            "SELECT fn AS function, MIN(depth) AS depth FROM r WHERE depth > 0 GROUP BY fn ORDER BY 2, 1",
            (qualname,))

    def callers(self, qualname, transitive=False):
        if not transitive:
            return self.sql("SELECT caller, condition, line FROM calls WHERE callee = ? ORDER BY caller", (qualname,))
        return self.sql(
            "WITH RECURSIVE r(fn, depth) AS (SELECT ?, 0 UNION "
            "SELECT c.caller, r.depth + 1 FROM calls c JOIN r ON c.callee = r.fn WHERE r.depth < 50) "
            "SELECT fn AS function, MIN(depth) AS depth FROM r WHERE depth > 0 GROUP BY fn ORDER BY 2, 1",
            (qualname,))

    def schedule(self, phase=None, function=None, equations=True, limit=None):
        """The flattened execution order of one hummod.step()."""
        where, params = [], []
        if phase:
            where.append("x.phase = ?")
            params.append(phase)
        if function:
            where.append("x.function = ?")
            params.append(function)
        if not equations:
            where.append("x.item = 'call'")
        q = ("SELECT x.ord, x.phase, x.depth, x.item, x.function, e.target, e.kind, x.condition "
             "FROM execution x LEFT JOIN equations e ON e.id = x.equation_id")
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY x.ord"
        if limit:
            q += " LIMIT %d" % int(limit)
        return self.sql(q, params)

    def loops(self, kind="feedback", limit=20):
        """Loops of the given kind: 'feedback' (all dependencies) or 'algebraic' (same-step only)."""
        return self.sql("SELECT * FROM loops WHERE kind = ? ORDER BY scc_id LIMIT ?", (kind, limit))

    def loop_of(self, qualname, kind="feedback"):
        column = {"feedback": "scc_id", "algebraic": "algebraic_scc_id"}[kind]
        row = self.con.execute("SELECT %s FROM variables WHERE qualname = ?" % column, (qualname,)).fetchone()
        if not row or row[0] is None:
            return None
        return dict(kind=kind, scc_id=row[0], members=[r[0] for r in self.con.execute(
            "SELECT qualname FROM variables WHERE %s = ? ORDER BY qualname" % column, (row[0],))])

    def issues(self, kind=None, limit=200):
        if kind:
            return self.sql("SELECT * FROM issues WHERE kind = ? ORDER BY id LIMIT ?", (kind, limit))
        return self.sql("SELECT * FROM issues ORDER BY id LIMIT ?", (limit,))
