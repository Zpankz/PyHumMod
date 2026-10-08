"""Build the PyHumMod architecture database from a static parse of src/hummod.py.

The extractor never imports or runs the model; it reads the source with
Python's ``ast`` module, so it works without scipy and on a model that does not
currently simulate correctly. Running it twice on the same source produces the
same database contents.

Usage:
    python -m archdb build [--source src/hummod.py] [--out build/hummod_arch.db]
"""

import ast
import hashlib
import json
import os
import re
import sqlite3
from collections import defaultdict

EXTRACTOR_VERSION = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_SOURCE = os.path.join(REPO_ROOT, "src", "hummod.py")
DEFAULT_DB = os.path.join(REPO_ROOT, "build", "hummod_arch.db")
SCHEMA = os.path.join(HERE, "schema.sql")

RUNTIME_MODULE = "hummod"  # pseudo-module holding the module-level step() function
RUNTIME_CLASSES = {"System", "Timer"}
IGNORED_NAMES = {"math", "np", "timervars"}


def _qual(*parts):
    return ".".join(parts)


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _dtype(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Timer":
        return "timer"
    value = _literal(node)
    if value is None:
        return "none" if isinstance(node, ast.Constant) else "expr"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return "expr"


class Extractor:
    def __init__(self, source_path):
        self.source_path = source_path
        with open(source_path, encoding="utf-8") as f:
            self.source = f.read()
        self.lines = self.source.splitlines()
        self.tree = ast.parse(self.source)
        # Python keeps the last definition of a repeated class name; earlier ones are dead code.
        self.classes = {n.name: n for n in self.tree.body if isinstance(n, ast.ClassDef)}
        self.shadowed = [n for n in self.tree.body
                         if isinstance(n, ast.ClassDef) and self.classes[n.name] is not n]
        self.instantiated = {
            n.targets[0].id
            for n in self.tree.body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == n.targets[0].id
        }

        self.modules = []
        self.functions = {}        # qualname -> row dict
        self.declared = {}         # var qualname -> row dict (from __init__)
        self.locals_seen = {}      # var qualname -> (module, line)
        self.equations = []        # row dicts, id = index + 1
        self.inputs = {}           # (eq_id, var, via) -> lagged
        self.curves = []
        self.curve_uses = []
        self.calls = []
        self.items = defaultdict(list)  # function qualname -> [(seq, 'eq'|'call', id)]
        self.issues = []
        self.integrator = {}       # state variable qualname -> integrator function name
        self.pending_residual = {} # nested implicit function qualname -> (node, ctx, conds, cond_reads)

    # ------------------------------------------------------------------ parse

    def run(self):
        for cls in self.tree.body:
            if isinstance(cls, ast.ClassDef) and self.classes[cls.name] is cls:
                self._module(cls)
        for cls in self.shadowed:
            live = self.classes[cls.name]
            same = ast.dump(cls) == ast.dump(live)
            self.issues.append(("duplicate_class", cls.name,
                                "shadowed by the definition at line %d%s"
                                % (live.lineno, " (identical body)" if same else " (bodies differ)"),
                                cls.lineno))
        step = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "step")
        self.modules.append(dict(
            name=RUNTIME_MODULE, subsystem=RUNTIME_MODULE, kind="runtime", instantiated=0,
            line_start=step.lineno, line_end=step.end_lineno,
            doc="Module-level step() driver of src/hummod.py"))
        self._function(RUNTIME_MODULE, step, kind="runtime", phase="step")
        self._schedule()
        self._resolve_variables()
        self._loops()

    def _module(self, cls):
        kind = "structure" if cls.name == "Structure" else ("runtime" if cls.name in RUNTIME_CLASSES else "model")
        self.modules.append(dict(
            name=cls.name, subsystem=cls.name.split("_")[0], kind=kind,
            instantiated=int(cls.name in self.instantiated),
            line_start=cls.lineno, line_end=cls.end_lineno, doc=ast.get_docstring(cls)))
        for node in cls.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name == "__init__":
                self._init(cls.name, node)
            elif node.name.endswith("_curve"):
                self._curve(cls.name, node)
            elif node.name.endswith("_func"):
                self._function(cls.name, node, kind="block", phase=node.name[: -len("_func")])
            else:
                self._function(cls.name, node, kind="runtime", phase=None)

    def _add_function(self, module, name, qualname, kind, phase, node):
        self.functions[qualname] = dict(
            module=module, name=name, qualname=qualname, kind=kind, phase=phase, scheduled=0,
            line_start=node.lineno, line_end=node.end_lineno,
            source="\n".join(self.lines[node.lineno - 1: node.end_lineno]))

    def _init(self, module, node):
        qualname = _qual(module, "__init__")
        self._add_function(module, "__init__", qualname, "init", None, node)
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                target = stmt.targets[0]
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    value = _literal(stmt.value)
                    self.declared[_qual(module, target.attr)] = dict(
                        initial_expr=ast.unparse(stmt.value), dtype=_dtype(stmt.value),
                        initial_value=float(value) if isinstance(value, (int, float)) else None,
                        line=stmt.lineno)

    def _curve(self, module, node):
        qualname = _qual(module, node.name)
        self._add_function(module, node.name, qualname, "curve", None, node)
        call = node.body[0].value
        xs, ys, slopes = (_literal(a) for a in call.args[1:4])
        self.curves.append(dict(
            function=qualname, module=module, name=node.name[: -len("_curve")], n_points=len(xs),
            x_json=json.dumps(xs), y_json=json.dumps(ys), slope_json=json.dumps(slopes),
            x_min=min(xs), x_max=max(xs), y_min=min(ys), y_max=max(ys)))

    def _function(self, module, node, kind, phase):
        qualname = _qual(module, node.name)
        self._add_function(module, node.name, qualname, kind, phase, node)
        local_names = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                local_names.add(sub.id)
            elif isinstance(sub, ast.FunctionDef) and sub is not node:
                local_names.add(sub.name)
                local_names.update(a.arg for a in sub.args.args)
        ctx = dict(module=module, method=qualname, locals=local_names, seq=[0])
        self._walk(node.body, ctx, qualname, [])

    # ------------------------------------------------------------ statements

    def _next_seq(self, ctx):
        ctx["seq"][0] += 1
        return ctx["seq"][0]

    def _local(self, ctx, name, line):
        qualname = _qual(ctx["method"], name)
        self.locals_seen.setdefault(qualname, (ctx["module"], line))
        return qualname

    def _reads(self, expr, ctx):
        """Variable qualnames read by an expression, and curve calls inside it."""
        found, curves = [], []
        called = {id(n.func) for n in ast.walk(expr) if isinstance(n, ast.Call)}
        for node in ast.walk(expr):
            if (id(node) in called and isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and not node.attr.endswith(("_func", "_curve")) and node.value.id not in IGNORED_NAMES):
                # e.g. self.float(x): a method the converter emitted but never defined
                self.issues.append(("missing_method", ast.unparse(node), ctx["method"], node.lineno))
                continue
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                base = node.value.id
                if base == "self":
                    owner = ctx["module"]
                elif base in self.classes:
                    owner = base
                else:
                    continue
                if node.attr.endswith("_func"):
                    continue
                if node.attr.endswith("_curve"):
                    continue
                found.append(_qual(owner, node.attr))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in ctx["locals"]:
                found.append(self._local(ctx, node.id, node.lineno))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr.endswith("_curve"):
                base = node.func.value.id if isinstance(node.func.value, ast.Name) else None
                owner = ctx["module"] if base == "self" else base
                curves.append((_qual(owner, node.func.attr), ast.unparse(node.args[0]) if node.args else ""))
        return list(dict.fromkeys(found)), curves

    @staticmethod
    def _qualify(text, ctx):
        """Rewrite self.X as Module.X so an expression reads the same outside its class."""
        return None if text is None else re.sub(r"\bself\.", ctx["module"] + ".", text)

    def _target(self, node, ctx):
        """Resolve an assignment target to a variable qualname."""
        while isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            node = node.value  # self.Timer.val -> self.Timer
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            base = node.value.id
            owner = ctx["module"] if base == "self" else base
            if owner not in self.classes:
                self.issues.append(("unknown_module", ast.unparse(node), ctx["method"], node.lineno))
            return _qual(owner, node.attr)
        if isinstance(node, ast.Name):
            return self._local(ctx, node.id, node.lineno)
        self.issues.append(("unsupported_target", ast.unparse(node), ctx["method"], node.lineno))
        return None

    def _add_equation(self, function, seq, target, kind, rhs, derivative, conds, cond_reads, line, ctx,
                      rhs_reads=None, state_read=None):
        eq_id = len(self.equations) + 1
        self.equations.append(dict(
            id=eq_id, function=function, seq=seq, target=target, kind=kind,
            expression=ast.unparse(rhs) if isinstance(rhs, ast.AST) else rhs,
            derivative=self._qualify(derivative, ctx), condition=" and ".join(conds) or None,
            first_exec=None, line=line))
        self.equations[-1]["resolved"] = self._qualify(self.equations[-1]["expression"], ctx)
        if rhs_reads is None:
            rhs_reads, curves = self._reads(rhs, ctx)
        else:
            curves = []
            if isinstance(rhs, ast.AST):
                curves = self._reads(rhs, ctx)[1]
        for var in rhs_reads:
            self.inputs[(eq_id, var, "rhs")] = 0
        for var in cond_reads:
            self.inputs[(eq_id, var, "condition")] = 0
        if state_read:
            self.inputs[(eq_id, state_read, "integrator_state")] = 1
        for curve, arg in curves:
            self.curve_uses.append(dict(equation_id=eq_id, curve=curve, argument=arg))
        self.items[function].append((seq, "eq", eq_id))
        return eq_id

    def _add_call(self, caller, callee, seq, conds, line):
        call_id = len(self.calls) + 1
        self.calls.append(dict(id=call_id, caller=caller, callee=callee, seq=seq,
                               condition=" and ".join(conds) or None, line=line))
        self.items[caller].append((seq, "call", call_id))

    def _walk(self, body, ctx, function, conds, cond_reads=()):
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    self._assign(stmt, target, stmt.value, ctx, function, conds, cond_reads)
            elif isinstance(stmt, ast.AugAssign):
                fake = ast.BinOp(left=stmt.target, op=stmt.op, right=stmt.value)
                ast.copy_location(fake, stmt)
                self._assign(stmt, stmt.target, fake, ctx, function, conds, cond_reads)
            elif isinstance(stmt, ast.If):
                test = ast.unparse(stmt.test)
                reads = list(cond_reads) + self._reads(stmt.test, ctx)[0]
                self._walk(stmt.body, ctx, function, conds + [test], reads)
                if stmt.orelse:
                    self._walk(stmt.orelse, ctx, function, conds + ["not (%s)" % test], reads)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                self._call_stmt(stmt, ctx, function, conds)
            elif isinstance(stmt, ast.FunctionDef):
                self._implicit_residual(stmt, ctx, conds, cond_reads)
            elif isinstance(stmt, ast.Return):
                seq = self._next_seq(ctx)
                target = self._local(ctx, function.split(".")[-1], stmt.lineno)
                self._add_equation(function, seq, target, "implicit_residual", stmt.value, None,
                                   conds, cond_reads, stmt.lineno, ctx)
            elif isinstance(stmt, (ast.Pass, ast.For)):
                if isinstance(stmt, ast.For):  # only in step(): timers count every step
                    self._add_call(function, "Timer.count", self._next_seq(ctx), conds, stmt.lineno)
            else:
                self.issues.append(("unsupported_statement", ast.unparse(stmt)[:120], function, stmt.lineno))

    def _call_stmt(self, stmt, ctx, function, conds):
        func = stmt.value.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            base = func.value.id
            owner = ctx["module"] if base == "self" else base
            if func.attr.endswith("_func"):
                callee = _qual(owner, func.attr)
                self._add_call(function, callee, self._next_seq(ctx), conds, stmt.lineno)
                return
            if base in IGNORED_NAMES:
                return
        self.issues.append(("unsupported_call", ast.unparse(stmt)[:120], function, stmt.lineno))

    def _implicit_residual(self, node, ctx, conds, cond_reads):
        qualname = _qual(ctx["method"], node.name)
        self._add_function(ctx["module"], node.name, qualname, "implicit_residual", None, node)
        self._local(ctx, node.name, node.lineno)
        # The residual body runs inside the solver call; it is emitted when the
        # impliciteq() equation that references it is reached.
        self.pending_residual[_qual(ctx["method"], node.name)] = (node, dict(ctx), conds, cond_reads)

    def _assign(self, stmt, target_node, rhs, ctx, function, conds, cond_reads):
        target = self._target(target_node, ctx)
        if target is None:
            return
        kind, derivative, rhs_reads, state_read = "algebraic", None, None, None
        if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name):
            name = rhs.func.id
            args = rhs.args
            if name in ("diffeq", "stablediffeq"):
                kind, derivative = "integrate", ast.unparse(args[0])
                rhs_reads = self._reads(args[0], ctx)[0]
                state_read = target
            elif name == "backwardeuler":
                kind = "integrate"
                derivative = "(%s) - (%s) * y" % (ast.unparse(args[0]), ast.unparse(args[1]))
                rhs_reads = self._reads(args[0], ctx)[0] + self._reads(args[1], ctx)[0]
                state_read = target
            elif name == "delay":
                kind = "delay"
                derivative = "(%s) * ((%s) - y)" % (ast.unparse(args[0]), ast.unparse(args[1]))
                rhs_reads = self._reads(args[0], ctx)[0] + self._reads(args[1], ctx)[0]
                state_read = target
            elif name == "impliciteq":
                kind = "implicit"
                residual_name = args[0].id if isinstance(args[0], ast.Name) else None
                self._emit_residual(ctx, residual_name, target, function)
            elif name == "Timer":
                kind = "timer"
            if kind in ("integrate", "delay"):
                self.integrator[target] = name
        elif (isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Attribute)
              and rhs.func.attr.endswith("_curve")):
            kind = "curve"
        elif _literal(rhs) is not None or (isinstance(rhs, ast.Constant) and rhs.value is None):
            kind = "constant"
        seq = self._next_seq(ctx)
        self._add_equation(function, seq, target, kind, rhs, derivative, conds, cond_reads,
                           stmt.lineno, ctx, rhs_reads=rhs_reads, state_read=state_read)

    def _emit_residual(self, ctx, residual_name, target, function):
        key = _qual(ctx["method"], residual_name or "")
        pending = self.pending_residual.pop(key, None)
        if pending is None:
            self.issues.append(("missing_residual", key, function, None))
            return
        node, saved_ctx, conds, cond_reads = pending
        self._add_call(function, key, self._next_seq(ctx), conds, node.lineno)
        sub_ctx = dict(saved_ctx, seq=[0])
        # The solver's trial value: residual parameter <- current estimate of the target.
        for arg in node.args.args:
            local = self._local(ctx, arg.arg, node.lineno)
            seq = self._next_seq(sub_ctx)
            self._add_equation(key, seq, local, "implicit_iterate", target, None, conds, cond_reads,
                               node.lineno, sub_ctx, rhs_reads=[target])
        self._walk(node.body, sub_ctx, key, conds, cond_reads)

    # -------------------------------------------------------------- schedule

    def _schedule(self):
        """Flatten one step() into an ordered list of block entries and equations."""
        self.execution = []
        roots = {"Structure.Context_func": "Context", "Structure.Parms_func": "Parms",
                 "Structure.Dervs_func": "Dervs", "Structure.Wrapup_func": "Wrapup"}
        def expand(function, depth, path_conds, parent, phase, stack):
            for _, item, item_id in sorted(self.items.get(function, [])):
                if item == "call":
                    call = self.calls[item_id - 1]
                    callee = call["callee"]
                    sub_phase = roots.get(callee, phase)
                    conds = path_conds + ([call["condition"]] if call["condition"] else [])
                    ord_ = len(self.execution) + 1
                    self.execution.append(dict(
                        ord=ord_, phase=sub_phase, depth=depth, item="call", function=callee,
                        equation_id=None, condition=" and ".join(conds) or None, parent_ord=parent))
                    if callee in self.functions:
                        self.functions[callee]["scheduled"] = 1
                        if callee not in stack:
                            expand(callee, depth + 1, conds, ord_, sub_phase, stack | {callee})
                    elif callee != "Timer.count":
                        self.issues.append(("missing_function", callee, call["caller"], call["line"]))
                else:
                    eq = self.equations[item_id - 1]
                    conds = path_conds + ([eq["condition"]] if eq["condition"] else [])
                    ord_ = len(self.execution) + 1
                    self.execution.append(dict(
                        ord=ord_, phase=phase, depth=depth, item="equation", function=function,
                        equation_id=item_id, condition=" and ".join(conds) or None, parent_ord=parent))
                    if eq["first_exec"] is None:
                        eq["first_exec"] = ord_

        step = _qual(RUNTIME_MODULE, "step")
        self.functions[step]["scheduled"] = 1
        expand(step, 0, [], None, "Step", frozenset([step]))

        # Lagged reads: on an equation's first execution, an input that is computed
        # somewhere in the step but has not yet been written was carried over from
        # the previous step (or its initial value).
        first_write = {}
        for row in self.execution:
            if row["item"] == "equation":
                first_write.setdefault(self.equations[row["equation_id"] - 1]["target"], row["ord"])
        for key in list(self.inputs):
            eq_id, var, via = key
            if via == "integrator_state":
                continue
            eq = self.equations[eq_id - 1]
            if eq["first_exec"] is None or var not in first_write:
                continue
            if first_write[var] >= eq["first_exec"]:
                self.inputs[key] = 1

    # ------------------------------------------------------------- variables

    def _resolve_variables(self):
        writers, foreign = defaultdict(set), defaultdict(bool)
        for eq in self.equations:
            fn = self.functions[eq["function"]]
            if fn["kind"] == "init":
                continue
            writers[eq["target"]].add(eq["function"])
            owner = eq["target"].split(".")[0]
            if fn["module"] != owner:
                foreign[eq["target"]] = True
        readers = defaultdict(set)
        for (eq_id, var, _via) in self.inputs:
            readers[var].add(eq_id)
        integrator = self.integrator

        names = set(self.declared) | set(writers) | set(readers)
        names = {n for n in names if n not in self.locals_seen}
        self.variables = []
        for qualname in sorted(names):
            module, name = qualname.split(".", 1)
            decl = self.declared.get(qualname)
            scheduled_writers = [w for w in writers[qualname] if self.functions[w]["scheduled"]]
            dtype = decl["dtype"] if decl else None
            if qualname in integrator:
                role = "state"
            elif dtype == "timer":
                role = "timer"
            elif scheduled_writers:
                role = "computed"
            elif writers[qualname]:
                role = "init_only"
            elif decl:
                role = "parameter" if dtype not in ("none",) else "unassigned"
            else:
                role = "undeclared"
                self.issues.append(("undeclared_variable", qualname,
                                    "read %d time(s), never declared or assigned" % len(readers[qualname]), None))
            if module not in self.classes and module != RUNTIME_MODULE:
                self.issues.append(("unknown_module", qualname, "module does not exist", None))
            self.variables.append(dict(
                module=module, name=name, qualname=qualname, scope="attribute", role=role, dtype=dtype,
                declared=int(decl is not None),
                initial_expr=decl["initial_expr"] if decl else None,
                initial_value=decl["initial_value"] if decl else None,
                writer_count=len(writers[qualname]), reader_count=len(readers[qualname]),
                foreign_writes=int(foreign[qualname]), integrator=integrator.get(qualname),
                scc_id=None, line=decl["line"] if decl else None))
        for qualname, (module, line) in sorted(self.locals_seen.items()):
            self.variables.append(dict(
                module=module, name=qualname.split(".", 2)[2], qualname=qualname, scope="local",
                role="local", dtype=None, declared=0, initial_expr=None, initial_value=None,
                writer_count=len(writers[qualname]), reader_count=len(readers[qualname]),
                foreign_writes=0, integrator=None, scc_id=None, line=line))
        for fn in self.functions.values():
            if fn["kind"] == "block" and not fn["scheduled"]:
                self.issues.append(("unscheduled_block", fn["qualname"],
                                    "not reachable from step()", fn["line_start"]))

    def _loops(self):
        """Tarjan SCC over the variable dependency graph (all reads, excluding self-integration)."""
        graph = defaultdict(set)
        for (eq_id, var, via) in self.inputs:
            if via == "integrator_state":
                continue
            target = self.equations[eq_id - 1]["target"]
            if var != target:
                graph[var].add(target)
        nodes = sorted({v["qualname"] for v in self.variables})
        index, low, on_stack, stack, comps = {}, {}, set(), [], []
        counter = 0
        for root in nodes:
            if root in index:
                continue
            work = [(root, iter(sorted(graph[root])))]
            index[root] = low[root] = counter
            counter += 1
            stack.append(root)
            on_stack.add(root)
            while work:
                node, it = work[-1]
                advanced = False
                for nxt in it:
                    if nxt not in index:
                        index[nxt] = low[nxt] = counter
                        counter += 1
                        stack.append(nxt)
                        on_stack.add(nxt)
                        work.append((nxt, iter(sorted(graph[nxt]))))
                        advanced = True
                        break
                    if nxt in on_stack:
                        low[node] = min(low[node], index[nxt])
                if advanced:
                    continue
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[node])
                if low[node] == index[node]:
                    comp = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        comp.append(w)
                        if w == node:
                            break
                    if len(comp) > 1:
                        comps.append(sorted(comp))
        comps.sort(key=lambda c: (-len(c), c[0]))
        by_name = {v["qualname"]: v for v in self.variables}
        self.loops = []
        for scc_id, comp in enumerate(comps, start=1):
            for q in comp:
                by_name[q]["scc_id"] = scc_id
            self.loops.append(dict(scc_id=scc_id, size=len(comp),
                                   modules=len({q.split(".")[0] for q in comp}),
                                   sample=", ".join(comp[:12])))

    # ----------------------------------------------------------------- write

    def write(self, db_path):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        tmp = db_path + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        con = sqlite3.connect(tmp)
        with open(SCHEMA, encoding="utf-8") as f:
            con.executescript(f.read())

        def insert(table, rows, cols):
            con.executemany(
                "INSERT INTO %s (%s) VALUES (%s)" % (table, ",".join(cols), ",".join("?" * len(cols))),
                [tuple(r[c] for c in cols) for r in rows])

        insert("modules", self.modules,
               ["name", "subsystem", "kind", "instantiated", "line_start", "line_end", "doc"])
        insert("functions", self.functions.values(),
               ["module", "name", "qualname", "kind", "phase", "scheduled", "line_start", "line_end", "source"])
        insert("variables", self.variables,
               ["module", "name", "qualname", "scope", "role", "dtype", "declared", "initial_expr",
                "initial_value", "writer_count", "reader_count", "foreign_writes", "integrator", "scc_id", "line"])
        insert("equations", self.equations,
               ["id", "function", "seq", "target", "kind", "expression", "resolved", "derivative", "condition",
                "first_exec", "line"])
        con.executemany("INSERT INTO equation_inputs VALUES (?,?,?,?)",
                        sorted((k[0], k[1], k[2], v) for k, v in self.inputs.items()))
        insert("curves", self.curves,
               ["function", "module", "name", "n_points", "x_json", "y_json", "slope_json",
                "x_min", "x_max", "y_min", "y_max"])
        insert("curve_uses", self.curve_uses, ["equation_id", "curve", "argument"])
        insert("calls", self.calls, ["id", "caller", "callee", "seq", "condition", "line"])
        insert("execution", self.execution,
               ["ord", "phase", "depth", "item", "function", "equation_id", "condition", "parent_ord"])
        con.executemany("INSERT INTO issues (kind, subject, detail, line) VALUES (?,?,?,?)",
                        sorted(set(self.issues), key=lambda i: (i[0], i[1], str(i[2]), i[3] or 0)))
        insert("loops", self.loops, ["scc_id", "size", "modules", "sample"])

        rel = os.path.relpath(self.source_path, REPO_ROOT)
        meta = {
            "source": rel.replace(os.sep, "/"),
            "source_sha256": hashlib.sha256(self.source.encode("utf-8")).hexdigest(),
            "source_lines": str(len(self.lines)),
            "extractor_version": EXTRACTOR_VERSION,
        }
        for table in ("modules", "functions", "variables", "equations", "equation_inputs", "curves",
                      "curve_uses", "calls", "execution", "issues", "loops"):
            meta["count_" + table] = str(con.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0])
        con.executemany("INSERT INTO meta VALUES (?,?)", sorted(meta.items()))
        con.commit()
        con.close()
        os.replace(tmp, db_path)
        return meta


def build(source=DEFAULT_SOURCE, out=DEFAULT_DB):
    extractor = Extractor(source)
    extractor.run()
    return extractor.write(out)
