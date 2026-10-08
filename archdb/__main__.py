"""Command line interface: python -m archdb <command> ...  (see archdb/README.md)."""

import argparse
import json
import signal
import sys

from . import extract
from .query import open_db


def _print_table(rows, limit=None):
    if not rows:
        print("(no rows)")
        return
    rows = rows[:limit] if limit else rows
    cols = list(rows[0].keys())
    text = [[("" if r[c] is None else str(r[c])) for c in cols] for r in rows]
    widths = [min(max(len(c), *(len(t[i]) for t in text)), 90) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for t in text:
        print("  ".join(v[:w].ljust(w) for v, w in zip(t, widths)))


def _emit(args, value):
    if args.json:
        print(json.dumps(value, indent=2, default=str))
    elif isinstance(value, list):
        _print_table(value)
    elif isinstance(value, dict):
        nested = {k: v for k, v in value.items() if isinstance(v, (list, dict))}
        for k, v in value.items():
            if k not in nested:
                text = str(v)
                if "\n" in text:
                    print("%s:\n    %s" % (k, text.replace("\n", "\n    ")))
                else:
                    print("%s: %s" % (k, text))
        for k, v in nested.items():
            print("\n[%s]" % k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                _print_table(v)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    print("  %s: %s" % (kk, vv))
            else:
                print("  %s" % (v,))
    else:
        print(value)


def _resolve_or_exit(db, name):
    kind, found = db.resolve(name)
    if kind is None:
        print("No exact match for %r. Candidates:" % name, file=sys.stderr)
        _print_table(found)
        sys.exit(1)
    return kind


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m archdb",
                                description="Query the PyHumMod architecture database.")
    p.add_argument("--db", default=extract.DEFAULT_DB, help="database path (default: build/hummod_arch.db)")
    p.add_argument("--source", default=extract.DEFAULT_SOURCE, help="model source (default: src/hummod.py)")
    p.add_argument("--json", action="store_true", help="print JSON instead of tables")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="print JSON")
    sub = p.add_subparsers(dest="cmd", required=True)

    def cmd(name, **kwargs):
        return sub.add_parser(name, parents=[common], **kwargs)


    cmd("build", help="(re)build the database from the source")
    cmd("stats", help="counts, roles, equation kinds, issues")
    s = cmd("search", help="find modules, functions and variables by name (* wildcard)")
    s.add_argument("pattern")
    s = cmd("show", help="describe a module, function or variable")
    s.add_argument("name")
    for name, helptext in (("up", "what a variable depends on (transitively)"),
                           ("down", "what a variable entails downstream (transitively)")):
        s = cmd(name, help=helptext)
        s.add_argument("variable")
        s.add_argument("-d", "--depth", type=int, default=None, help="maximum distance")
        s.add_argument("--same-step", action="store_true", help="skip reads of last step's values")
        s.add_argument("--no-conditions", action="store_true", help="skip if-condition dependencies")
    s = cmd("entails", help="summary of everything a change to a variable propagates to")
    s.add_argument("variable")
    s.add_argument("-d", "--depth", type=int, default=None)
    s = cmd("path", help="shortest dependency chain from one variable to another")
    s.add_argument("src")
    s.add_argument("dst")
    for name in ("calls", "callers"):
        s = cmd(name, help="call graph %s of a *_func block" % ("callees" if name == "calls" else name))
        s.add_argument("function")
        s.add_argument("-t", "--transitive", action="store_true")
    s = cmd("schedule", help="the flattened execution order of one step()")
    s.add_argument("--phase", choices=["Step", "Context", "Parms", "Dervs", "Wrapup"])
    s.add_argument("--function")
    s.add_argument("--calls-only", action="store_true")
    s.add_argument("--limit", type=int, default=200)
    s = cmd("loops", help="feedback or algebraic loops (strongly connected components)")
    s.add_argument("--kind", choices=["feedback", "algebraic"], default="feedback")
    s.add_argument("--limit", type=int, default=20)
    s = cmd("loop", help="the loop a variable belongs to")
    s.add_argument("variable")
    s.add_argument("--kind", choices=["feedback", "algebraic"], default="feedback")
    s = cmd("issues", help="conversion defects found during extraction")
    s.add_argument("--kind")
    s = cmd("curve", help="show a curve, or evaluate it at --x")
    s.add_argument("function", help="e.g. ADHSecretion.NeuralEffect_curve")
    s.add_argument("--x", type=float)
    s = cmd("sql", help="run a read-only SQL query")
    s.add_argument("query")

    args = p.parse_args(argv)
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # allow piping into head

    if args.cmd == "build":
        _emit(args, extract.build(args.source, args.db))
        return
    db = open_db(args.db, args.source)

    if args.cmd == "stats":
        _emit(args, db.stats())
    elif args.cmd == "search":
        _emit(args, db.search(args.pattern))
    elif args.cmd == "show":
        kind = _resolve_or_exit(db, args.name)
        _emit(args, getattr(db, kind)(args.name))
    elif args.cmd in ("up", "down"):
        _resolve_or_exit(db, args.variable)
        fn = db.upstream if args.cmd == "up" else db.downstream
        _emit(args, fn(args.variable, args.depth, include_conditions=not args.no_conditions,
                       include_lagged=not args.same_step))
    elif args.cmd == "entails":
        _resolve_or_exit(db, args.variable)
        _emit(args, db.entailments(args.variable, args.depth))
    elif args.cmd == "path":
        chain = db.path(args.src, args.dst)
        _emit(args, chain if args.json else (" -> ".join(chain) if chain else "(no dependency path)"))
    elif args.cmd == "calls":
        _emit(args, db.callees(args.function, args.transitive))
    elif args.cmd == "callers":
        _emit(args, db.callers(args.function, args.transitive))
    elif args.cmd == "schedule":
        _emit(args, db.schedule(args.phase, args.function, equations=not args.calls_only, limit=args.limit))
    elif args.cmd == "loops":
        _emit(args, db.loops(args.kind, args.limit))
    elif args.cmd == "loop":
        _emit(args, db.loop_of(args.variable, args.kind) or "(not in a %s loop)" % args.kind)
    elif args.cmd == "issues":
        _emit(args, db.issues(args.kind))
    elif args.cmd == "curve":
        if args.x is not None:
            _emit(args, db.eval_curve(args.function, args.x))
        else:
            _emit(args, db.curve(args.function))
    elif args.cmd == "sql":
        db.con.execute("PRAGMA query_only = ON")
        _emit(args, db.sql(args.query))


if __name__ == "__main__":
    main()
