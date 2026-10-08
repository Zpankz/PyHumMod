# archdb: the PyHumMod architecture as a queryable database

`archdb` reads `src/hummod.py` with Python's `ast` module and writes a SQLite
database describing every module, block, variable, equation, curve, call and
the order they run in one `step()`. It never imports or runs the model, so it
needs only the standard library and works even while the simulator itself is
broken.

```
python -m archdb build            # writes build/hummod_arch.db (~1 s); other commands build it on demand
python -m archdb stats
python -m archdb show ADHPool.conc_ADH
python -m unittest discover tests
```

The database is rebuilt automatically when `src/hummod.py` changes (its
SHA-256 is stored in `meta`), and two builds of the same source are identical.

## What is in it

| table / view | one row per |
|---|---|
| `modules` | class in hummod.py (a HumMod `.DES` component), plus the `hummod` pseudo-module for `step()` |
| `functions` | method: `block` (`*_func`), `curve` (`*_curve`), `init`, `implicit_residual` (the nested function handed to `impliciteq`) |
| `variables` | `Module.attr`, or `Module.method.local` for function locals |
| `equations` | assignment inside a function, in statement order, with its guarding `if` condition |
| `equation_inputs` | variable read by an equation (`via` = `rhs`, `condition` for an enclosing `if`, `call_condition` for a guard on a call that reaches it, `timestep`, or `integrator_state`) |
| `curves`, `curve_uses` | spline points/slopes, and where each curve is evaluated and with what argument |
| `calls` | block-to-block call edge, with its condition |
| `execution` | entry in the flattened program of one `step()`: every block entry and equation, in run order |
| `loops` | strongly connected component of the dependency graph: `feedback` (all reads) or `algebraic` (same-step reads only) |
| `issues` | conversion defect found while extracting |
| `dependencies` (view) | variable edge `src -> dst` |
| `state_variables`, `parameters` (views) | integrated variables with their rate, and settable inputs |

### Variable roles

- `state`: integrated each step by `diffeq`, `stablediffeq`, `backwardeuler` or `delay` (the rate is in `equations.derivative`).
- `computed`: assigned by a block that `step()` reaches.
- `parameter`: given a non-`None` value in `__init__` and never reassigned. These are the knobs, e.g. `BetaBlockade.Block_percent`.
- `init_only`: only assigned by blocks `step()` never reaches.
- `unassigned`: declared `None` and never assigned.
- `undeclared`: read but never declared or assigned (also listed in `issues`).
- `timer`, `local`.

### Lagged reads

`equation_inputs.lagged = 1` means that, the first time the equation runs in a
step, the variable it reads has not yet been written in that step, so it sees
the previous step's value. This separates same-step algebra from feedback that
closes across time steps.

## Python API

```python
from archdb import open_db
db = open_db()

db.variable("ADHPool.Mass")                 # role, defining equations with inputs, readers
db.function("ADH.Dervs_func")               # equations and calls in order, and its callers
db.module("ADHPool")
db.inputs("ADHPool.conc_ADH")               # direct dependencies
db.upstream("Heart_Ventricles.Rate", 3)     # transitive dependencies, with distance and role
db.downstream("BetaBlockade.Block_percent") # everything a change entails
db.entailments("BetaBlockade.Block_percent")  # counts, affected states, affected modules
db.path("BetaBlockade.Block_percent", "Heart_Ventricles.Rate")
db.callees("Structure.Dervs_func", transitive=True)
db.schedule(phase="Dervs", equations=False)
db.loop_of("ADHPool.Mass")                 # kind="algebraic" for the same-step loop
db.eval_curve("ADHSecretion.NeuralEffect_curve", 1.1)
db.sql("SELECT * FROM state_variables")
```

`upstream`/`downstream` take `include_conditions=False` to ignore `if`
dependencies and `include_lagged=False` to stay within one step.

## CLI

```
python -m archdb search PO2*            # names, * wildcard
python -m archdb show Skin_Flow.Calc_func
python -m archdb up Heart_Ventricles.Rate -d 2
python -m archdb down BetaBlockade.Block_percent --same-step
python -m archdb entails BetaBlockade.Block_percent
python -m archdb path BetaBlockade.Block_percent Heart_Ventricles.Rate
python -m archdb calls Structure.Parms_func -t
python -m archdb schedule --phase Dervs --calls-only
python -m archdb loops                  # --kind algebraic for same-step loops only
python -m archdb issues --kind undeclared_variable
python -m archdb curve ADHSecretion.NeuralEffect_curve --x 1.1
python -m archdb sql "SELECT kind, COUNT(*) FROM equations GROUP BY kind"
```

Every command takes `--json`.

## Limits

- The analysis is static. `if` conditions are recorded, not evaluated, so the
  execution order and lagged flags assume every branch can run.
- A few helper modules (`HgbTissue`, `HgbProps`, `Blood_GasToBase`, ...) are
  used as shared scratchpads: many blocks write their inputs and read their
  outputs. Those writes are flagged with `variables.foreign_writes`, and they
  join otherwise separate subsystems into the largest loops.
- `hummod.py` defines 53 classes twice. Python keeps the last definition, so
  only that one is extracted; the shadowed ones are listed under
  `issues.kind = 'duplicate_class'`.
