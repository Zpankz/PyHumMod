-- PyHumMod architecture database.
-- Built by archdb/extract.py from a static (AST) parse of src/hummod.py.
-- Nothing in here is produced by running the simulation.

PRAGMA foreign_keys = ON;

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- One row per class in hummod.py. Each class is one HumMod .DES component.
CREATE TABLE modules (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    subsystem    TEXT NOT NULL,          -- name prefix before the first "_" (e.g. OtherTissue_Flow -> OtherTissue)
    kind         TEXT NOT NULL,          -- model | structure | runtime
    instantiated INTEGER NOT NULL,       -- 1 if hummod.py creates the module-level singleton
    line_start   INTEGER NOT NULL,
    line_end     INTEGER NOT NULL,
    doc          TEXT
);

-- Methods: procedural blocks (*_func), curves (*_curve), constructors (__init__),
-- and implicit-equation residual functions nested inside a *_func.
CREATE TABLE functions (
    id          INTEGER PRIMARY KEY,
    module      TEXT NOT NULL REFERENCES modules(name),
    name        TEXT NOT NULL,
    qualname    TEXT NOT NULL UNIQUE,    -- Module.method
    kind        TEXT NOT NULL,           -- block | curve | init | implicit_residual | runtime
    phase       TEXT,                    -- block name without _func (Parms, Dervs, CalcConc, ...)
    scheduled   INTEGER NOT NULL DEFAULT 0, -- 1 if reachable from hummod.step()
    line_start  INTEGER NOT NULL,
    line_end    INTEGER NOT NULL,
    source      TEXT NOT NULL
);

-- Variables: module attributes (Module.attr) plus function-local names
-- (Module.method.local). Roles are derived from how the variable is written.
CREATE TABLE variables (
    id             INTEGER PRIMARY KEY,
    module         TEXT NOT NULL,
    name           TEXT NOT NULL,
    qualname       TEXT NOT NULL UNIQUE,
    scope          TEXT NOT NULL,        -- attribute | local
    role           TEXT NOT NULL,        -- state | computed | parameter | timer | unassigned | undeclared | local
    dtype          TEXT,                 -- float | int | bool | str | none | expr | timer
    declared       INTEGER NOT NULL,     -- 1 if set in the module's __init__
    initial_expr   TEXT,                 -- right-hand side in __init__, verbatim
    initial_value  REAL,                 -- numeric initial value when it is a literal
    writer_count   INTEGER NOT NULL DEFAULT 0,  -- distinct functions that assign it (excluding __init__)
    reader_count   INTEGER NOT NULL DEFAULT 0,  -- distinct equations that read it
    foreign_writes INTEGER NOT NULL DEFAULT 0,  -- 1 if assigned from a module other than its own
    integrator     TEXT,                 -- diffeq | delay | backwardeuler | stablediffeq (state variables)
    scc_id         INTEGER,              -- strongly connected component in the same-step dependency graph
    line           INTEGER
);

-- Every assignment statement inside a function, in source order.
CREATE TABLE equations (
    id            INTEGER PRIMARY KEY,
    function      TEXT NOT NULL REFERENCES functions(qualname),
    seq           INTEGER NOT NULL,      -- statement order inside the function (shared with calls.seq)
    target        TEXT NOT NULL,         -- variable qualname
    kind          TEXT NOT NULL,         -- algebraic | integrate | delay | implicit | curve | constant | timer | implicit_residual | implicit_iterate
    expression    TEXT NOT NULL,         -- right-hand side, verbatim
    resolved      TEXT NOT NULL,         -- right-hand side with self.X rewritten as Module.X
    derivative    TEXT,                  -- for integrate/delay: the rate expression (resolved)
    condition     TEXT,                  -- conjunction of enclosing if/else tests, NULL if unconditional
    first_exec    INTEGER,               -- execution.ord of its first run inside one step(), NULL if never run
    line          INTEGER NOT NULL
);

-- Variables read by an equation (right-hand side or guarding condition).
CREATE TABLE equation_inputs (
    equation_id INTEGER NOT NULL REFERENCES equations(id),
    variable    TEXT NOT NULL,
    via         TEXT NOT NULL,           -- rhs | condition | integrator_state
    lagged      INTEGER NOT NULL DEFAULT 0, -- 1 if, on first execution, the value read was last written in the previous step
    PRIMARY KEY (equation_id, variable, via)
);

-- Curve (cubic Hermite spline) definitions.
CREATE TABLE curves (
    function    TEXT PRIMARY KEY REFERENCES functions(qualname),
    module      TEXT NOT NULL,
    name        TEXT NOT NULL,           -- without _curve
    n_points    INTEGER NOT NULL,
    x_json      TEXT NOT NULL,
    y_json      TEXT NOT NULL,
    slope_json  TEXT NOT NULL,
    x_min REAL, x_max REAL, y_min REAL, y_max REAL
);

-- Where each curve is evaluated, with its input expression.
CREATE TABLE curve_uses (
    equation_id INTEGER NOT NULL REFERENCES equations(id),
    curve       TEXT NOT NULL,
    argument    TEXT NOT NULL
);

-- Call graph between *_func blocks.
CREATE TABLE calls (
    id         INTEGER PRIMARY KEY,
    caller     TEXT NOT NULL,
    callee     TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    condition  TEXT,
    line       INTEGER NOT NULL
);

-- The flattened program of one hummod.step(): every block call and every
-- equation, in the order it runs, with the call path that reached it.
CREATE TABLE execution (
    ord         INTEGER PRIMARY KEY,
    phase       TEXT NOT NULL,           -- Context | Parms | Dervs | Wrapup | Step
    depth       INTEGER NOT NULL,
    item        TEXT NOT NULL,           -- call | equation
    function    TEXT NOT NULL,           -- the block being entered (call) or containing the equation
    equation_id INTEGER,
    condition   TEXT,                    -- accumulated guards along the path
    parent_ord  INTEGER
);

-- Unresolved references found while extracting (likely conversion defects).
CREATE TABLE issues (
    id        INTEGER PRIMARY KEY,
    kind      TEXT NOT NULL,
    subject   TEXT NOT NULL,
    detail    TEXT,
    line      INTEGER
);

-- Strongly connected components (feedback loops) of the same-step dependency graph.
CREATE TABLE loops (
    scc_id  INTEGER PRIMARY KEY,
    size    INTEGER NOT NULL,
    modules INTEGER NOT NULL,
    sample  TEXT
);

-- Variable-to-variable dependency edges: src is read to compute dst.
CREATE VIEW dependencies AS
SELECT DISTINCT i.variable AS src,
       e.target            AS dst,
       i.via               AS via,
       i.lagged            AS lagged,
       e.kind              AS eq_kind
FROM equation_inputs i
JOIN equations e ON e.id = i.equation_id
WHERE i.variable <> e.target OR e.kind IN ('integrate', 'delay');

CREATE VIEW state_variables AS
SELECT v.qualname, v.integrator, e.derivative, e.function, e.line
FROM variables v
JOIN equations e ON e.target = v.qualname AND e.kind IN ('integrate', 'delay')
WHERE v.role = 'state';

CREATE VIEW parameters AS
SELECT qualname, dtype, initial_expr, initial_value, reader_count
FROM variables WHERE role = 'parameter';

CREATE INDEX idx_eq_target   ON equations(target);
CREATE INDEX idx_eq_function ON equations(function);
CREATE INDEX idx_in_var      ON equation_inputs(variable);
CREATE INDEX idx_calls_caller ON calls(caller);
CREATE INDEX idx_calls_callee ON calls(callee);
CREATE INDEX idx_var_module  ON variables(module);
CREATE INDEX idx_exec_eq     ON execution(equation_id);
CREATE INDEX idx_exec_fn     ON execution(function);
