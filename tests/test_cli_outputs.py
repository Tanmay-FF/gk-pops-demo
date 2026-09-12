# Author: Tanmay Thaker <tthaker@gatekeepersystems.com>
"""The output-slot map cannot drift from the tuple the engine returns.

Run with:  python tests/test_cli_outputs.py

This is the regression test for `api/main.py:207`, which positionally unpacks 19
names from a tuple the engine grew to 21 and therefore raises `ValueError`
before `/api/run` can return anything. Nothing caught that because nothing was
checking, and it is exactly the failure a headless caller reading slots by
position would hit next.

Everything here is read with `ast`, and nothing under test is imported:

  * `engine/tracker.py` pulls in ultralytics and torch — about six seconds, and
    the detection weights on some paths. Counting the elements of a `Return`
    node costs milliseconds.
  * `app_poc_v2.py` opens `with gr.Blocks(...)` at module scope, so importing it
    builds the entire demo, registers every event and scans the sample-video
    directory. A test file cannot do that.

Text-matching the `return (` statement would be the obvious shortcut and is
brittle: it spans six lines today and any reformatting breaks it. The AST
survives reformatting, comments and line joins.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKER = os.path.join(REPO, "engine", "tracker.py")
APP = os.path.join(REPO, "app_poc_v2.py")


def _load_run_outputs():
    """engine/run_outputs.py loaded as a FILE, not as `engine.run_outputs`.

    `import engine.run_outputs` would execute `engine/__init__.py` first, and
    that does `from .tracker import TrackingEngine` — six seconds of ultralytics
    and torch for a module that holds a list of strings.

    Loading it off disk is also the assertion that it stays dependency-free:
    `run_outputs.py` imports nothing from the package, so this works. The day
    someone adds `from .config import ...` to it, this line raises and says so.
    """
    import importlib.util
    path = os.path.join(REPO, "engine", "run_outputs.py")
    spec = importlib.util.spec_from_file_location("_run_outputs_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ro = _load_run_outputs()
ENGINE_OUTPUT_NAMES = _ro.ENGINE_OUTPUT_NAMES
ENGINE_IDX = _ro.ENGINE_IDX
N_ENGINE_OUTPUTS = _ro.N_ENGINE_OUTPUTS
as_dict = _ro.as_dict

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def parse(path: str) -> ast.Module:
    with open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    """The named FunctionDef anywhere in the module, including inside a class."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function named {name!r}")


def final_return_arity(fn: ast.FunctionDef) -> int:
    """How many values the function's LAST `return` statement yields.

    The last one rather than the first: `_process_video` has earlier returns on
    its early-exit paths, and the tuple this map describes is the one at the
    end of the happy path.
    """
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    if not returns:
        raise AssertionError(f"{fn.name} has no return statement")
    last = max(returns, key=lambda n: n.lineno)
    if not isinstance(last.value, ast.Tuple):
        raise AssertionError(
            f"{fn.name}'s last return is a {type(last.value).__name__}, "
            f"not a tuple — this test assumes the positional tuple")
    return len(last.value.elts)


def module_list_literal(tree: ast.Module, name: str):
    """A module-level `name = [...]` as Python values, or None if it is not a
    plain list literal (which is the case after the run_outputs rewire)."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            if isinstance(node.value, ast.List):
                return [ast.literal_eval(e) for e in node.value.elts]
            return None
    raise AssertionError(f"no module-level assignment to {name!r}")


def run_output_names_expr(tree: ast.Module) -> ast.AST:
    """The right-hand side of app_poc_v2's `RUN_OUTPUT_NAMES = ...`."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "RUN_OUTPUT_NAMES"
                for t in node.targets):
            return node.value
    raise AssertionError("app_poc_v2 has no module-level RUN_OUTPUT_NAMES")


# ---------------------------------------------------------------------------
section("the map itself")

check("ENGINE_OUTPUT_NAMES has no duplicates",
      len(set(ENGINE_OUTPUT_NAMES)) == N_ENGINE_OUTPUTS)
check("ENGINE_IDX agrees with the list's order",
      all(ENGINE_OUTPUT_NAMES[i] == n for n, i in ENGINE_IDX.items()))
check("result_tabs is NOT an engine slot",
      "result_tabs" not in ENGINE_IDX,
      "it is the one slot the UI adds")

# ---------------------------------------------------------------------------
section("engine/tracker.py's return tuple")

tracker_tree = parse(TRACKER)
arity = final_return_arity(find_function(tracker_tree, "_process_video"))
check("_process_video returns exactly N_ENGINE_OUTPUTS values",
      arity == N_ENGINE_OUTPUTS,
      f"tuple has {arity}, map has {N_ENGINE_OUTPUTS}")

# ---------------------------------------------------------------------------
section("app_poc_v2.py is derived, not retyped")

app_tree = parse(APP)
expr = run_output_names_expr(app_tree)
check("RUN_OUTPUT_NAMES is no longer a hand-written list literal",
      module_list_literal(app_tree, "RUN_OUTPUT_NAMES") is None,
      "it must be derived from ENGINE_OUTPUT_NAMES")
check("...and it mentions ENGINE_OUTPUT_NAMES",
      any(isinstance(n, ast.Name) and n.id == "ENGINE_OUTPUT_NAMES"
          for n in ast.walk(expr)))
check("...and appends exactly ['result_tabs']",
      any(isinstance(n, ast.List)
          and [ast.literal_eval(e) for e in n.elts] == ["result_tabs"]
          for n in ast.walk(expr)))
check("app_poc_v2 imports ENGINE_OUTPUT_NAMES from engine.run_outputs",
      any(isinstance(n, ast.ImportFrom) and n.module == "engine.run_outputs"
          and any(a.name == "ENGINE_OUTPUT_NAMES" for a in n.names)
          for n in ast.walk(app_tree)))
check("the result_tabs-is-last assert survives the rewire",
      any(isinstance(n, ast.Assert) and "result_tabs" in ast.dump(n.test)
          for n in ast.walk(app_tree)),
      "_IDX maps engine outputs by position and relies on it")

# The UI's own lookups must still resolve. A name it reads that this map does
# not carry would be a KeyError in run_analysis, at runtime, on a real run.
_ui_reads = set()
for node in ast.walk(app_tree):
    if (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name) and node.value.id == "_IDX"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)):
        _ui_reads.add(node.slice.value)
_unknown = sorted(_ui_reads - set(ENGINE_OUTPUT_NAMES) - {"result_tabs"})
check("every _IDX[...] the UI reads is a name this map carries",
      not _unknown, f"unknown: {_unknown}" if _unknown else
      f"{len(_ui_reads)} distinct lookups")

# ---------------------------------------------------------------------------
section("as_dict is the guard api/main.py lacks")

good = as_dict(tuple(range(N_ENGINE_OUTPUTS)))
check("as_dict maps every slot by name",
      set(good) == set(ENGINE_OUTPUT_NAMES)
      and good["json_output"] == ENGINE_IDX["json_output"])
check("as_dict accepts a list as well as a tuple",
      as_dict(list(range(N_ENGINE_OUTPUTS))) == good)


def raises_runtime(fn) -> str:
    try:
        fn()
    except RuntimeError as e:
        return str(e)
    except Exception as e:                       # wrong type is a failure
        return f"!{type(e).__name__}: {e}"
    return ""


short = raises_runtime(lambda: as_dict(tuple(range(N_ENGINE_OUTPUTS - 2))))
check("a short tuple raises RuntimeError naming both counts",
      str(N_ENGINE_OUTPUTS - 2) in short and str(N_ENGINE_OUTPUTS) in short,
      short[:70])
long_ = raises_runtime(lambda: as_dict(tuple(range(N_ENGINE_OUTPUTS + 1))))
check("a long tuple raises too",
      str(N_ENGINE_OUTPUTS + 1) in long_ and str(N_ENGINE_OUTPUTS) in long_)
check("an empty sequence raises rather than returning {}",
      bool(raises_runtime(lambda: as_dict(()))))

# ---------------------------------------------------------------------------
print(f"\n{len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    for name in _FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
