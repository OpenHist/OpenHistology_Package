"""Private import namespaces for unmodified, pinned upstream working trees.

Local absolute imports are rewritten to the repository's private namespace.
Optional exact source replacements are declared by each adapter family.
No global ``models``/``utils`` aliases or sys.path mutations are used.
"""

import ast
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import threading
import types

from ..snapshot import validate_source_tree

ROOT = Path(__file__).resolve().parents[1]
_PREFIX = "openhisto.mil.models.code_bases._native"
_LOCK = threading.RLock()
_CONFIG = {}


def repository(slug):
    path = ROOT / "upstream" / slug
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing upstream source {slug}; run python -m openhisto.mil.models.code_bases.sources"
        )
    return path


def validate_source(slug):
    """Validate the pinned revision and tracked bytes before the first import."""
    directory = repository(slug)
    record = ROOT / "compat" / "sources" / (slug + ".json")
    if not record.is_file():
        raise ImportError(f"Upstream source is unmanaged: {slug}")
    try:
        return validate_source_tree(ROOT, slug, json.loads(record.read_text()), directory=directory)
    except RuntimeError as exc:
        raise ImportError(
            f"Pinned source has changed: {slug}; {exc}; preserve edits and use an explicit source revision"
        ) from exc


class _Imports(ast.NodeTransformer):
    def __init__(self, prefix, local):
        self.prefix, self.local = prefix, local

    def visit_ImportFrom(self, node):
        if node.level == 0 and node.module and node.module.split(".")[0] in self.local:
            node.module = self.prefix + "." + node.module
        return node

    def visit_Import(self, node):
        result = []
        for alias in node.names:
            if alias.name.split(".")[0] not in self.local:
                result.append(ast.Import(names=[alias]))
            elif alias.asname or "." not in alias.name:
                result.append(
                    ast.Import(
                        names=[
                            ast.alias(
                                name=self.prefix + "." + alias.name,
                                asname=alias.asname or alias.name,
                            )
                        ]
                    )
                )
            else:
                top = alias.name.split(".")[0]
                result.append(
                    ast.Import(
                        names=[
                            ast.alias(
                                name=self.prefix + "." + alias.name, asname="_native_import_" + top
                            )
                        ]
                    )
                )
                result.append(
                    ast.ImportFrom(module=self.prefix, names=[ast.alias(name=top)], level=0)
                )
        return [ast.copy_location(item, node) for item in result]


class _Loader(importlib.abc.Loader):
    def __init__(self, path, prefix, config, package=False):
        self.path, self.prefix, self.config, self.package = path, prefix, config, package

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if self.path is None:
            return
        module.__file__ = str(self.path)
        source = self.path.read_text(encoding="utf-8-sig")
        relative = str(self.path.relative_to(repository(self.config["slug"])))
        for old, new in self.config["replacements"].get(relative, ()):
            if old not in source:
                raise ImportError(f"Pinned compatibility patch no longer matches {relative}")
            source = source.replace(old, new)
        tree = _Imports(self.prefix, self.config["local"]).visit(
            ast.parse(source, filename=str(self.path))
        )
        ast.fix_missing_locations(tree)
        exec(compile(tree, str(self.path), "exec"), module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for prefix, config in _CONFIG.items():
            if not fullname.startswith(prefix + "."):
                continue
            relative = fullname[len(prefix) + 1 :].replace(".", "/")
            for root in config["roots"]:
                package = root / relative
                if package.is_dir():
                    init = package / "__init__.py"
                    loader = _Loader(init if init.is_file() else None, prefix, config, True)
                    spec = importlib.util.spec_from_loader(fullname, loader, is_package=True)
                    spec.submodule_search_locations = [str(package)]
                    if init.is_file():
                        spec.origin = str(init)
                    return spec
                candidate = root / (relative + ".py")
                if candidate.is_file():
                    spec = importlib.util.spec_from_loader(
                        fullname, _Loader(candidate, prefix, config)
                    )
                    spec.origin = str(candidate)
                    return spec
        return None


def load_native(slug, module, *, roots=("",), replacements=None):
    """Import module relative to the first matching repository root.

    Register all replacements for a repo on its first load. Model families own
    disjoint repos, so a repo must never be reconfigured after modules load.
    """
    with _LOCK:
        prefix = _PREFIX + "." + slug
        if prefix not in _CONFIG:
            validate_source(slug)
            directory = repository(slug)
            paths = [directory / root for root in roots]
            local = {
                p.stem if p.is_file() else p.name
                for root in paths
                for p in root.iterdir()
                if p.is_dir() or p.suffix == ".py"
            }
            _CONFIG[prefix] = {
                "slug": slug,
                "roots": paths,
                "local": local,
                "replacements": replacements or {},
            }
            for name in (_PREFIX, prefix):
                if name not in sys.modules:
                    package = types.ModuleType(name)
                    package.__path__ = [str(path) for path in paths] if name == prefix else []
                    package.__package__ = name
                    sys.modules[name] = package
            if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
                sys.meta_path.insert(0, _Finder())
        else:
            config = _CONFIG[prefix]
            if replacements is not None and replacements != config["replacements"]:
                raise ImportError("Repository compatibility patches cannot change after import")
        return importlib.import_module(prefix + "." + module)
