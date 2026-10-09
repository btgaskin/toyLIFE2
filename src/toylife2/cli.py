"""Command-line interface for configured and recorded toyLIFE experiments."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from .config import apply_overrides, load_spec
from .records import inspect_record, verify_record
from .reports import render_comparison, render_report
from .workspaces import check_workspace, init_workspace, list_records, new_record_path


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _spec(args: argparse.Namespace):
    return apply_overrides(load_spec(args.config), args.set or [])


def _destination(args: argparse.Namespace, spec):
    if args.destination and args.workspace:
        raise ValueError("choose --destination or --workspace")
    if args.destination:
        return Path(args.destination)
    if args.workspace:
        return new_record_path(args.workspace, spec, args.name)
    return new_record_path(Path.cwd(), spec, args.name)


def _tables(args: argparse.Namespace) -> None:
    from .tables import Tables

    cache = args.cache_dir if args.tables_command == "generate" else None
    tables = Tables.load(source=args.source, cache_dir=cache)
    _json(
        {
            "status": "validated",
            "source": tables.source,
            "digest": tables.digest,
            "proteins": len(tables.protein_perimeter),
            "metabolites": len(tables.met_strings),
            "cache_dir": cache,
        }
    )


def _runner_function(name: str):
    try:
        runner = importlib.import_module("toylife2.runner")
    except ModuleNotFoundError as error:
        if error.name == "toylife2.runner":
            raise RuntimeError("experiment runner is unavailable in this build") from error
        raise
    function = getattr(runner, name, None)
    if function is None:
        raise RuntimeError(f"runner.{name} is unavailable in this build")
    return function


def _checkpoint_record(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "manifest.json").is_file():
            return candidate
    raise FileNotFoundError(f"no record manifest for checkpoint: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toylife2", description="Recorded toyLIFE experiments")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a TOML run configuration")
    validate.add_argument("config", type=Path)
    validate.add_argument("--set", action="append", default=[], metavar="PATH=TOML_VALUE")

    for name in ("run", "profile", "qualify"):
        command = commands.add_parser(name, help=f"{name} a configured experiment")
        command.add_argument("config", type=Path)
        command.add_argument("--set", action="append", default=[], metavar="PATH=TOML_VALUE")
        command.add_argument("--destination", type=Path)
        command.add_argument("--workspace", type=Path)
        command.add_argument("--name")

    resume = commands.add_parser("resume", help="resume a recorded checkpoint")
    resume.add_argument("checkpoint", type=Path)
    resume.add_argument("--destination", type=Path)
    resume.add_argument("--backend", choices=("cpu", "metal"))
    resume.add_argument("--trace-budget-mib", type=int)
    for resource in ("evolution-updates", "world-ticks", "chemical-updates"):
        resume.add_argument("--max-" + resource, type=int)

    batch = commands.add_parser("run-batch", help="run named v2 configs in one runtime")
    batch.add_argument("config", type=Path)
    batch.add_argument("--destination", type=Path, required=True)
    batch.add_argument("--resume", action="store_true")

    convert = commands.add_parser("convert-checkpoint", help="qualify a v1 to v2 continuation")
    convert.add_argument("checkpoint", type=Path)
    convert.add_argument("destination", type=Path)

    reproduce = commands.add_parser("reproduce", help="verify and rerun a saved record")
    reproduce.add_argument("record", type=Path)
    reproduce.add_argument("destination", type=Path)

    inspect = commands.add_parser("inspect", help="inspect provenance and integrity")
    inspect.add_argument("record", type=Path)

    report = commands.add_parser("report", help="render a static HTML report")
    report.add_argument("record", type=Path, nargs="?")
    report.add_argument("--destination", type=Path)
    report.add_argument("--compare", type=Path, nargs="+")

    workspace = commands.add_parser("workspace", help="manage a run workspace")
    workspace_commands = workspace.add_subparsers(dest="workspace_command", required=True)
    for name in ("init", "status", "check"):
        subcommand = workspace_commands.add_parser(name)
        subcommand.add_argument("path", type=Path)

    tables = commands.add_parser("tables", help="validate or cache the model lookup tables")
    table_commands = tables.add_subparsers(dest="tables_command", required=True)
    for name in ("generate", "check"):
        subcommand = table_commands.add_parser(name)
        subcommand.add_argument("--source", type=Path)
        subcommand.add_argument("--cache-dir", type=Path, default=Path(".cache/toylife2"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            from .specs import digest, scientific_identity

            spec = _spec(args)
            _json(
                {
                    "valid": True,
                    "operation": spec.operation,
                    "spec_sha256": digest(spec),
                    "scientific_identity": scientific_identity(spec),
                }
            )
        elif args.command in ("run", "profile", "qualify"):
            spec = _spec(args)
            destination = _destination(args, spec)
            function = _runner_function(args.command)
            print(function(spec, destination=destination))
        elif args.command == "resume":
            function = _runner_function("resume")
            execution = None
            recording = None
            if args.backend or args.trace_budget_mib is not None:
                from .records import load_record_spec

                record = _checkpoint_record(args.checkpoint)
                original = load_record_spec(record)
                if args.backend:
                    execution = dataclasses.replace(original.execution, backend=args.backend)
                if args.trace_budget_mib is not None:
                    recording = dataclasses.replace(
                        original.recording, trace_budget_mib=args.trace_budget_mib
                    )
            kwargs = {"destination": args.destination, "execution": execution}
            if recording is not None:
                kwargs["recording"] = recording
            changes = {
                "max_" + key: getattr(args, "max_" + key)
                for key in ("evolution_updates", "world_ticks", "chemical_updates")
                if getattr(args, "max_" + key) is not None
            }
            if changes:
                from .records import load_record_spec

                original = load_record_spec(_checkpoint_record(args.checkpoint))
                kwargs["limits"] = dataclasses.replace(original.limits, **changes)
            print(function(args.checkpoint, **kwargs))
        elif args.command == "run-batch":
            import tomllib

            from .batch import BatchRunner

            definition = tomllib.loads(args.config.read_text())
            if set(definition) != {"runs"} or not isinstance(definition["runs"], dict):
                raise ValueError("batch TOML requires [runs] mapping names to config paths")
            specs = {
                name: load_spec(args.config.parent / path)
                for name, path in definition["runs"].items()
            }
            if not specs:
                raise ValueError("batch TOML requires at least one run")
            with BatchRunner(next(iter(specs.values())).execution) as runner:
                outputs = runner.run_batch(specs, destination=args.destination, resume=args.resume)
                _json({name: str(path) for name, path in outputs.items()})
        elif args.command == "reproduce":
            check = verify_record(args.record)
            if not check["ok"]:
                raise ValueError("source record failed integrity: " + "; ".join(check["errors"]))
            print(_runner_function("reproduce")(args.record, args.destination))
        elif args.command == "convert-checkpoint":
            print(_runner_function("convert_checkpoint")(args.checkpoint, args.destination))
        elif args.command == "inspect":
            _json({"record": inspect_record(args.record), "integrity": verify_record(args.record)})
        elif args.command == "report":
            if args.compare:
                if not args.destination:
                    raise ValueError("--compare requires --destination")
                print(render_comparison(args.compare, args.destination))
            elif args.record:
                print(render_report(args.record, args.destination))
            else:
                raise ValueError("report requires a record or --compare")
        elif args.command == "workspace":
            if args.workspace_command == "init":
                print(init_workspace(args.path))
            elif args.workspace_command == "status":
                _json(list_records(args.path))
            else:
                _json(check_workspace(args.path))
        elif args.command == "tables":
            _tables(args)
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"toylife2: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
