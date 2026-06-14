from __future__ import annotations

import sys
from collections.abc import Sequence

from simple_parsing import ArgumentParser

from trustport.wheelhouse.commands import (
    AppraiseArgs,
    EvaluateArgs,
    ExportArgs,
    FuseArgs,
    InspectArgs,
    TrainArgs,
    run_appraise,
    run_evaluate,
    run_export,
    run_fuse,
    run_inspect,
    run_train,
)

_VERBS = ("train", "appraise", "fuse", "inspect", "evaluate", "export")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in _VERBS:
        sys.stderr.write(
            "usage: trustport {train|appraise|fuse|inspect|evaluate|export} [options]\n"
        )
        return 1
    verb, rest = args[0], args[1:]
    parser = ArgumentParser()
    if verb == "train":
        parser.add_arguments(TrainArgs, dest="opts")
        run_train(parser.parse_args(rest).opts)
    elif verb == "appraise":
        parser.add_arguments(AppraiseArgs, dest="opts")
        run_appraise(parser.parse_args(rest).opts)
    elif verb == "fuse":
        parser.add_arguments(FuseArgs, dest="opts")
        run_fuse(parser.parse_args(rest).opts)
    elif verb == "inspect":
        parser.add_arguments(InspectArgs, dest="opts")
        run_inspect(parser.parse_args(rest).opts)
    elif verb == "evaluate":
        parser.add_arguments(EvaluateArgs, dest="opts")
        run_evaluate(parser.parse_args(rest).opts)
    else:
        parser.add_arguments(ExportArgs, dest="opts")
        run_export(parser.parse_args(rest).opts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
