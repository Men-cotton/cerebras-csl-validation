"""Stable uv module entrypoint for operation, discovery and offline checks."""
import sys


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in {"-h", "--help"}:
        print("Usage: uv run --locked -m csl_validation {watch|fetch|jobs|check} ...")
        return 0
    if args[0] == "watch":
        from .watch import main as run
    elif args[0] == "fetch":
        from .fetch import main as run
    elif args[0] == "jobs":
        from .catalog import main as run
    elif args[0] == "check":
        from .checks import main as run
    else:
        print("unknown command: " + args[0], file=sys.stderr)
        return 2
    return run(args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
