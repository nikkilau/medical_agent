from micro_mdt import cli as _impl
from micro_mdt.cli import *  # noqa: F401,F403


def main(argv=None):
    _impl.load_env_file = load_env_file
    return _impl.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
