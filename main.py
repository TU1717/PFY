"""Command-line entry point for the public Gumbel-PFY runner."""

import argparse
import json
from pathlib import Path
from typing import Dict

from trainer import build_parser, run_experiment


def load_config(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError("Config file not found: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("The config file must contain a JSON object")
    defaults = value.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError("config.json must define a defaults object")
    return defaults


def main() -> None:
    default_path = Path(__file__).with_name("config.json")
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default=str(default_path))
    known, _ = preliminary.parse_known_args()

    defaults = load_config(Path(known.config))

    parser = build_parser(default_config=str(default_path))
    valid_keys = {action.dest for action in parser._actions}
    unknown = sorted(set(defaults) - valid_keys)
    if unknown:
        raise ValueError("Unknown config keys: {}".format(", ".join(unknown)))
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
