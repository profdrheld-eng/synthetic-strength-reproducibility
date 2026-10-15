"""Generate invented cohorts only. No input-data argument exists."""
import argparse
from pathlib import Path

from strength_privacy.population import MODELS, NOISE
from .runner import run, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohorts', type=int, default=1)
    parser.add_argument('--releases', type=int, default=1)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--knowledge', nargs='+', choices=NOISE, default=list(NOISE))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new path')
    result = run(models=tuple(args.models), cohorts=args.cohorts, releases=args.releases,
                 levels=tuple(args.knowledge))
    write_report(result, args.output)
    print(f"ARTIFICIAL_ONLY: {len(result['records'])} joint release records written.")


if __name__ == '__main__':
    main()
