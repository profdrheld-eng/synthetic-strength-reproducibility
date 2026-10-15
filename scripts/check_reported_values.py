"""Check all 145 fixed artificial manuscript entries, without participant data."""
import argparse
import json
import math
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    expected=json.loads((root/'reference/artificial_reported_values.json').read_text())
    maximum=0.
    for row in expected['values']:
        path=args.output/row['source']
        value=json.loads(path.read_text())
        for key in row['keys']:value=value[key]
        wanted=row['value']
        if isinstance(wanted,(float,int)) and not isinstance(wanted,bool):
            error=abs(value-wanted)
            if not math.isfinite(error) or error>expected['absolute_tolerance']:
                raise AssertionError((row['label'],value,wanted))
            maximum=max(maximum,error)
        elif value!=wanted:raise AssertionError((row['label'],value,wanted))
    result={'passed':True,'artificial_entries':len(expected['values']),
            'maximum_absolute_difference':maximum,'absolute_tolerance':expected['absolute_tolerance'],
            'empirical_results_checked':False}
    (args.output/'REPORTED_VALUE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
