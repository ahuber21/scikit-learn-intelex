import os
import re
from pathlib import Path


def main(fname: Path):
    # print(fname)
    with open(fname) as fp:
        lines = fp.readlines()

    with open(fname, "w") as fp:
        skipping = False
        for line in lines:
            if "# let's try to use pandas' fast csv reader" in line:
                fp.write("from readcsv import pd_read_csv\n")
                skipping = True
            if skipping and line.startswith("def"):
                # found the next function definition, start writing again
                fp.write("\n\n")
                skipping = False
            if skipping:
                continue

            fp.write(line)


if __name__ == "__main__":
    for root, _, files in os.walk(
        "/localdisk2/mkl/huberand/scikit-learn-intelex/examples/daal4py"
    ):
        for f in files:
            if not f.endswith(".py"):
                continue
            main(Path(root) / f)
