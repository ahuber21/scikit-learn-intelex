import os
import re
from pathlib import Path


def main(fname: Path):
    # print(fname)
    with open(fname) as fp:
        lines = fp.readlines()

    legacy_found = False
    for line in lines:
        if "from pathlib import Path" in line:
            legacy_found = False
            break
        if "os.path.join" in line:
            legacy_found = True

    if not legacy_found:
        return

    with open(fname, "w") as fp:
        base_path_generated = False
        for line in lines:
            if "import daal4py as d4p" in line:
                fp.write("from pathlib import Path\n")

            if match := re.search(r"^(\s+)(\w+ = )\"(\./data/.*)\"$", line):
                print(match.groups())
                indent, variable, path = match.groups()
                data_dir = "/".join(path.split("/")[:-1])
                data_file = path.split("/")[-1]
                if base_path_generated is False:
                    data_dirs = data_dir.lstrip("./").split("/")
                    data_dirs_with_quotes = map(lambda x: f'"{x}"', data_dirs)
                    base_path = "data_path = Path(__file__).parent / " + " / ".join(
                        data_dirs_with_quotes
                    )
                    fp.write(indent + base_path + "\n")
                    base_path_generated = True
                fp.write(f'{indent}{variable}data_path / "{data_file}"\n')
            else:
                fp.write(line)


if __name__ == "__main__":
    for root, _, files in os.walk(
        "/localdisk2/mkl/huberand/scikit-learn-intelex/examples/daal4py"
    ):
        for f in files:
            if not f.endswith(".py"):
                continue
            main(Path(root) / f)
