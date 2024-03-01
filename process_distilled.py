import os
from pathlib import Path

result_dir = Path("data/amazon_reviews/distilled/results")
output_path = Path("data/amazon_reviews/distilled.txt")

files = os.listdir(result_dir)
files = [int(file.split(".")[0]) for file in files]
files = [str(file).zfill(3) + ".txt" for file in sorted(files)]

with open(output_path, "a") as out_file:
    for file in files:
        with open(result_dir / Path(file), "r") as in_file:
            lines = in_file.readlines()
        lines = [line.strip().lower() for line in lines if line.strip()]
        line = " ".join(lines)
        out_file.write(line + "\n")
