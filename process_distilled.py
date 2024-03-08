import os
from pathlib import Path

result_dir = Path("data/amazon_reviews/distilled/few-shot/")
output_path = Path("data/amazon_reviews/distilled/few-shot.txt")

# files = os.listdir(result_dir)
# files = [int(file.split(".")[0]) for file in files]
# files = [str(file).zfill(5) + ".txt" for file in sorted(files)]

files = sorted(os.listdir(result_dir))
with open(output_path, "a") as out_file:
    for file in files:
        print(file)
        with open(result_dir / Path(file), "r") as in_file:
            lines = in_file.readlines()
        lines = [line.strip().lower() for line in lines if line.strip()]
        line = " ".join(lines)
        out_file.write(line + "\n")
