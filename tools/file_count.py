from pathlib import Path
from statistics import fmean


datasets_dir = Path("/home/joachan/isa_detection/data/datasets/isa-detect-data/new_new_dataset")
# subdir_name = "binaries"
subdir_name = "binaries_code_sections_only"

subdir_path = datasets_dir.joinpath(subdir_name)

arch_binary_mapping = {}
for dir_entry in subdir_path.iterdir():
    if not dir_entry.is_dir():
        continue
    arch_name = dir_entry.name
    arch_binary_mapping[arch_name] = [
        entry
        for entry in dir_entry.iterdir()
        if not (entry.suffix.endswith("json") or entry.suffix.endswith("hex"))
    ]

total_count = sum(list(map(len, arch_binary_mapping.values())))
print(f"Total count for {subdir_name}: {total_count}")

for arch, files in sorted(arch_binary_mapping.items()):
    print(f"{arch}:    \t{len(files)}\t{fmean([entry.stat().st_size for entry in files])}")
