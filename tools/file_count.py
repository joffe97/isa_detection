from pathlib import Path
from statistics import fmean

labels_raw = [
    "	6502.corpus	8	LE	8-32	CISC",
    "	68HC08.corpus	8	BE	8-16	CISC",
    "	68HC11.corpus	8	BE	8-40	CISC",
    "	8051.corpus	32	LE	8-128	CISC",
    "arm64	ARM64.corpus	64	LE	32	RISC",
    "	ARMeb.corpus	32	BE	32	RISC",
    "armel	ARMel.corpus	32	LE	32	RISC",
    "armhf	ARMhf.corpus	32	LE	32	RISC",
    "	ARcompact.corpus	32	LE	16-32	RISC",
    "	AVR.corpus	8	LE	16-32	RISC",
    "alpha	Alpha.corpus	64	LE	32	RISC",
    "	AxisCris.corpus	32	LE	16	RISC",
    "	Blackfin.corpus	32	LE	16-32	RISC",
    "	CLIPPER.corpus	32	LE	2-8	RISC",
    "	CUDA.corpus	32	LE	32	NA",
    "	Cell-SPU.corpus	32	BE	32	",
    "	CompactRISC.corpus	16	LE	16	RISC",
    "	Cray.corpus	64	NA		",
    "	Epiphany.corpus	32	LE	16-32	RISC",
    "	FR-V.corpus	32	NA		RISC/VLIW",
    "	FR30.corpus	32	BE	16	RISC",
    "	FT32.corpus	32	NA		RISC",
    "	H8-300.corpus	8	BE	8-16	CISC",
    "	H8S.corpus	16	BE		CISC",
    '	HP-Focus.corpus	32	NA		"stack-oriented"',
    "hppa	HP-PA.corpus	64	BE	32	RISC",
    "	IA-64.corpus	64	LE	128	EPIC",
    "	IQ2000.corpus	32	BE		RISC",
    "	M32C.corpus	32	NA		CISC",
    "	M32R.corpus	32	BI	16-32	RISC",
    "m68k	M68k.corpus	32	BE		CISC",
    "	M88k.corpus	32	BI	32	RISC",
    "	MCore.corpus	32	BE	16	RISC",
    "mips64el		64	LE	32	RISC",
    "	MIPS16.corpus	16	BI	16	RISC",
    "mips	MIPSeb.corpus	32	BE	32	RISC",
    "mipsel	MIPSel.corpus	32	LE	32	RISC",
    "	MMIX.corpus	64	BE	32	RISC",
    "	MN10300.corpus	32	LE		",
    "	MSP430.corpus	16	LE		RISC",
    "	Mico32.corpus	32	BE	32	RISC",
    "	MicroBlaze.corpus	32/64	BI	32	RISC",
    "	Moxie.corpus	32	BI	32-48	",
    "	NDS32.corpus	32/16	BI	16-32	CISC",
    "	NIOS-II.corpus	32	LE	32	RISC",
    "	OCaml.corpus	NA	NA		",
    "	PDP-11.corpus	16	LE	16	CISC",
    "	PIC10.corpus	8	LE		RISC",
    "	PIC16.corpus	8	LE		RISC",
    "	PIC18.corpus	8	LE		RISC",
    "	PIC24.corpus	16	LE	24	RISC",
    "ppc64	PPCeb.corpus	64	BE		RISC",
    "ppc64el	PPCel.corpus	64	LE		RISC",
    "riscv64	RISC-V.corpus	64	LE	32	RISC",
    "	RL78.corpus	16	LE		CISC",
    "	ROMP.corpus	32	BE	8-32	RISC",
    "	RX.corpus	16/32/64	LE		CISC",
    "s390x		64	BE		CISC",
    "s390	S-390.corpus	32	BE		CISC",
    "sparc		32	BE	32	RISC",
    "sparc64	SPARC.corpus	64	BE	32	RISC",
    "	STM8.corpus	8			CISC",
    "	Stormy16.corpus	16	LE		",
    "sh4	SuperH.corpus	32	BI		RISC",
    "	TILEPro.corpus	32			RISC",
    "	TLCS-90.corpus	8			",
    "	TMS320C2x.corpus	16/32			",
    "	TMS320C6x.corpus	32	BI		VLIW",
    "	TriMedia.corpus				",
    "	V850.corpus	32			RISC",
    "	Visium.corpus	32			",
    '	WASM.corpus	32	LE		"stack based"',
    "	WE32000.corpus	32			",
    "amd64	X86-64.corpus	64	LE	8-120	CISC",
    "i386	X86.corpus	32	LE	8-120	CISC",
    "	Xtensa.corpus	32	BI	16-24	",
    "	Z80.corpus	8	LE	8-32	CISC",
    "	i860.corpus	32/64	BI		RISC",
    "ia64		64	LE	128	EPIC",
    "x32		32	LE		CISC",
    "powerpc		32	BE	32	RISC",
    "powerpcspe		32	BE	32	RISC",
    "	78k.corpus	8 or 16			CISC",
]

labels = dict(
    (
        item_split[0],
        {
            "endianness": item_split[3],
            "instruction_size": item_split[4],
        },
    )
    for item in labels_raw
    if (item_split := item.split("\t"))[0]
)

datasets_dir = Path(
    "/home/joachan/isa_detection/data/datasets/isa-detect-data/new_new_dataset"
)
subdir_name = "binaries"
# subdir_name = "binaries_code_sections_only"

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
    print(
        f"{arch}:    \t{len(files)}\t{fmean([entry.stat().st_size for entry in files])}"
    )

print()
print(
    f"Matching archs: {len(set(arch_binary_mapping.keys()).intersection(labels.keys()))}"
)
print()

endianness_counts_dict = dict(
    (
        isafeaturename,
        dict(
            (
                _class,
                [
                    len(arch_binary_mapping[arch])
                    for arch, item in labels.items()
                    if item[isafeature] in _class.split(",")
                ],
            )
            for _class in classes
        ),
    )
    for isafeature, isafeaturename, classes in [
        ("endianness", "Endianness", ["BE,LE", "BE", "LE"]),
        (
            "instruction_size",
            "Instruction size",
            [
                values["instruction_size"]
                for values in labels.values()
                if values["instruction_size"]
            ],
        ),
        (
            "instruction_size",
            "Fixed/variable instruction size",
            ["32,128,8-120", "32,128", "8-120", "32", "128"],
        ),
    ]
)

print(
    "\n".join(
        f"{isafeature}: {'   '.join([f'{_class}={sum(class_values)}' for _class, class_values in classes.items()])}"
        for isafeature, classes in endianness_counts_dict.items()
    )
)

# print(
#     f"Endianness: total={endianness_total_count} be={endianness_total_count_be} le={endianness_total_count_le}"
# )
