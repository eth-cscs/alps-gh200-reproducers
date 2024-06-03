import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

# List of files to process
file_paths = [
    "data/job_n_00004_N_0001_TPN_4.out",
    "data/job_n_00004_N_0004_TPN_1.out",
    "data/job_n_00016_N_0004_TPN_4.out",
    "data/job_n_00016_N_0016_TPN_1.out",
    "data/job_n_00032_N_0008_TPN_4.out",
    "data/job_n_00032_N_0032_TPN_1.out",
    "data/job_n_00064_N_0016_TPN_4.out",
    "data/job_n_00064_N_0064_TPN_1.out",
    "data/job_n_00128_N_0032_TPN_4.out",
    "data/job_n_00128_N_0128_TPN_1.out",
    "data/job_n_00256_N_0064_TPN_4.out",
    "data/job_n_00256_N_0256_TPN_1.out",
    "data/job_n_00512_N_0128_TPN_4.out",
    "data/job_n_01024_N_0256_TPN_4.out",
]

# Output directory for the plots
output_dir = "plots"


# Function to parse integers from the filename
def parse_filename(file_path):
    filename = os.path.basename(file_path)
    pattern = re.compile(r"job_n_(\d+)_N_(\d+)_TPN_(\d+)\.out")
    match = pattern.match(filename)
    if match:
        n = int(match.group(1))
        N = int(match.group(2))
        TPN = int(match.group(3))
        return n, N, TPN
    else:
        raise ValueError(f"Filename {filename} does not match the expected pattern")


# Function to parse a single file and extract the blocks
def parse_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    # Regex pattern to match the blocks
    pattern = re.compile(
        r"={39}\ntype:\s*(\w+)\nsize:\s*([0-9.]+)\npad:\s*([0-9.]+)\nnaive:\s*([0-9.]+)\ntime:\s*([0-9.]+)\ntime0:\s*([0-9.]+)\nbw:\s*([0-9.]+)\nbw0:\s*([0-9.]+)\n={39}"
    )
    blocks = pattern.findall(content)

    # Extracting data into a list of dictionaries
    data = []
    for block in blocks:
        data.append(
            {
                "type": block[0],
                "size": float(block[1]),
                "pad": int(block[2]),
                "naive": int(block[3]),
                "time": float(block[4]),
                "time0": float(block[5]),
                "bw": float(block[6]),
                "bw0": float(block[7]),
            }
        )
    return data


# Function to generate block names based on type, naive, and pad
def create_names(block):
    base_name = f"{block['type']}_memory"
    block_name = base_name
    if block["naive"] == 1:
        block_name += "_naive"
        small_block_name = "naive"
    else:
        block_name += "_mpi"
        small_block_name = "mpi"
    if block["pad"] != 0:
        block_name += "_pad"
        small_block_name += "_pad"
    return base_name, block_name, small_block_name


def plot_data_2(grouped_lines, base_name, ns, tpn, pad, bw0, combined=False):
    colors = sns.color_palette("husl", len(grouped_lines) + 4)
    markers = ["o", "s", "D", "^", "v", "p", "*", "X"]
    markers = [
        "o",
        "p",
        "s",
        "D",
        "^",
        "v",
        ">",
        "<",
        "d",
        "P",
        "h",
        "x",
        "|",
        "H",
        "+",
        "_",
    ]
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    alpha = 0.7
    markersize = 10

    if combined:
        title = f"{base_name}: allreduce bus-bandwidth (4 GiB fp32)"
        filename = f"{base_name}"
    else:
        title = f"{base_name}: allreduce bus-bandwidth (4 GiB fp32, {tpn} ranks/node)"
        filename = f"{base_name}_{tpn}"
        if pad:
            title += ": padded MPI alloc"
            filename += "_pad"
    if bw0:
        title += ", with startup iteration"
        filename += "_startup"
        
    print(f"  plotting {title} and saving in {filename}")

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.ylabel("bandwidth [GB/s]")
    plt.xlabel("number of ranks")
    plt.xticks(sorted(set(ns)))
    plt.grid(True)
    plt.yscale("log")
    plt.xscale("log")

    idx = 0
    l = 0
    for small_block_name, data in grouped_lines.items():
        if combined or (
            data["tpn"] == tpn
            and (
                (pad and ("pad" in small_block_name))
                or (not pad and (not "pad" in small_block_name))
            )
        ):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            line_style = line_styles[idx % len(line_styles)]
            plt.plot(
                data["n"],
                data["bw"],
                color=color,
                marker=marker,
                linestyle=line_style,
                label=f"{small_block_name}",
                alpha=alpha,
                markersize=markersize,
            )
            l = l + 1
            if bw0:
                plt.scatter(
                    data["n"],
                    data["bw0"],
                    facecolors="none",
                    edgecolors=color,
                    marker=marker,
                    label=f"{small_block_name}_0",
                    alpha=alpha,
                    s=markersize * 10,
                )
                l = l + 1
        idx = idx + 1

    if base_name == "device_memory":
        plt.ylim([0.4, 500])
        plt.xlim([3, 1500])

        if combined or tpn == 4:
            n_nccl = [4, 16, 32, 64, 128, 256, 512, 1024]
            bw_nccl = [344.23, 91.29, 91.22, 91.36, 91.39, 91.43, 91.42, 91.08]
            plt.plot(
                n_nccl,
                bw_nccl,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linestyle=line_styles[idx % len(line_styles)],
                label=f"NCCL-Tests (ranks/node=4)",
                alpha=alpha,
                markersize=markersize,
            )
        idx = idx + 1

        if combined or tpn == 1:
            n_nccl = [4, 16, 32, 64, 128, 256]
            bw_nccl = [22.84, 22.87, 22.89, 22.90, 22.90, 22.92]
            plt.plot(
                n_nccl,
                bw_nccl,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linestyle=line_styles[idx % len(line_styles)],
                label=f"NCCL/uenv (ranks/node=1)",
                alpha=alpha,
                markersize=markersize,
            )
        idx = idx + 1

        if combined or tpn == 4:
            n_nccl = [4, 16, 32, 64]
            bw_nccl = [343.360, 91.740, 91.673, 91.646]
            plt.plot(
                n_nccl,
                bw_nccl,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linestyle=line_styles[idx % len(line_styles)],
                label=f"NCCL/container (ranks/node=4)",
                alpha=alpha,
                markersize=markersize,
            )
        idx = idx + 1

        if combined or tpn == 4:
            n_nccl = [4, 16, 32, 64, 128, 256]
            bw_nccl = [342.326, 91.305, 91.311, 91.331, 91.437, 91.462]
            plt.plot(
                n_nccl,
                bw_nccl,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linestyle=line_styles[idx % len(line_styles)],
                label=f"NCCL/uenv (ranks/node=4)",
                alpha=alpha,
                markersize=markersize,
            )
        idx = idx + 1

    # Add custom legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = list(
        dict.fromkeys(labels)
    )  # Remove duplicates while preserving order
    if bw0:
        unique_handles = [
            handles[i] for i in range(len(unique_labels)) if i % 2 == 0 or i >= l
        ]  # Keep every second handle
        unique_labels = [
            unique_labels[i] for i in range(len(unique_labels)) if i % 2 == 0 or i >= l
        ]  # Keep every second label
        plt.legend(
            unique_handles,
            unique_labels,
            fontsize="small",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
    else:
        unique_handles = [handles[i] for i in range(len(unique_labels))]
        unique_labels = [unique_labels[i] for i in range(len(unique_labels))]
        plt.legend(
            unique_handles,
            unique_labels,
            fontsize="small",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )

    # Use tight layout
    plt.tight_layout(rect=[0, 0, 1, 1])

    # Save the plot to a file
    plt.savefig(
        os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(output_dir, f"{filename}.svg"), dpi=300, bbox_inches="tight"
    )
    plt.close()


# Function to plot the data blocks and save to files
def plot_data(blocks, ns, tpns, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a dictionary to group blocks by their base names
    grouped_blocks = {}
    for file_blocks in blocks:
        for block in file_blocks:
            base_name, block_name, small_block_name = create_names(block)
            if base_name not in grouped_blocks:
                grouped_blocks[base_name] = []
            grouped_blocks[base_name].append(
                (
                    small_block_name,
                    block,
                    ns[blocks.index(file_blocks)],
                    tpns[blocks.index(file_blocks)],
                )
            )

    for base_name, block_data in grouped_blocks.items():
        print(f"plotting {base_name}")

        # Create a dictionary to group lines by their (short) block name
        grouped_lines = {}
        for idx, (small_block_name, block, n, tpn) in enumerate(block_data):
            small_block_name = f"{small_block_name} (ranks/node={tpn})"
            if small_block_name not in grouped_lines:
                grouped_lines[small_block_name] = {
                    "tpn": tpn,
                    "n": [],
                    "bw": [],
                    "bw0": [],
                }
            grouped_lines[small_block_name]["n"].append(n)
            grouped_lines[small_block_name]["bw"].append(block["bw"])
            grouped_lines[small_block_name]["bw0"].append(block["bw0"])

        # plot different selections

        plot_data_2(grouped_lines, base_name, ns, 4, False, False)
        plot_data_2(grouped_lines, base_name, ns, 4, False, True)
        plot_data_2(grouped_lines, base_name, ns, 4, True, False)
        plot_data_2(grouped_lines, base_name, ns, 4, True, True)

        plot_data_2(grouped_lines, base_name, ns, 1, False, False)
        plot_data_2(grouped_lines, base_name, ns, 1, False, True)
        plot_data_2(grouped_lines, base_name, ns, 1, True, False)
        plot_data_2(grouped_lines, base_name, ns, 1, True, True)

        plot_data_2(grouped_lines, base_name, ns, 0, True, True, True)
        plot_data_2(grouped_lines, base_name, ns, 0, True, False, True)


# Main function to process multiple files
def main(file_paths, output_dir):
    all_blocks = []
    ns = []
    tpns = []

    for file_path in file_paths:
        n, _, tpn = parse_filename(file_path)
        blocks = parse_file(file_path)
        ns.append(n)
        tpns.append(tpn)
        all_blocks.append(blocks)

    plot_data(all_blocks, ns, tpns, output_dir)


# Run the main function
main(file_paths, output_dir)
