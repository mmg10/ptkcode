# !/usr/bin/env/python3

from pathlib import Path
import torchvision
import torch
from sklearn.model_selection import train_test_split
import numpy as np


def preprocess():
    output_path = "."

    Path(output_path).mkdir(parents=True, exist_ok=True)

    trainset = torchvision.datasets.CIFAR10(root="./", train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root="./", train=False, download=True)

    Path(output_path + "/train").mkdir(parents=True, exist_ok=True)
    Path(output_path + "/val").mkdir(parents=True, exist_ok=True)
    Path(output_path + "/test").mkdir(parents=True, exist_ok=True)

    RANDOM_SEED = 25

    y = trainset.targets
    trainset, valset, y_train, y_val = train_test_split(
        trainset, y, stratify=y, shuffle=True, test_size=0.2, random_state=RANDOM_SEED
    )

    yv = valset.targets
    yt = testset.targets
    y_array = np.array(y)

    y_train_array = np.array(y)
    y_val_array = np.array(yv)
    y_test_array = np.array(yt)

    label_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    label_counts = dict(zip(*np.unique(y_array, return_counts=True)))  # fix this
    train_label_counts = dict(zip(*np.unique(y_train_array, return_counts=True)))
    val_label_counts = dict(zip(*np.unique(y_val_array, return_counts=True)))
    test_label_counts = dict(zip(*np.unique(y_test_array, return_counts=True)))

    label_counts = [
        label_counts,
        train_label_counts,
        val_label_counts,
        test_label_counts,
    ]

    label_dict = {}
    TOTAL_COUNT = len(y)
    rows = []
    # for key, value in label_counts.items():
    #     print(
    #         "Label Counts of [{}]({}) : {}".format(key, label_names[key].upper(), value)
    #     )
    #     label_dict[label_names[key].upper()] = int(value)
    #     row = f"| {key} | {value} |"
    #     rows.append(row)

    for label in label_counts[0].items():
        counts = " | ".join(str(d[label]) for d in label_counts)
        row = f"| {label} | {counts} |"
        rows.append(row)

    # rows.append(f"| {TOTAL_COUNT} | {value} |")

    header = "| Label | Total Count | Train Count | Val Count | Test Count |\n|-------|---------|---------|---------|-------|"
    table = "\n".join(rows)
    markdown_table = f"{header}\n{table}"

    # label_dict["TOTAL_COUNT"] = int(TOTAL_COUNT)

    metadata = {
        "outputs": [
            {
                "storage": "inline",
                "source": """# Model Overview
    ## Label Count
    {}
    """.format(
                    markdown_table
                ),
                "type": "markdown",
            },
        ]
    }

    return metadata

    # (
    #     label_names,
    #     label_counts,
    #     train_label_counts,
    #     val_label_counts,
    #     test_label_counts,
    # )
