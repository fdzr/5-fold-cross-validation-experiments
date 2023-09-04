import pandas as pd
from pprint import pprint
from ast import literal_eval as F


def transform_file():
    data = pd.read_csv("v2/results_new_dataset/spectral_clustering.csv")

    # print(
    #     data[
    #         [
    #             "number_cluster_selected_by_calinski",
    #             "number_cluster_selected_by_calinski",
    #         ]
    #     ]
    # )
    # duplicated_col = data.columns[data.columns.duplicated()]
    # data.drop(columns=duplicated_col, inplace=True)
    drop_columns = []

    for index in range(1, 6):
        for cv in ["silhouette", "calinski", "eigengap"]:
            drop_columns.append(f"number_cluster_selected_by_{cv}_r{index}.1")

    data.drop(columns=drop_columns, inplace=True)
    data.to_csv(
        "v2/results_new_dataset/spectral_clustering_.csv",
        header=True,
        index=False,
    )


def check_avg():
    data = pd.read_csv(f"v2/results_new_dataset/spectral_clustering.csv")
    pprint(
        data[
            [
                "number_cluster_selected_by_silhouette_r1",
                "number_cluster_selected_by_silhouette_r2",
                "number_cluster_selected_by_silhouette_r3",
                "number_cluster_selected_by_silhouette_r4",
                "number_cluster_selected_by_silhouette_r5",
            ]
        ].mean(axis=1)
    )


if __name__ == "__main__":
    check_avg()
