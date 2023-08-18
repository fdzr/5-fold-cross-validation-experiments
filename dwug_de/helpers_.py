from ast import literal_eval as F
import json

import pandas as pd
from scipy.stats import spearmanr


def splitted_data(data: pd.DataFrame, number_of_words: int):
    n = data.shape[0]
    for index in range(int(n / number_of_words)):
        start = index * number_of_words
        end = start + number_of_words
        yield data.loc[start : end - 1, :]


def load_config_file():
    with open("config/methods.json", "r") as f_in:
        data = json.load(f_in)

    return data


def load_gold_data_semeval():
    data = pd.read_csv(
        "resources/dwug_de/misc/dwug_de_sense/stats/maj_3/stats_groupings.csv",
        sep="\t",
    )
    return data


def load_gold_change_graded_semeval():
    data = load_gold_data_semeval()
    gold_change_graded = (
        data[["lemma", "change_graded"]]
        .set_index("lemma")["change_graded"]
        .to_dict()
    )

    return gold_change_graded


def select_words(list_of_words, index_of_words):
    selected_words = []
    for index in index_of_words:
        selected_words.append(list_of_words[index])

    return selected_words


def prepare_data_for_spr(data, list_of_words, index_of_words):
    selected_words = select_words(list_of_words, index_of_words)
    gold_data = load_gold_change_graded_semeval()
    vector1 = [gold_data[word] for word in selected_words]
    vector2 = [round(float(val), 2) for val in data["avg_jsd"].to_list()]

    return vector1, vector2


def get_avg_ari_subset_words_sc(data, config):
    cluster_validation_method = ["silhouette", "calinski", "eigengap"]
    subset_data = data.copy()

    for validation_method in cluster_validation_method:
        subset_data[f"avg_ari_{validation_method}"] = subset_data[
            config["spectral_clustering"][f"avg_ari_{validation_method}"]
        ].mean(axis=1)
        subset_data[f"avg_ari_{validation_method}_old"] = subset_data[
            config["spectral_clustering"][f"avg_ari_{validation_method}_old"]
        ].mean(axis=1)
        subset_data[f"avg_ari_{validation_method}_new"] = subset_data[
            config["spectral_clustering"][f"avg_ari_{validation_method}_new"]
        ].mean(axis=1)

    return subset_data


def train_spectral_clustering_method(
    data: pd.DataFrame,
    config,
    training_set: list[int],
    list_of_words: list[str],
):
    best_result_ari = {
        "parameters_silhouette": {},
        "parameters_calinski": {},
        "parameters_eigengap": {},
        "parameters_silhouette_old": {},
        "parameters_calinski_old": {},
        "parameters_eigengap_old": {},
        "parameters_silhouette_new": {},
        "parameters_calinski_new": {},
        "parameters_eigengap_new": {},
        "ari_silhouette": 0.0,
        "ari_calinski": 0.0,
        "ari_eigengap": 0.0,
        "ari_silhouette_old": 0.0,
        "ari_calinski_old": 0.0,
        "ari_eigengap_old": 0.0,
        "ari_silhouette_new": 0.0,
        "ari_calinski_new": 0.0,
        "ari_eigengap_new": 0.0,
        "index_silhouette": -1,
        "index_calinski": -1,
        "index_eigengap": -1,
        "index_silhouette_old": -1,
        "index_calinski_old": -1,
        "index_eigengap_old": -1,
        "index_silhouette_new": -1,
        "index_calinski_new": -1,
        "index_eigengap_new": -1,
    }
    best_result_jsd = {"parameters": [], "jsd": 0.0, "index": -1}
    number_of_subset_of_words = -1
    cluster_validation_methods = ["silhouette", "calinski", "eigengap"]
    data = get_avg_ari_subset_words_sc(data, config)

    for set_of_words in splitted_data(data, 24):
        number_of_subset_of_words += 1
        subset_training = set_of_words.iloc[training_set].copy()
        parameters = F(subset_training["parameters_r1"].to_list()[0])
        parameters.pop("word", None)

        subset_training["avg_jsd"] = subset_training[
            config["test"]["avg_jsd"]
        ].mean(axis=1)

        gold_data_graded_change, avg_jsd = prepare_data_for_spr(
            subset_training, list_of_words, training_set
        )
        spr, _ = spearmanr(gold_data_graded_change, avg_jsd)
        if spr > best_result_jsd["jsd"]:
            best_result_jsd["jsd"] = spr
            best_result_jsd["index"] = number_of_subset_of_words
            best_result_jsd["parameters"] = parameters

        for method in cluster_validation_methods:
            avg = subset_training[f"avg_ari_{method}"].mean(axis=0)
            if avg > best_result_ari[f"ari_{method}"]:
                best_result_ari[f"ari_{method}"] = avg
                best_result_ari[f"index_{method}"] = number_of_subset_of_words
                best_result_ari[f"parameters_{method}"] = parameters

            avg = subset_training[f"avg_ari_{method}_old"].mean(axis=0)
            if avg > best_result_ari[f"ari_{method}_old"]:
                best_result_ari[f"ari_{method}_old"] = avg
                best_result_ari[
                    f"index_{method}_old"
                ] = number_of_subset_of_words
                best_result_ari[f"parameters_{method}_old"] = parameters

            avg = subset_training[f"avg_ari_{method}_new"].mean(axis=0)
            if avg > best_result_ari[f"ari_{method}_new"]:
                best_result_ari[f"ari_{method}_new"] = avg
                best_result_ari[
                    f"index_{method}_new"
                ] = number_of_subset_of_words
                best_result_ari[f"parameters_{method}_new"] = parameters

    return best_result_ari, best_result_jsd


def eval_spectral_clustering(
    data: pd.DataFrame,
    test_set: list[int],
    config,
    best_configuration_for_ari_training_set,
    best_configuration_for_jsd_training_set,
    target_words: list[str],
) -> tuple[dict, float]:
    start = best_configuration_for_jsd_training_set["index"] * 24
    end = start + 24
    subset_test_for_jsd = data.loc[start : end - 1, :].iloc[test_set].copy()
    subset_test_for_jsd["avg_jsd"] = subset_test_for_jsd[
        config["test"]["avg_jsd"]
    ].mean(axis=1)
    gold_change_graded, jsd = prepare_data_for_spr(
        subset_test_for_jsd, target_words, test_set
    )

    spr, _ = spearmanr(gold_change_graded, jsd)

    results_ari = {}

    cluster_validation_method = ["silhouette", "calinski", "eigengap"]

    for validation_method in cluster_validation_method:
        start = best_configuration_for_ari_training_set[
            f"index_{validation_method}"
        ]
        end = start + end
        subset_test_for_ari = (
            data.loc[start : end - 1, :].iloc[test_set].copy()
        )
        results_ari[f"avg_ari_{validation_method}"] = subset_test_for_ari[
            f"avg_ari_{validation_method}"
        ].mean(axis=0)

        start = best_configuration_for_ari_training_set[
            f"index_{validation_method}_old"
        ]
        end = start + end
        subset_test_for_ari = (
            data.loc[start : end - 1, :].iloc[test_set].copy()
        )
        results_ari[f"avg_ari_{validation_method}_old"] = subset_test_for_ari[
            f"avg_ari_{validation_method}_old"
        ].mean(axis=0)

        start = best_configuration_for_ari_training_set[
            f"index_{validation_method}_new"
        ]
        end = start + end
        subset_test_for_ari = (
            data.loc[start : end - 1, :].iloc[test_set].copy()
        )
        results_ari[f"avg_ari_{validation_method}_new"] = subset_test_for_ari[
            f"avg_ari_{validation_method}_new"
        ].mean(axis=0)

    return results_ari, spr


def train(
    method: str, training_set: list[int], target_words: list[str]
) -> tuple[dict, dict]:
    config = load_config_file()["5-fold-cv"]
    data = pd.read_csv(f"outputs/results_whole_dataset/{method}.csv")
    if method == "spectral_clustering":
        return train_spectral_clustering_method(
            data, config, training_set, target_words
        )

    best_result_ari = {
        "parameters": {},
        "ari": 0.0,
        "index": -1,
    }
    best_result_jsd = {"parameters": {}, "jsd": 0.0, "index": -1}
    number_of_subset_of_words = -1

    for set_of_words in splitted_data(data, 24):
        number_of_subset_of_words += 1
        subset_training = set_of_words.iloc[training_set].copy()
        subset_training["avg_ari"] = subset_training[
            config["test"]["avg_ari"]
        ].mean(axis=1)
        avg_ari_training_set = subset_training["avg_ari"].mean(axis=0)

        subset_training["avg_jsd"] = subset_training[
            config["test"]["avg_jsd"]
        ].mean(axis=1)

        gold_change_graded, jsd = prepare_data_for_spr(
            subset_training, target_words, training_set
        )
        spr, _ = spearmanr(gold_change_graded, jsd)

        if avg_ari_training_set > best_result_ari["ari"]:
            best_result_ari["ari"] = avg_ari_training_set
            parameters = F(subset_training["parameters_r1"].to_list()[0])
            parameters.pop("word", None)
            best_result_ari["parameters"] = parameters
            best_result_ari["index"] = number_of_subset_of_words

        if spr > best_result_jsd["jsd"]:
            best_result_jsd["jsd"] = spr
            parameters = F(subset_training["parameters_r1"].to_list()[0])
            parameters.pop("word", None)
            best_result_jsd["parameters"] = parameters
            best_result_jsd["index"] = number_of_subset_of_words

    return best_result_ari, best_result_jsd


def eval(
    method: str,
    test_set: list[int],
    best_configuration_for_ari_training_set,
    best_configuration_for_jsd_training_set,
    target_words: list[str],
) -> tuple[float, float]:
    config = load_config_file()["5-fold-cv"]
    data = pd.read_csv(f"outputs/results_whole_dataset/{method}.csv")

    if method == "spectral_clustering":
        data = get_avg_ari_subset_words_sc(data, config)
        return eval_spectral_clustering(
            data,
            test_set,
            config,
            best_configuration_for_ari_training_set,
            best_configuration_for_jsd_training_set,
            target_words,
        )

    start = best_configuration_for_ari_training_set["index"] * 24
    end = start + 24
    subset_test_for_ari = data.loc[start : end - 1, :].iloc[test_set].copy()
    subset_test_for_ari["avg_ari"] = subset_test_for_ari[
        config["test"]["avg_ari"]
    ].mean(axis=1)

    start = best_configuration_for_jsd_training_set["index"] * 24
    end = start + 24
    subset_test_for_jsd = data.loc[start : end - 1, :].iloc[test_set].copy()
    subset_test_for_jsd["avg_jsd"] = subset_test_for_jsd[
        config["test"]["avg_jsd"]
    ].mean(axis=1)

    gold_change_graded, jsd = prepare_data_for_spr(
        subset_test_for_jsd, target_words, test_set
    )

    spr, _ = spearmanr(gold_change_graded, jsd)
    ari = subset_test_for_ari["avg_ari"].mean(axis=0)

    return ari, spr


def get_fields_to_report_for_spectral_clustering():
    config = load_config_file()["5-fold-cv"]
    fields = config["spectral_clustering"].keys()
    results = {}

    for field in fields:
        results[field] = 0.0

    return results


def calculate_results_for_spectral_clustering_method(
    avg_ari_for_spectral_clustering: dict, result_avg_ari: dict
):
    config = load_config_file()["5-fold-cv"]
    fields = config["spectral_clustering"].keys()
    for field in fields:
        avg_ari_for_spectral_clustering[field] += result_avg_ari[field]


def calculate_average(
    results_per_method: dict, methods: list[str], k_fold: int
):
    for m in methods:
        results_per_method[m]["spr_lscd"] = float(
            results_per_method[m]["spr_lscd"] / k_fold
        )
        if m != "spectral_clustering":
            results_per_method[m]["ari"] = float(
                results_per_method[m]["ari"] / k_fold
            )
        else:
            config = load_config_file()["5-fold-cv"]
            fields = config["spectral_clustering"].keys()
            for field in fields:
                results_per_method[m]["ari"][field] = float(
                    results_per_method[m]["ari"][field] / k_fold
                )


def present_results(results_per_method: dict, methods: list[str]):
    config = load_config_file()["5-fold-cv"]
    fields = config["spectral_clustering"].keys()

    for m in methods:
        print(f"{m}")
        print(f"    Spr_LSCD: {results_per_method[m]['spr_lscd']}")

        if m != "spectral_clustering":
            print(f"    ARI: {results_per_method[m]['ari']}")
        else:
            for field in fields:
                print(
                    f"    {field.upper()}: {results_per_method[m]['ari'][field]}"
                )

        print()
