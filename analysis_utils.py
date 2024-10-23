import lang2vec.lang2vec as l2v
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv

def detect_null_columns(vecs_matrix):
    nulls=[]
    for i, vec in enumerate(vecs_matrix):
        for j, val in enumerate(vec):
            if val == "--":
                nulls.append(j)
    return list(set(nulls))

def get_langs_for_task(task):
    if task == "xnli":
        langs = ["en", "es", "de"]
    return langs

def load_lang2vec_vectors(task="xnli", features=[]):
    "returns a matrix of lang vecs"
    vecs_matrix = []
    langs = get_langs_for_task(task)
    try:
        lang2vec = l2v.get_features(langs, features, minimal=True)
        for lang in langs:
            vecs_matrix.append(lang2vec[lang])
        null_columns = detect_null_columns(vecs_matrix)
        if len(null_columns) == len(vecs_matrix[0]):
            return None
        elif len(null_columns) > 0:
            vecs_matrix = np.delete(vecs_matrix, null_columns, axis=1)
        assert len(vecs_matrix) == len(langs)
        return np.array(vecs_matrix)
    except Exception as e:
        return None

def load_sheet_scores_from_experiment(strategy, k, task="xnli", path="./results_lower_level/lang-distance-results - ", model="mbert"):

    path = path + task + "_" + model + ".tsv"
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        scores_dict = {}
        langs = []
        for line in reader:
            if line[0] == strategy and task in ["xnli"]:
                langs = [lang for lang in line[1:] if lang != '']
            elif len(langs) > 0 and line[0] == str(k):
                scores_dict = {langs[int(i/2)]: float(line[i]) for i in range(len(line)) if i > 0 and i % 2 == 1}
                break
        langs = get_langs_for_task(task)
        scores = [scores_dict[lang] for lang in langs]
        return scores

def load_average_scores_from_experiment(task="xnli", model="mbert", sampling_strategy="k_first", k=0):
    all_accs = []
    langs = get_langs_for_task(task)
    if k == 0:
        iterations = [1]
    else:
        iterations = [1, 2, 3, 4, 5]
    for iteration in iterations:
        all_scores_for_iteration = []
        for lang in langs:
            try:
                if task == "xnli":
                    if model == "mbert":
                        path = (f"/data/eval_xnli_retrain_{k}_{sampling_strategy}_{lang}_3e-5_1.0_{iteration}/eval_results_test.txt")

                    with open(path, "r") as f:
                        line = f.readline()
                        all_scores_for_iteration.append(float(line.split("acc = ")[1].strip()))

            except Exception as e:
                print(e)
                break
        if len(all_scores_for_iteration) == len(langs):
            np.array(all_scores_for_iteration)
            all_accs.append(all_scores_for_iteration)
        else:
            raise ValueError()
    avg_accs = np.average(all_accs, axis=0)
    #std_accs = np.std(all_accs, axis=0)
    return avg_accs

def compute_similarities_of_lang_vecs(vecs_matrix, strategy="to_en"):
    similarities = cosine_similarity(vecs_matrix, vecs_matrix)
    if strategy == "to_en":
        return similarities[0]
    else:
        raise ValueError()

def get_experiment_scores(task, model, sampling_strategy, k):
    if task == "xnli":
        scores = load_average_scores_from_experiment(task, model, sampling_strategy, k)
    return scores

def mbert_input_corpus_sizes(langs):
    """
    source: size of wikipedias before 4th of November 2018 (first commit on multilingual readme.md)
    so from the history here https://en.wikipedia.org/w/index.php?title=List_of_Wikipedias&offset=20190902142837&action=history
    we get the 21th of October 2018: https://en.wikipedia.org/w/index.php?title=List_of_Wikipedias&oldid=865111482
    and choose to use num_articles
    :param langs:
    :return:
    """
    sizes = {
        "en": 6066642,
        "sv": 3731442,
        "de": 2425587,
        "fr": 2207291,
        "ru": 1617641,
        "it": 1601868,
        "es": 1593742,
        "vi": 1244270,
        "ja": 1202398,
        "zh": 1114453,
        "ar": 1040809,
        "ko": 492690,
        "fi": 483562,
        "eu": 354975,
        "tr": 349750,
        "he": 263993,
        "bg": 261622,
        "hr": 217323,
        "el": 177284,
        "ur": 153480,
        "hi": 137998,
        "th": 136996,
        "sw": 58397
    }

    return [sizes[lang] for lang in langs]