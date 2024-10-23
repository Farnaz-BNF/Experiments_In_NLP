import numpy as np

# getting the xnli results
def get_results_xnli():

    for k in [0, 10, 50, 100, 200, 500, 1000]:
        all_accs = []
        for lang in ["en", "es", "de"]:
            try:
                path = (f"/data/eval_xlmr_xnli_retrain_{k}_k_shortest_{lang}_3e-5_1.0_3/eval_results_test.txt")
                with open(path,"r") as f:
                    line = f.readline()

                    all_accs.append(float(line.split("acc = ")[1].strip()))
            except Exception as e:
                print(str(k) + "\t" + "\t".join([str(acc) for acc in all_accs]))
                all_accs = []
                break
        print(str(k) + "\t" + "\t".join([str(acc) for acc in all_accs]))

def get_results_xnli_average_iterations():

    for k in [10, 50, 100, 200, 500, 1000]:
        all_accs = []
        if k==0:
            iterations = [1]
        else:
            iterations = [1, 2, 3, 4, 5]
        for iteration in iterations:
            all_accs_for_iteration = []
            for lang in ["en", "es", "de"]:
                try:
                    path = (f"/data/eval_xlmr_xnli_retrain_{k}_k_shortest_{lang}_3e-5_1.0_{iteration}/eval_results_test.txt")
                    with open(path, "r") as f:
                        line = f.readline()
                        all_accs_for_iteration.append(float(line.split("acc = ")[1].strip()))
                except Exception as e:
                    #print(str(k) + "\t" + "\t".join([str(acc) for acc in all_accs]))
                    #all_accs = []
                    break
            if len(all_accs_for_iteration) == len(["en", "es", "de"]):
                np.array(all_accs_for_iteration)
                all_accs.append(all_accs_for_iteration)
        avg_accs = np.average(all_accs, axis=0)
        std_accs = np.std(all_accs, axis=0)
        avg_std_accs = list(zip(*(avg_accs,std_accs)))
        print(str(k) + "\t" + "\t".join([str(acc[0]) + "\t" + str(acc[1]) for acc in avg_std_accs]))


def main():
    get_results_xnli_average_iterations()

if __name__=="__main__":
    main()