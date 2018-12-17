from sklearn.datasets import make_classification
from clustering_algorithm import Kmeans_algo
from clustering_algorithm import GMM_algo
from main_K_clustering import run
from sklearn.metrics import confusion_matrix, adjusted_mutual_info_score
from pdb import set_trace
from multiprocessing import Pool
from random import randint
from numpy import zeros, savetxt
from os import system


def function_to_map(l):
    return(run(l[0], l[1], n_save = 100000))

N_VIEWS = 3
N_CLUSTERS = 3
N_ITER = 30
N_CLASSES = 3

arguments = {
        'n_samples': 1000,
        'n_features': 10*N_VIEWS,
        'n_informative': 4*N_VIEWS,
        'n_redundant': 4*N_VIEWS,
        'n_classes': N_CLASSES
}
X = make_classification(**arguments)
X = [X[0][:, i*10:(i+1)*10] for i in range(N_VIEWS)]

if __name__ == '__main__':
    dirname = "clusters{}_classes{}_iter{}".format(N_CLUSTERS, N_CLASSES, 
            N_ITER)
    system("mkdir " + dirname)
    p = Pool(3)
    pool = []
    for _ in range(N_ITER):
        algos = [Kmeans_algo(N_CLUSTERS, random_state=randint(0,10000)) for _ in 
                range(N_VIEWS)]
        pool.append((X, algos))
    results = p.map(function_to_map, pool)
    for i in range(N_ITER):
        fname = dirname + '/clustering-'+str(i)+'-iter.csv'
        savetxt(fname, results[i], delimiter=',')

    scores = zeros((N_ITER, N_ITER))
    for i in range(N_ITER):
        for j in range(N_ITER):
            if i > j:
                score = 0
                for l in range(N_VIEWS):
                    score = score + adjusted_mutual_info_score(results[i][l], 
                            results[j][l])
                scores[i,j] = score / N_VIEWS

    fname = dirname + '/scores_mutual_info-'+str(N_ITER)+'-iter.csv'
    savetxt(fname, scores, delimiter=',')

    print("Score moyen : {}".format(scores[scores.nonzero()].mean()))

