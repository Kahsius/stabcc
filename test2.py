from os import system
from random import randint
from pdb import set_trace
from multiprocessing import Pool

from numpy import zeros, savetxt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, adjusted_mutual_info_score

from clustering_algorithm import Kmeans_algo, GMM_algo
from main_K_clustering import run


# Fonction utilitaire permettant de lancer la parallelisation
def function_to_map(l):
    return(run(l[0], l[1], n_save = 100000))

# Hyperparametres
N_VIEWS = 3
N_CLUSTERS = 3
N_CLASSES = 3
N_SAMPLES_MIN = 100
N_SAMPLES_MAX = 300
N_SAMPLES_STEP = 100
N_ITER = int((N_SAMPLES_MAX - N_SAMPLES_MIN ) /N_SAMPLES_STEP)

# Hyperparametres par vue
N_FEATURES = 10 # Nombre total de features (informative + redundant + noise)
N_INFORMATIVE = 4 # Nombre de features utiles
N_REDUNDANT = 4 # Nombre de features redondants par rapport a N_INFORMATIVE


if __name__ == '__main__':

    # Creation de l'algo
    dirname = "test2\clusters{}_classes{}_samples{}-{}".format(N_CLUSTERS, 
            N_CLASSES, N_SAMPLES_MIN, N_SAMPLES_MAX)
    system("mkdir " + dirname)
    seed = random_state=randint(0,10000)
    algos = [Kmeans_algo(N_CLUSTERS, seed) for _ in range(N_VIEWS)]

    # Apprentissage des algos 
    p = Pool(3)
    pool = []
    for N_SAMPLES in range(N_SAMPLES_MIN, N_SAMPLES_MAX, N_SAMPLES_STEP):
        # Generation des donnees
        arguments = {
                'n_samples': N_SAMPLES,
                'n_features': N_FEATURES*N_VIEWS,
                'n_informative': N_INFORMATIVE*N_VIEWS,
                'n_redundant': N_REDUNDANT*N_VIEWS,
                'n_classes': N_CLASSES
        }
        X = make_classification(**arguments)
        X = [X[0][:, i*10:(i+1)*10] for i in range(N_VIEWS)]
        pool.append((X, algos))
    results = p.map(function_to_map, pool)

    # Sauvegarde des clusterings intermediaires
    for i in range(N_ITER):
        fname = dirname + '/clustering-'+str(i)+'-iter.csv'
        savetxt(fname, results[i], delimiter=',')

    # Calcul des scores par paire de clustering
    # TODO: prévoir le test pour des nombres d'individus différents
    # scores = zeros((N_ITER, N_ITER))
    # for i in range(N_ITER):
    #     for j in range(N_ITER):
    #         if i > j:
    #             score = 0
    #             for l in range(N_VIEWS):
    #                 score = score + adjusted_mutual_info_score(results[i][l], 
    #                         results[j][l])
    #             scores[i,j] = score / N_VIEWS

    # Sauvegarde des scores finaux
    # fname = dirname + '/scores_mutual_info-{}-{}-samples.csv'.format( 
    # N_SAMPLES_MIN, N_SAMPLES_MAX)
    # savetxt(fname, scores, delimiter=',')

    # print("Score moyen : {}".format(scores[scores.nonzero()].mean()))

