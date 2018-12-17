from clustering_algorithm import Kmeans_algo
from clustering_algorithm import GMM_algo
import numpy as np
from sklearn.metrics import confusion_matrix
import random
import time
from itertools import repeat
from pdb import set_trace


def K_int(n):
    return np.floor(np.log2(n))

def run(X, algos, n_save = 50):
    J = len(algos)
    S = [] # solutions for each algo
    N = len(X)
    
    # Local:
    print("Local phase")
    for i in range(J):
        print("Train algo " + str(i))
        S.append(algos[i].fit_predict(X[i]))
        print("Done")
    
    
    # Global:
    print("Collab phase")
    rules, exceptions = compute_global_rules(S, J)
    continue_loop = True
    newS = S
    newExceptions = exceptions
    n_iter = 1
    while continue_loop:
        print("Iter " + str(n_iter))
        newS, newExceptions, continue_loop = \
        global_optimization_greedy_with_exceptions(X, newS, rules, newExceptions, J, N, algos)
        n_iter += 1
        # if (n_iter % n_save) == 0:
            # for j in range(J):
                # fname = 'temp-' + str(j)
                # np.savetxt(fname, newS[j], delimiter=',')
            # print('Temporary files saved')
    
    return newS

def compute_global_rules(S, J):
    rules = [[[] for j in range(J)] for i in range(J)]
    exceptions = [[{} for j in range(J)] for i in range(J)]
    n_clusters = [1 + max(S[i]) for i in range(J)]
    
    for i in range(J):
        for j in range(J):
            if i == j: continue
            rules[i][j], exceptions[i][j], _ = \
            compute_solutions_complexity(S[i], S[j], n_clusters[i], n_clusters[j])
    return rules, exceptions


def global_optimization_greedy_with_exceptions(X, S, rules, exceptions, J, N, algos, random_seed=42):
    # Build rules and exceptions
    n_clusters = [1 + max(S[i]) for i in range(J)]
    
    best_error = None
    max_Delta_K = 0
    
    random.seed(random_seed)
    
    for i in random.sample(range(J), J):
        for j in random.sample(range(J), J):            
            for n in exceptions[i][j]:                
                # Correct value by changing Si[n] = rule[Sj[n]]
                newValue = rules[i][j][S[j][n]]
                newExceptions = [[dict(exceptions[i][j]) for j in range(J)] for i in range(J)]
                
                # How much does it cost to correct this error in K(S_i | S_j)?
                x = X[i][n].reshape(1, -1)
                delta_K = (J - 1) * (algos[i].complexity(x, np.array([newValue])) \
                - algos[i].complexity(x, np.array([S[i][n]])))

                for l in range(J):
                    if l == i: continue
                    
                    # Impact on K(S_l | S_i)
                    if n in newExceptions[l][i]:
                        if rules[l][i][newValue] == S[l][n]:
                            # Remove exception[l][i][n]
                            delta_K -= K_int(N) + K_int(n_clusters[l])
                            del newExceptions[l][i][n]
                        else:
                            newExceptions[l][i][n] = newValue
                    else:
                        delta_K += K_int(N) + K_int(n_clusters[l])
                        newExceptions[l][i][n] = newValue
                        
                    # Impact on K(S_i | S_l)
                    
                    if n in newExceptions[i][l]:
                        if rules[i][l][S[l][n]] == newValue:
                            # Previously: exception for S[l][n] -> S[i][n]
                            # Now: no exception => remove exception[i][l][n]
                            delta_K -= K_int(N) + K_int(n_clusters[i])
                            del newExceptions[i][l][n]
                        else:
                            newExceptions[i][l][n] = newValue
                    else: 
                        delta_K += K_int(N) + K_int(n_clusters[i])
                        newExceptions[i][l][n] = newValue
                
                if delta_K < max_Delta_K:
                    best_error = [i, n, newValue]
                    max_Delta_K = delta_K
                    best_exceptions = newExceptions
    
    if best_error is None:
        # Current configuration is the best. No change has to be done!
        return S, exceptions, False
    else:
        S_modif = [np.array(S[i]) for i in range(J)]
        i, n, newValue = best_error
        S_modif[i][n] = newValue
        return S_modif, best_exceptions, True


def compute_solutions_complexity(S1, S2, K1, K2):
    # K(S1 | S2)
    cnf_matrix = confusion_matrix(S2, S1)
    rules = cnf_matrix.argmax(1)
    complexity = K2 * (K_int(K1) + K_int(K2))
    
    # Search of exceptions
    exceptions = {}
    for i in range(S1.shape[0]):
        if S1[i] != rules[S2[i]]:
            exceptions[i] = S1[i]
            complexity += K_int(S1.shape[0]) + K_int(K1)
    return rules, exceptions, complexity



###########################"

def run_files_kmeans(files, n_cluster, n_save=10):
    X = [np.genfromtxt(file, delimiter=',') for file in files]
    algos = [Kmeans_algo(n_cluster) for _ in range(len(X))]
    start_time = time.time()
    S = run(X, algos, n_save=n_save)
    print('-- %s seconds --' % (time.time() - start_time))
    for i in range(len(S)):
        fname = files[i].split('.')[0] + '_solution-kmeans.csv'
        np.savetxt(fname, S[i], delimiter=',')
    print('Solution files saved')

def run_files_gmm(files, n_cluster, n_save=10):
    X = [np.genfromtxt(file, delimiter=',') for file in files]
    algos = [GMM_algo(n_cluster) for _ in range(len(X))]
    start_time = time.time()
    S = run(X, algos, n_save=n_save)
    print('-- %s seconds --' % (time.time() - start_time))
    for i in range(len(S)):
        fname = files[i].split('.')[0] + '_solution-gmm.csv'
        np.savetxt(fname, S[i], delimiter=',')
    print('Solution files saved')


def run_files_both(files, n_cluster, n_save=10):
    X = [np.genfromtxt(file, delimiter=',') for file in files]
    J = len(X)
    algos_knn = [Kmeans_algo(n_cluster) for _ in range(len(X))]
    algos_gmm = [GMM_algo(n_cluster) for _ in range(len(X))]
    algos = algos_knn + algos_gmm
    X = [x for item in X for x in repeat(item, 2)]
    start_time = time.time()
    S = run(X, algos, n_save=n_save)
    print('-- %s seconds --' % (time.time() - start_time))
    for i in range(J):
        fname = files[i].split('.')[0] + '_solution-both-1.csv'
        np.savetxt(fname, S[2 * i], delimiter=',')
        fname = files[i].split('.')[0] + '_solution-both-2.csv'
        np.savetxt(fname, S[2 * i + 1], delimiter=',')
    print('Solution files saved')



# files = ['data/spam2/spam1.csv', 'data/spam2/spam2.csv', 'data/spam2/spam3.csv']
# n_clusters = 2

# run_files_gmm(files, n_clusters, n_save=20)
# run_files_both(files, n_clusters, n_save=20)
