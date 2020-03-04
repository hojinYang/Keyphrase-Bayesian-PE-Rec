from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sparse
from scipy.sparse import vstack
import numpy as np

def puresvd1(matrix_train,
             iteration=10, rank=256, fb=False, seed=1, **unused):
    """
    PureSVD algorithm
    :param matrix_train: rating matrix
    :param iteration: number of random SVD iterations
    :param rank: SVD top K eigenvalue ranks
    :param fb: facebook package or sklearn package. boolean
    :param seed: Random initialization seed
    :param unused: args that not applicable for this algorithm
    :return:
    """

    matrix_input = matrix_train
 


    P, sigma, Qt = randomized_svd(matrix_input,
                                    n_components=rank,
                                    n_iter=iteration,
                                    power_iteration_normalizer='QR',
                                    random_state=seed)

    P = P*np.sqrt(sigma)
    Q = Qt.T*np.sqrt(sigma)

    return P, Q


def puresvd(matrix_train, embeded_matrix=np.empty((0)),
             iteration=10, rank=200, seed=1, **unused):
    """
    PureSVD algorithm
    :param matrix_train: rating matrix
    :param embeded_matrix: item or user embedding matrix(side info)
    :param iteration: number of random SVD iterations
    :param rank: SVD top K eigenvalue ranks
    :param fb: facebook package or sklearn package. boolean
    :param seed: Random initialization seed
    :param unused: args that not applicable for this algorithm
    :return:
    """
    #progress = WorkSplitter()
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    #progress.subsection("Randomized SVD")
    #start_time = time.time()


    P, sigma, Qt = randomized_svd(matrix_input,
                                    n_components=rank,
                                    n_iter=iteration,
                                    power_iteration_normalizer='QR',
                                    random_state=seed)

    P = P*np.sqrt(sigma)
    Q = Qt.T*np.sqrt(sigma)
    #print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    return P, Q.T, None