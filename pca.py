from scipy.linalg import eigh  
import numpy as np
import matplotlib.pyplot as plt  

def load_and_center_dataset(filename):
    x = np.load(filename) 
    x = np.reshape(x,(2000,784))
    x = x - np.mean(x, axis=0)
    return x

def get_covariance(dataset):
    return (np.dot(np.transpose(dataset), dataset) / (len(dataset) - 1))

def get_eig(S, m):
    w, v = eigh(S, subset_by_index=[len(S)-m, len(S)-1])
    i = np.flip(np.argsort(w))  
    return np.diag(w[i]), v[:, i]

def get_eig_perc(S, perc):
    w, v = eigh(S)
    percent = np.sum(w) * perc
    new_w, new_v = eigh(S, subset_by_value=[percent, np.inf])
    i = np.flip(np.argsort(new_w))
    return np.diag(new_w[i]), new_v[:, i]

def project_image(image, U):
    alphas = np.dot(np.transpose(U), image)
    return np.dot(U, alphas)

def display_image(orig, proj):
    orig = np.reshape(orig, (28,28))
    proj = np.reshape(proj, (28,28))

    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (9,3))
    ax1.set_title('Original')
    ax2.set_title('Projection')

    ax1Map = ax1.imshow(orig, aspect = 'equal', cmap='gray')
    fig.colorbar(ax1Map, ax=ax1)
    ax2Map = ax2.imshow(proj, aspect = 'equal', cmap='gray')
    fig.colorbar(ax2Map, ax=ax2)
    plt.show()


def main():
    x = load_and_center_dataset('mnist.npy')
    '''print(len(x))
    print(len(x[0]))
    print(np.average(x))'''

    ##x = np.array([[1,2,5],[3,4,7]])
    #x = get_covariance(x)
    ##print(x)
    #print(x[350][350])

    #x = load_and_center_dataset('mnist.npy')
    S = get_covariance(x)

    Lambda, U = get_eig(S, 20)

    '''print(Lambda)
    print(np.sum(U))

    Lambda, U = get_eig_perc(S, 0.07)
    print(Lambda)
    print(np.sum(U))'''

    proj = project_image(x[3],U)

    display_image(x[3], proj)

if __name__ == '__main__':
    main() 
