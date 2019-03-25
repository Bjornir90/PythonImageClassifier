import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.neighbors import KNeighborsClassifier
import time



def barycentre(x, y, dev, devLabel):

    classe=[]
    moyenne=[]

    # calculating barycentres
    for i in range(10):
        classe.append(x[y == i])
        moyenne.append(np.mean(classe[i], axis=0))

    result = []

    #guessing result by calculating the closest barycentre for every datapoint
    for j in range(5000):
        minimumDistance = 10000000
        minimumIndex = 0
        for i in range(10):
            distance = np.linalg.norm(dev[j] - moyenne[i])
            if distance < minimumDistance:
                minimumIndex = i
                minimumDistance = distance
        result.append(minimumIndex)

    finalResult = result != devLabel


def svm(newX, y, newDev, devLabel, t):


    clf = LinearSVC(random_state=0, tol=t)

    clf.fit(newX, y)

    svmResult = clf.predict(newDev)
    finalResult = svmResult != devLabel



def nu(newX, y, newDev, devLabel):
    clNu = NuSVC(gamma='scale')

    clNu.fit(newX, y)

    nuResult = clNu.predict(newDev)

    finalResult = nuResult != devLabel


def kneighbors(X, Y, dev, devLabel):
    neighbors = KNeighborsClassifier(n_neighbors=10)
    neighbors.fit(X, Y)

    neighborsResult = neighbors.predict(dev)
    finalResult = neighborsResult != devLabel

    print("Taux d'images non reconnues par le k-neighbors : ")
    print(finalResult[finalResult].size / finalResult.size)




def main():
    X = np.load('images_et4/data/trn_img.npy')
    Y = np.load('images_et4/data/trn_lbl.npy')
    dev = np.load('images_et4/data/dev_img.npy')
    devLabel = np.load('images_et4/data/dev_lbl.npy')

    print("Résultat sans PCA : ")
    startTime = time.process_time()
    kneighbors(X, Y, dev, devLabel)
    endTime = time.process_time()
    print("Exécuté en ", endTime-startTime)

    pca = PCA(n_components=163)
    reducedX = pca.fit_transform(X)
    reducedDev = pca.transform(dev)

    print("Résultat avec PCA : ")
    startTime = time.process_time()
    kneighbors(reducedX, Y, reducedDev, devLabel)
    endTime = time.process_time()
    print("Exécuté en ", endTime - startTime, " Dimension 163")


main()


