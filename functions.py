import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time



def barycentre(x, y, dev, devLabel):

    classe=[]
    moyenne=[]

    #barycentres
    for i in range(10):
        classe.append(x[y == i])
        moyenne.append(np.mean(classe[i], axis=0))

    result = []

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
    print("Taux d'images non reconnues : ")
    # Most beautiful thing ever
    print(finalResult[finalResult].size/finalResult.size)


def svm(newX, y, newDev, devLabel, t):


    clf = LinearSVC(random_state=0, tol=t)
    print(clf.get_params())
    clf.fit(newX, y)
    svmResult = clf.predict(newDev)
    svmTrainResult = clf.predict(newX)
    finalResult = svmResult != devLabel
    finalTrainResult = svmTrainResult != y

    print("Taux d'images non reconnues par le svm : ")
    # Most beautiful thing ever
    print(finalResult[finalResult].size/finalResult.size)
    print("Taux d'erreurs sur l'ensemble de train : ")
    print(finalTrainResult[finalTrainResult].size/finalTrainResult.size)


def nu(newX, y, newDev, devLabel):
    clNu = NuSVC(gamma='scale')

    clNu.fit(newX, y)

    nuResult = clNu.predict(newDev)

    finalResult = nuResult != devLabel

    print("Taux d'image non reconnues par le svc nu : ")
    print(finalResult[finalResult].size/finalResult.size)

def kneighbors(X, Y, dev, devLabel):
    neighbors = KNeighborsClassifier(n_neighbors=10)
    neighbors.fit(X, Y)

    neighborsResult = neighbors.predict(dev)
    finalResult = neighborsResult != devLabel

    print("Taux d'images non reconnues par le k-neighbors : ")
    # Most beautiful thing ever
    print(finalResult[finalResult].size / finalResult.size)




def main():
    X = np.load('images_et4/data/trn_img.npy')
    Y = np.load('images_et4/data/trn_lbl.npy')
    dev = np.load('images_et4/data/dev_img.npy')
    devLabel = np.load('images_et4/data/dev_lbl.npy')
    timeList = []
    dimensionList = []

    for i in range(10, 311, 50):
        pca = PCA(n_components=i)
        newX = pca.fit_transform(X)
        newDev = pca.transform(dev)
        startTime = time.process_time()
        #barycentre(newX, Y, newDev, devLabel)
        #svm(newX, Y, newDev, devLabel, 100)
        #nu(newX, Y, newDev, devLabel)
        kneighbors(newX, Y, newDev, devLabel)
        endTime = time.process_time()
        dimensionList.append(i)
        timeList.append(endTime-startTime)
        print("Time to classify : ", endTime-startTime, "s with ", i, " dimensions")

    plt.plot(dimensionList, timeList)
    plt.show()


main()


