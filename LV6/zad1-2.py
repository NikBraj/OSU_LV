import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

# Zad 6.5.1
def do_KNN(neighbour: int):
    KNN_model = KNeighborsClassifier(n_neighbors = neighbour)
    KNN_model.fit(X_train_n, y_train)
    y_test_p_KNN = KNN_model.predict(X_test_n)
    y_train_p_KNN = KNN_model.predict(X_train_n)

    print(f"KNN: {neighbour}")
    print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
    print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

    plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
    plt.tight_layout()
    plt.show()

do_KNN(5)
do_KNN(1) # underfit
do_KNN(100) # overfit

# KNN_model = KNeighborsClassifier(n_neighbors=7)
# KNN_model.fit(X_train_n, y_train)
# y_test_p_KNN = KNN_model.predict(X_test_n)
# y_train_p_KNN = KNN_model.predict(X_train_n)

# model = KNeighborsClassifier()
# scores = cross_val_score(KNN_model, X_train, y_train, cv=5)
# print(scores)

# array = np.arange(1, 101)
# param_grid = {'n_neighbors':array}
# knn_gscv = GridSearchCV(model, param_grid , cv=5, scoring ='accuracy', n_jobs =-1)
# knn_gscv.fit(X_train, y_train)

# k_range = range(1, 100)
# cv_scores = []

# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')  # 5-fold CV
#     cv_scores.append(scores.mean())

# optimal_k = k_range[np.argmax(cv_scores)]
# print(f"Optimalna vrijednost K je: {optimal_k}")

# plt.plot(k_range, cv_scores)
# plt.xlabel('Vrijednost K za KNN')
# plt.ylabel('Preciznost (Cross-Validation)')
# plt.title('Odabir optimalnog K')
# plt.show()

param_grid = {'n_neighbors': np.arange(1, 101)}  # K od 1 do 100

# KNN model + GridSearchCV s 5-fold cross-validacijom
model = KNeighborsClassifier()
knn_gscv = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
knn_gscv.fit(X_train_n, y_train)

# Rezultati
optimal_k = knn_gscv.best_params_['n_neighbors']
best_score = knn_gscv.best_score_

print("-Unakrsna validacija-")
print(f"Optimalni broj susjeda (K): {optimal_k}")
print(f"Najbolja tocnost (cross-val): {best_score:.3f}")

# Treniraj KNN s optimalnim K
KNN_best = KNeighborsClassifier(n_neighbors=optimal_k)
KNN_best.fit(X_train_n, y_train)

# Evaluacija
y_test_p_best = KNN_best.predict(X_test_n)
y_train_p_best = KNN_best.predict(X_train_n)

print("KNN s optimalnim K:")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_best)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_p_best)))

# Prikaz granice odluke
plot_decision_regions(X_train_n, y_train, classifier=KNN_best)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title(f"K={optimal_k}, Tocnost train: {accuracy_score(y_train, y_train_p_best):.3f}")
plt.tight_layout()
plt.show()

# Dodatno: Graf točnosti za sve K vrijednosti
results = pd.DataFrame(knn_gscv.cv_results_)
plt.plot(param_grid['n_neighbors'], results['mean_test_score'])
plt.xlabel("Broj susjeda K")
plt.ylabel("Tocnost (cross-val)")
plt.title("Odabir optimalnog K za KNN")
plt.grid()
plt.show()
