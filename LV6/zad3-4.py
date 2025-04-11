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

# Zad 3
def do_SVM(c, gamma):
    SVM_model = svm.SVC(kernel ='rbf', gamma = gamma, C=c)
    SVM_model.fit(X_train_n, y_train)

    y_test_p_SVM = SVM_model.predict(X_test)
    y_train_p_SVM = SVM_model.predict(X_train)

    print(f"SVM: C:{c} Gamma: {gamma}")
    print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
    print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_SVM))))

    plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc='upper left')
    plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
    plt.tight_layout()
    plt.show()

do_SVM(0.1, 1)
do_SVM(10, 1)
do_SVM(0.1, 100)

# Zadatak 4

param_grid_svm = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10, 100]
}

# Model + GridSearchCV
svm_model = svm.SVC(kernel='rbf')
svm_gscv = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
svm_gscv.fit(X_train_n, y_train)

# Najbolji parametri
best_C = svm_gscv.best_params_['C']
best_gamma = svm_gscv.best_params_['gamma']
best_score_svm = svm_gscv.best_score_

print("-Unakrsna validacija SVM-")
print(f"Optimalni parametri: C={best_C}, gamma={best_gamma}")
print(f"Najbolja tocnost (cross-val): {best_score_svm:.3f}")

# Treniraj SVM s najboljim parametrima
SVM_best = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma)
SVM_best.fit(X_train_n, y_train)

# Evaluacija
y_train_p_svm_best = SVM_best.predict(X_train_n)
y_test_p_svm_best = SVM_best.predict(X_test_n)

print("SVM s optimalnim parametrima:")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_svm_best)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_p_svm_best)))

# Prikaz granice odluke
plot_decision_regions(X_train_n, y_train, classifier=SVM_best)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title(f"SVM: C={best_C}, gamma={best_gamma}, Tocnost train: {accuracy_score(y_train, y_train_p_svm_best):.3f}")
plt.tight_layout()
plt.show()

means = svm_gscv.cv_results_['mean_test_score']
Cs = svm_gscv.cv_results_['param_C'].data
gammas = svm_gscv.cv_results_['param_gamma'].data

# Nacrtajmo ovisnost točnosti o kombinaciji C i gamma
plt.figure(figsize=(10, 5))
for g in sorted(set(gammas)):
    accs = [means[i] for i in range(len(means)) if gammas[i] == g]
    Cs_g = [Cs[i] for i in range(len(means)) if gammas[i] == g]
    plt.plot(Cs_g, accs, marker='o', label=f'gamma = {g}')

plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('Tocnost (CV srednja)')
plt.title('Tocnost SVM modela za razlicite C i gamma')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()