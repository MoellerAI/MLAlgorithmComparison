from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster import hierarchy
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def prepare_data(df: pd.DataFrame, target_column: str, split: bool = True, train_size: int = 400):
    X, y = df.loc[:, df.columns != target_column], df.loc[:, df.columns == target_column]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size/df.shape[0], random_state=1, shuffle = True, 
                                                    stratify = y)
        # reshaping such that it matches the requirements of sklearn
        X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train).reshape(len(y_train,)), np.array(y_test).reshape(len(y_test,))
        return X_train, X_test, y_train, y_test
    else:
        X_train, y_test = np.array(X), np.array(y).reshape(len(y,))
        return X_train, y_test

def plot_decision_tree(classifier: DecisionTreeClassifier, df: pd.DataFrame, target_name: str):
    feature_names = list(df.columns[df.columns != target_name])
    class_names = list(df[target_name].unique())
    fig = plt.figure(figsize=(25,20))
    plot_tree(classifier, label = 'none', impurity = False, feature_names=feature_names,  
                   class_names=class_names, filled=False)

def plot_feature_importance(classifier: RandomForestClassifier, df: pd.DataFrame, target_name: str):
    fig = plt.figure(figsize=(20,15))
    df_temp = pd.DataFrame({'variables': list(df.columns[df.columns != target_name]), 'importance': classifier.feature_importances_})
    df_temp = df_temp.sort_values(by = 'importance', ascending = True)
    variable_names = list(df_temp.variables)
    plt.barh(variable_names, df_temp.importance)

def create_hierarchical_clustering(X, distance = 'euclidean', linkage = 'complete', distance_threshold = 0, n_clusters=None):
    model = AgglomerativeClustering(affinity=distance, distance_threshold=distance_threshold, n_clusters=n_clusters, linkage = linkage)
    model = model.fit(X)
    if distance == 'manhattan':
        distance_temp = 'cityblock'
    else:
        distance_temp = distance
    Z = hierarchy.linkage(X, method = linkage, metric = distance_temp)
    if n_clusters is not None:
        distance_threshold = model.distance_threshold
    print(f"linkage: {linkage}")
    print(f"distance: {distance}")
    print(f"distance threshold: {distance_threshold}")
    plt.figure(figsize=(7,6))
    dn = hierarchy.dendrogram(Z)
    if distance_threshold is not None:
        # Cutting the dendrogram at max_d
        plt.axhline(y=distance_threshold, c='k')
        plt.title(linkage + " + " + distance)

def optimal_amount_of_clusters(X):
    # calculate distortion for a range of number of cluster
    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(X)
        distortions.append(km.inertia_)
    # plot
    plt.figure(figsize=(10,5))
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()