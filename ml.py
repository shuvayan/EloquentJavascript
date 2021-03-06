import subprocess as sp
tmp = sp.call('cls',shell=True)


''' [SupL] Linear Regression (Least squares method)'''

#import numpy as np
#import pandas as pd
#import scipy.stats as stats
#import matplotlib.pyplot as plt
#import sklearn
#
### Exploring Boston Housing Dataset
## Predicting house price
#
#from sklearn.datasets import load_boston
#boston = load_boston()
#
##print(boston.data.shape)
##print(boston.feature_names)
##print(boston.DESCR)
#
#df = pd.DataFrame(boston.data)
##print(df.head(5))
#
## Lets define exact names of the features 
#df.columns = boston.feature_names
##print(df.head(5))
#
## The house price is defined inside target
##print(boston.target)
#
## Adding up the price column into our dataframe
#df['PRICE'] = boston.target
##print(df.head(5))
#
## Taking features other than target
#X = df.drop('PRICE', axis = 1)
#
## splitting data into train-test data (alternative -> model_selection)
#X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(
#        X, df.PRICE, test_size = 0.33, # Test data will be 33% of data
#        random_state = 42) # assign it to some values, to get same values on each fresh run
#
##print(X_train.shape)
#
#from sklearn.linear_model import LinearRegression
#
#lm = LinearRegression()
#
#lm.fit(X_train, Y_train)
#
#pred_train = lm.predict(X_train)
#pred_test = lm.predict(X_test)
#
#coeff_df = pd.DataFrame(X_train.columns, lm.coef_)
#
##print(coeff_df)
#
#
#plt.scatter(X_train.RM, Y_train)
##plt.scatter(pred_train, Y_train)
##plt.scatter(pred_test, Y_test)
#
### To calculate MSE
##print(np.mean((Y_train - pred_train)**2))
#
### Residual plots
### Less pattern -> less error
##plt.scatter(pred_train, pred_train - Y_train, c='b',s=10)
##plt.scatter(pred_test, pred_test - Y_test, c='g',s=10)
##plt.hlines(y = 0, xmin = 0, xmax = 50)
##plt.title("Blue (Training data) | Green (Test data)")



''' [SupL] Classification: Decision Trees '''

## Balance Scale Problem
## Tip to which direction, right/left/balanced ?
#
#import pandas as pd
#from sklearn.cross_validation import train_test_split
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score
#from sklearn import tree
#
#df = pd.read_csv(
#'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
#                           sep= ',', header= None)
#
##print(df.info)
#
#X = df.values[:, 1:5]   # Other features
#Y = df.values[:,0]  # Target
#
## Splitting data
#X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, 
#                                                    random_state = 42)
#
## Decision Tree Classifier
## criterion-> gini/entropy
#
#lm_gini = DecisionTreeClassifier(criterion = "gini", random_state = 42,
#                               max_depth=3, 
#                               min_samples_leaf=5) # min. samples req. at leaf node
#lm_gini.fit(X_train, y_train)
#y_pred = lm_gini.predict(X_test)
#
##print(lm_gini)
### Prediction of target values
##print(y_pred)
#
## converting into the pdf file
#with open('lm_gini.dot', "w") as f:
#    f = tree.export_graphviz(lm_gini, out_file=f)
#
## install graphviz package
## run -> dot -Tps lm.dot -o lm.png
#
#lm_ig = DecisionTreeClassifier(criterion = "entropy", random_state = 42,
#                               max_depth=3, 
#                               min_samples_leaf=5)
#lm_ig.fit(X_train, y_train)
#y_pred_ig = lm_ig.predict(X_test)
#
### Prediction of target values
##print(y_pred_ig)
#
## Predicting the accuracy of both the models
#
#print("Accuracy of Gini Index model: ", 
#      accuracy_score(y_test, y_pred) * 100)
#
#print("Accuracy of Information Gain model: ", 
#      accuracy_score(y_test, y_pred_ig) * 100)



''' [SupL] Random Forest '''

# They overcome simple decision trees in overfitting issues & correcting bias

# Iris Dataset

#from sklearn.datasets import load_iris
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.cross_validation import train_test_split
#import pandas as pd
#import numpy as np
#
#iris = load_iris()
#df = pd.DataFrame(iris.data, 
#                  columns=iris.feature_names)
#
## Adding a column to predict -> Species
#df['species'] = pd.Categorical.from_codes(
#                    iris.target, iris.target_names)
#
## Making train-test data
#X_train, X_test, Y_train, Y_test = train_test_split( df.iloc[:,0:4], df.iloc[:,4], 
#                                                    test_size = 0.3, random_state = 42)
#                                   
#features = df.columns[:4]
##print(features.shape)
#
## Changing flower species names to digits
#target = pd.factorize(df['species'])[0]
##print(target)
#
#forest = RandomForestClassifier(n_jobs = 2,
#                                random_state = 42)
#
#forest.fit(X_train, Y_train)
#forest.predict(X_test)
#
### Predicted Probability of first 5 observations
### With 3 species of flower, (0, 1, 0) suggest that there is 100% prob.
### that certain flower belongs to class B
##print(forest.predict_proba(X_test)[0:5])
#
### Importance of each feature
##print(list(zip(X_train, forest.feature_importances_)))




''' [UnSupL]Clustering '''


''' K-Means clustering '''
## Each centroid defines one of the clusters.
## Each data point is assigned to its nearest centroid
## Centroids are recomputed by accounting for the mean 
##           of all the data points assigned to that cluster
#
#
#import pandas as pd
#import numpy as np
#from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
#
#df = pd.read_csv('https://raw.githubusercontent.com/datascienceinc/learn-data-science/master/Introduction-to-K-means-Clustering/Data/data_1024.csv')
##print(df.info)
#
## Converting 1 Column into 3 columns
#foo = lambda x: pd.Series([i for i in reversed(x.split('\t'))])
#df = df['Driver_ID\tDistance_Feature\tSpeeding_Feature'].apply(foo)
#
#df.columns = ['Speeding_Feature','Distance_Feature','Driver_ID']
##print (df.head(5))
#
#f1 = df['Distance_Feature'].values#[0:200]
#f2 = df['Speeding_Feature'].values#[0:200]
#
#
#X = pd.DataFrame({'dist': f1,
#                  'speed':f2})
##print(X)
#
#kmeans = KMeans(n_clusters = 4) # don't exceed 4 else increase colormap    
#
#k_fit = kmeans.fit(X)
#pred = kmeans.predict(X)
#
#centroids = kmeans.cluster_centers_
##print(centroids)
#
#colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'k'}
#colors = list(map(lambda x: colmap[x+1], pred))
#
#plt.scatter(f1, f2, color = colors,
#            alpha = 0.4, edgecolor = 'k')
#
#plt.scatter(centroids[:, 0], centroids[:, 1], color = 'y')
#
#plt.show()



''' [UnSupL] Hierarchical Clustering '''
# Ward -> Looks for spherical or similar size clusters
# Complete -> Looks for similar observations making more compact cluster
# Average -> Looks for centroids


#import pandas as pd
#import numpy as np
#from sklearn.cluster import AgglomerativeClustering
#import matplotlib.pyplot as plt
#
#df = pd.read_csv('https://raw.githubusercontent.com/datascienceinc/learn-data-science/master/Introduction-to-K-means-Clustering/Data/data_1024.csv')
##print(df.info)
#
## Converting 1 Column into 3 columns
#foo = lambda x: pd.Series([i for i in reversed(x.split('\t'))])
#df = df['Driver_ID\tDistance_Feature\tSpeeding_Feature'].apply(foo)
#
#df.columns = ['Speeding_Feature','Distance_Feature','Driver_ID']
##print (df.head(5))
#
#f1 = df['Distance_Feature'].values[0:200]
#f2 = df['Speeding_Feature'].values[0:200]
#
#
#X = pd.DataFrame({'dist': f1,
#                  'speed':f2})
##print(X)
#
#hclus = AgglomerativeClustering(n_clusters = 4,
#                                affinity = 'euclidean',
#                                linkage = 'ward')
#
#f = hclus.fit(X)


''' Hierarchical Clustering: Dendograms '''

#from matplotlib import pyplot as plt
#from scipy.cluster.hierarchy import dendrogram, linkage
#import numpy as np
#
## Generating 2 clusters
#np.random.seed(4711)  # for repeatability of this tutorial
#a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
#b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
#X = np.concatenate((a, b),)
##print X.shape  # 150 samples with 2 dimensions
#plt.scatter(X[:,0], X[:,1])
#plt.show()


## Linkage matrix
#L = linkage(X, 'ward')
#
#plt.figure()
#
#dendrogram(
#    L,
#    leaf_font_size=8.,  # font size for the x axis labels
#)
#plt.show()



''' [UnSupL] Dimensionality Reduction using PCA '''

## PCA reduces the dimensionality of feature space by restricting attention to 
## those directions along which the scatter of the cloud is greatest.
#
#
#import matplotlib.pyplot as plt
#
#from sklearn.datasets import fetch_olivetti_faces
#from sklearn import decomposition
#
#n_row, n_col = 2, 3
#n_components = n_row * n_col
#image_shape = (64, 64)
#
#dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
#faces = dataset.data
#
#n_samples, n_features = faces.shape
#
## global centering
#faces_centered = faces - faces.mean(axis=0)
#
## local centering
#faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
#
#print("Dataset consists of %d faces" % n_samples)
#
#
#def plot_gallery(title, images, n_col=n_col, n_row=n_row):
#    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
#    plt.suptitle(title, size=16)
#    for i, comp in enumerate(images):
#        plt.subplot(n_row, n_col, i + 1)
#        vmax = max(comp.max(), -comp.min())
#        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
#                   interpolation='nearest',
#                   vmin=-vmax, vmax=vmax)
#        plt.xticks(())
#        plt.yticks(())
#    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
#
#
#estimators = [
#    ('Eigenfaces - PCA using randomized SVD',
#     decomposition.PCA(n_components=n_components, svd_solver='randomized',
#                       whiten=True),
#     True)]
#
#
#plot_gallery("First centered Olivetti faces", faces_centered[:n_components])
#
#
#for name, estimator, center in estimators:
#    estimator.fit(faces_centered)
#    
#    if hasattr(estimator, 'cluster_centers_'):
#        components_ = estimator.cluster_centers_
#    else:
#        components_ = estimator.components_
#
#    plot_gallery(name, components_[:n_components])
#
#plt.show()
