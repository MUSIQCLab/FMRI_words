

# currently does not work to calculate w - soemthing wrong with my lasso function
def crossValidationPCA (dimensions):
    pca = PCA(n_components=dimensions)
    pca.fit(fmri_train)
    xtrainpcaed= pca.transform(fmri_train)
    xtrainPCA = sparse.csc_matrix (xtrainpcaed)
    xtest = pca.transform (fmri_test)
    num_features = ytrain.shape[1]
    d = xtrainPCA.shape[1]
    ntotdata = xtrainPCA.shape[0]
    bestw = np.zeros([num_features,d])
    lasso = lassoSolver.LassoClass()
    bestw0 = np.zeros([num_features,d])


    for i in range(2): #num_features):
        print ('looking at feature ', i)
        #bestlambda = lasso.findLambdaCrossValidation(ytrain[:,i].reshape(ntotdata,1),xtrainPCA,5)
        bestlambda = 142.5
        [w_0,w_new,y_hat]  = lasso.cordDescentLasso (ytrain[:,i].reshape(ntotdata,1),xtrainPCA, bestlambda)
        bestw[i,:] = w_new.reshape(d)
        bestw0 [i,:] = w_0
    wfile = "w_crossvalidationlambda_pca300.mtx"
    w0file = "w0_crossvalidationlambda_pca300.mtx"
    io.mmwrite(wfile, bestw)
    io.mmwrite(w0file, bestw0)
    test_suite.main(bestw,bestw0,wordid_train,wordid_test,wordfeature_std,xtest)
    return [accuracy,bestw,w0, pca]