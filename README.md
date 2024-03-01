We will build up the Fisher Linear Discriminant step by step and generalize it for multiple classes. 
We will first recap the simplest case, K = 2. In general, we can take any D-dimensional input vector and project it down to D′-dimensions.  
Here, D represents the original input dimensions while D′ is the projected space dimensions. For our purpose, consider D′ less than D.

If we want to project the data into D′ = 1 dimension, we can pick a threshold t to separate the classes in the new space. Given an input vector x:
->x belongs to class C1 (class 1), if predicted y >= t, where y = wT x

->otherwise, it is classified as C2 (class 2).

In figure below, the original data with D = 2 and we want to reduce the original data dimensions from D = 2 to D′ = 1. First we can compute the mean 
vectors of the two classes and consider using the class means as a measure of separation. In other words, we want to project the data onto the vector w joining the 2 class means.

![image](https://github.com/koushik16/Fisher-Linear-Discriminent-for-Dimentionality-Reduction/assets/63333977/f932cdd7-91fa-4409-90f8-9be38d2d3d79)
/nFigure: FLD projection


Note that during such projection,  there can be information loss.  In our case,  clearly in the original space, the two classes can be seperated by a line. However, after projection, the yellow ellipse 
demonstrates over- lapping of the two classes. For FLD, it looks for a large variance among the dataset classes and a smallvariance within each of the dataset classes.  The project w  could be:  w ∝ Sw−1(m2 − m1) and please refer to our slides for the meaning of each symbol and index here.

The above is a short summary of what we discussed during our lecture for 2 classes case. Now we will generalize the model to K > 2 classes. Here, we need generalization forms for the within-class and between- class covariance matrices, which is similar to the definitions for the two classes case:
•	Mean vector:  mk  =    1/Nk (Σi ∈ Cxi)
•	Within class variance:  Sk  = Σi∈Ck (xi − mk)(xi − mk)^T
•	Total within class variance: SW = ΣK	Sk
•	Between class variance: SB = ΣK	Nk(mk − m)(mk − m)^T
Then we can have the project vector:
w = maxD′ (eig(S−1SB)) - (1)
For between-class covariance SB estimation, for each class k = 1, 2, 3, ..., K, take the outer product of the local class mean mk and the global mean m, and then scale it by the number of instances in class k.

The maximization of the criterion of FLD, we need to solve w with an eigendecomposition of the matrix multiplication between S−1 and SB. To find the projection vector w, we then need to take the D′ eigen- vectors that correspond to their largest eigenvalues. 
For example, if we want to reduce our input dimension from D = 784 to D′ = 2. Now we have all the necessary settings for building a FLD for multiple classes. To create a discriminant, we first can use a multivariate Gaussian distribution over a D-dimensional input vector x for each class K as:
N(x|µ, Σ) =	1/(2π)D/2 * |Σ|^1/2   exp− 1/2 (x − µ)T Σ−1(x − µ) - (2)
 
Here µ (the mean) is a D−dimensional vector.  Σ (sigma) is a D × D  covariance matrix.  |Σ| is the determinant of the covariance.

Using the projected input data, the parameters of the Gaussian distribution µ and Σ can be computed for each of the class k = 1, 2, 3, ..., k. And the class priors P(Ck) probabilities can be calculated simply by the fractions from the training dataset for each class.

Once we have the Gaussian parameters and also the priors for each class, then we can estimate the class- conditional densities P(x|Ck, µk, Σk) for each class k: For each data point x, we first project it from D space to D′ then we evaluate N(x|µ, Σ).  By Bayes’s rule, then we can get the posterior class probabilities P(Ck|x) for each class k = 1, 2, 3, ..., k using
p(Ck|x) = p(x|Ck)P(Ck)	(3)
We then can assign the input data x to the class k ∈ K with the largest posterior.
Given all the instructions above, we can try on a dataset called MNIST, which has D = 784 dimensions. We want to do a projection to D′ = 2 or D′ = 3.

We have Implemented the following to demonstrate FDA: 
-> Implement the FLD for high-dimension to low-dimension projection with multivariant Gaussian.
-> Split the MNIST dataset into training and testing. And report the accuracy on test set with D = 2 and D′ = 3.
-> Make two plots for your testing data with a 2D and 3D plots. You can use different colors on the data points to indicate different classes. See if your results make sense to you or not.
