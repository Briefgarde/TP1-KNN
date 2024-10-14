# TP 1 : KNN from scratch

In this TP, we will implement the KNN algorithm. KNN is a simple, lazy learning algorithm which uses the distance between the new, queried point and each points in the training dataset to make a prediction. 

## Steps
1. "Training" : The model memorize the entire dataset. 
2. "Prediction" : The model receive a new point (a query) and make a prediction.
   1. We calculate the distance between the query and every other points in the training dataset. This is done using the norm of the difference between the query and the points. We obtain a list of the points ranked by order of "closeness" to the query.
   2.  We take the **k** closest point in the training data set. **K** is an hyperparameter given when making a query. 
   3.  For regression : 
       1.  We take the target value of the **k** closest point and average them out. The average is our prediction. 
   4. For classification : 
      1. We take the target classes of the **k** closest point and choose the most common class. This is our prediction. Using an odd **k** (1,3,5,etc...) prevent having a tie.

## Questions : 

### Write on your report how to compute the difference between two vectors theoretically : 

$$Difference (a,b) = \textbf{a}-\textbf{b} = \left[  \begin{matrix}
a_1 \\
a_2\\
... \\
a_n
\end{matrix}\right] - 
\left[  \begin{matrix}
b_1 \\
b_2\\
... \\
b_n
\end{matrix}\right] = \left[ 
\begin{matrix}
a_1-b_1 \\
a_2-b_2\\
... \\
a_n-b_n
\end{matrix}
\right]$$

$$\textbf{a}-\textbf{b} = \left[  \begin{matrix}
3.0 \\
5.2
\end{matrix}\right] - 
\left[  \begin{matrix}
3.5 \\
1.3
\end{matrix}\right] = \left[ 
\begin{matrix}
3.0 - 3.5 \\
5.2 - 1.3
\end{matrix}
\right] = \left[ 
\begin{matrix}
-0.5 \\
3.9
\end{matrix}
\right]
$$

Another name for the difference between two vector is called the **displacement vector**. It's the vector that goes from the tip of one vector, to the tip of another vector (assuming they start from the origin point both). 

### Is there a difference between (point - query) or (query - point) ? If there is any, comment on the differences.

$$\textbf{b}-\textbf{a} = \left[  \begin{matrix}
3.5 \\
1.3
\end{matrix}\right] - 
\left[  \begin{matrix}
3.0 \\
5.2
\end{matrix}\right] = \left[ 
\begin{matrix}
3.5 - 3.0 \\
1.3 - 5.2
\end{matrix}
\right] = \left[ 
\begin{matrix}
0.5 \\
-3.9
\end{matrix}
\right]
$$
As we can see, the vector is similar, but its **signs are reversed**. This mean **its direction is the opposite of the first vector**. Here, in the KNN algorithm, it doesn't matter which we do, because we'll be computing the distance between the two points, and the equation for the distance ignore the direction. In other context, it may be important. 

### Do the same with the inner product between point and query
$$\mathbf{a}^\top \mathbf{b} = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

$$\mathbf{a}^\top \mathbf{b} = 
\left[ \begin{matrix} 
3.0 \\ 
5.2 
\end{matrix} \right]^\top
\left[ \begin{matrix} 
3.5 \\ 
1.3 
\end{matrix} \right]
= 3.0 \times 3.5 + 5.2 \times 1.3
= 10.5 + 6.76
= 17.26
$$

### Do the same with the norm between the difference of point and query.
The Norm, or Euclidean distance, is the distance of the vector, its length. It's close to saying "this vector is 3 meter/unit long", in a way 

$$\|\mathbf{x}\| = \sqrt{x_1^2 + x_2^2+ \cdots + x_n^2 + } = \sqrt{\mathbf{x}^\top \mathbf{x}}
$$

In KNN, we aren't interested in the norm of the point vector, or the query vector. We only care about the norm of the difference between the point and the query (or the query and the point, it's equal), so we write it like this if the difference vector is **d** :

$$\text{dist}(p, q)) = \|\mathbf{p} - \mathbf{q}\| = \|\mathbf{q} - \mathbf{p}\| = \|\mathbf{d}\|
$$
$$\|\mathbf{d}\| = \sqrt{d_1^2 + d_2^2+ \cdots + d_n^2 + } = \sqrt{\mathbf{d}^\top \mathbf{d}}
$$
$$
\|\mathbf{d}\| = \sqrt{0.5^2 + (-3.9)^2} = \sqrt{0.25 + 15.21} = \sqrt{15.46} \approx 3.93
$$

### Why do we need the norm for knn ?
The norm is the measure that allow us to evaluate the distance, and thus the closeness, of any given point with the query. We can use this measure to rank the points in the training set, and then choose the k nearest to make our prediction. 

### Is there a difference between the norm of the difference between Q and P, and P and Q? Explain the differences (or lack of)
There are no differences. While there is a difference between q-p and p-q, the difference is removed when we square the values in the dotproduct. 

### Explain to what does the norm correspond in the plots you had before (p-q)?
The norm is a  measure of the length of the vector. It's a scalar, roughly equivalent to saying "This vector is x meters long". The analogy doesn't hold as much in 3+D space, but it works well enough here to understand it. 

### Expectation about features use : 
We obtained 66% accuracy with 2 features, 94% with 3 and 92% with 4. 

The performance of the algorithm augment drastically from 2 to 3 features, but then goes down slightly with a fourth feature. The basic expectation would be that giving the algo more information would directly lead to a better performance, but we observe here that it isn't the case. 

What's happening here is that, past 3 features, the model starts to overfit a lot and the added information from adding a new feature become more noise than anything. The model does not have the capacity to handle this much informations and its performance go down.

### Decision surfaces by value of K. 
In those diagram, we observe how the model would predict various query using two features. The surfaces tell us which class the model would give to the query, allowing us to understand its limit, and seeing how clearly it is sensible to outliers, data imbalance, and its general state of over and underfitting. 
When k is low (1, 3 to some extent), the edges of the surfaces are very jagged : the model is very sensitive to outliers. 
When k is medium (3-5), the edges get smoother. The noise is slightly reduced and the model is likely to be at its peak there. 
When is k (15), the edges are extremely smooth, to the point that the model is generalizing too much. In the case where the classes of the classification problem are far apart, this works fine (but it also likely wouldn't have needed a (KNN) model to be solved in the first place anyway), but if the classes are closer together, the model is underfitting by losing information information and become less accurate. 

## Lazy learning algorithm

KNN algorithm is a lazy-learning computation algorithm, in the sense that it doesn't use or do anything to the training data until we ask a query for a new point. In essence, a lazy-learning algo memorize the training set, and will compute something with it only when tasked with predicting something. Lazy learning is the opposite of eager learning algorithm, in which the algo will need to  train on the data before being able to predict anything. 

#### Example of lazy learning algorithm : 
- Knn
- Local regression
- Lazy naive Bayes rules

#### Advantage : 
- Eliminate any and all training time, since there are no training. 
- Adaptability : Lazy learning algorithms can adapt quickly to new or changing data. Since the learning process happens at prediction time, they can incorporate new instances without requiring complete retraining of the model.

#### Disadvantages : 
- The algo will need to parse the entire training set for each new prediction it makes, making it computionnaly hungry and potentially slow. 
- Lack of transparency : Lazy learning methods do not provide explicit models or rules that can be easily interpreted. This lack of transparency makes it challenging to understand the reasoning behind specific predictions or to extract actionable insights from the model.
- Storage requirements. Lazy learning algorithms need to store the entire training dataset or a representative subset of it. This can be memory-intensive, particularly when dealing with large datasets with high-dimensional features.
  
From [Datacamp](https://www.datacamp.com/blog/what-is-lazy-learning) (partially)