# TP 1.2 : Linear regression from scratch

The linear regression is the basis for a lot of more advanced algorithms, but it also a predictive method on its own. Linear regression, as its name suggests, will try to fit the best line through a dataset, attempting to minize the error the line, necessarily flawed, will have compared to real world datas and situations. 

## Steps : 
The linear regression model as opposite to the KNN model, is part of the class of the eager-learner algorithms, which mean that it has an actual learning process. Let's explore how it goes.

### Training / Learning
The linear regression starts by a period of learning from the dataset. During this period, the model will try to find the line that can best describe the data it has, and will need to define the coefficient and intercept for the line in question. Here are the steps of the learning process : 
1. We start with random coefficients and intercept. 0 is typically good for either of them.
2. Based on those value, we make a first prediction, y_pred. 
3. We calculate the error value (MSE) between y_pred and the real value y. 
4. We slightly adjust the coefs and/or the intercept in one direction, adding or substracting the adjustement value (learning).
   1. We decide if we add or substract based on wether the MSE score has increased over the past iterations
   2. During the first 2 iterations at least, when we're establishing the first error value, we will go in the same direction twice. Afterward, depending on the data, this choice may adjust. 
   3. Basically, adding or substacting is decided by asking "Is our current result better than the last result after we added/substracted the adjustment ?" If yes, we keep doing the same as before. If no, we switch direction. 
5. We go back to step 2 of the process and repeat until we've done all the iterations required. 
   
### Predicting
Predicting is the easy part with linear regression. During the learning phase, the model learned what the best coefficients and intercept was. When we get a new query, we simply plug its value with our coefficients and intercept we learned and we get a result. 

It's a simple linear combination that can be expressed as:

$$ y = a_1x_1 + a_2x_2 + \dots + a_nx_n + b $$

Where:
- $(a_1, a_2, \dots, a_n)$ are the coefficients the model learned.
- $(x_1, x_2, \dots, x_n)$ are the values of the query.
- $b$ is the intercept the model learned.
- $y$ is the value we predict


## Questions Chapter 1: 
### 1.1 : Vector notation
*Which of the following expressions use correct notation? When the expression does make sense, give its length. In the following, $\mathbf{a}$ and $\mathbf{b}$ are 10-vectors, and $\mathbf{c}$ is a 20-vector.*

To answer the questions, I'll assume we use a "non-numpy/python" way of selecting elements. Thus, the vector c_(3:12) below takes from 3 to 12 included, making a 10-vector. 
[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

1. $\mathbf{a} + \mathbf{b} - \mathbf{c}_{3:12}$
   1. This notation is correct. a and b have the same length, so they can be added element-wise without problem. c, as selected with (3:12), is also length 10, meaning it can be substracted without problem. The final result is a length 10 vector.  
2. $(\mathbf{a}, \mathbf{b}, \mathbf{c}_{3:13})$
   1. The notation (a,b,c) mean this is a stacked vector. Elements of b are added after the end of a, making it longer. With a and b being 10-vector each, and c_(3:13) selecting 11 elements, the resulting stacked vector is a 31-vector. 
3. $2\mathbf{a} + \mathbf{c}$
   1. This vector is not correct, they do not have matching shape. Multiplying the a vector by 2 simply scales it, multiplying by 2 its elements. It does not add or double the number of total elements. 
4. $(\mathbf{a}, 1) + (\mathbf{c}_1, \mathbf{b})$
   1. This expression is correct. Each of the parenthesis is a stacked vector, where a single [1] is added after the element of a for the first set of parenthesis, and the first element of c is added on top of b. This results in two 11-vectors, who can be added element-wise correctly. The final result is also a 11-vectors. 
5. $((\mathbf{a}, \mathbf{b}), \mathbf{a})$
   1. This notation is correct. The inner parenthesis produces a stacked, 20-vectors with the elements of a and b. Then, we add a second time the elements of a at the bottom of the 20-vector, resulting in a 30-vector. 
6. $[\mathbf{a} \ \mathbf{b}] + 4\mathbf{c}$
   1. Putting a and b like that creates a 10,2 matrix. In standard mathematic notation, we can not add a matrix and a vector like that, since their dimension do not match even on one side. 
7. $\begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix} + 4\mathbf{c}$
   1. This notation makes sense. By putting the two vector vertically, we create a 20,1 matrix that is equivalent to a 20-vector. It can then be additionned to 4c, which is still a 20-vector after being multiplied by 4. This result in a final 20-vector. 

### 1.2: Interpreting Sparsity

1. $\mathbf{x}$ represents the daily cash flow of some business over $n$ days.
   1. Each elements of the vector represent how much the business gained (or lost) money overall, each day. If only a few entry are not zero, it means the business isn't gaining or spending at all most days. Maybe the business works like that, with only a few big sales and spending here and there. 
2. $\mathbf{x}$ represents the annual dollar value purchases by a customer of $n$ products or services.
   1. The vector account for how much a customer has spend for each of the n products on the catalog. Since most customers don't buy a little of everything, most of the products of the catalog are at 0, since the customer never bought it over the year. 
3. $\mathbf{x}$ represents a portfolio, i.e., the dollar value holdings of $n$ stocks.
   1. Similarly, the vector represent how much the portfolio holds of each n stocks. While an investor will diversify and buy different stocks, they are unlikely to buy into every stocks. Over the catalog of n stocks, the investor will only buy into a few of them. 
4. $\mathbf{x}$ represents a bill of materials for a project, i.e., the amounts of $n$ materials needed.
   1. The vector represent a list of every possible materials that can be needed. If only a few elements are not zero, the project only require a few materials, maybe because it's a small project or it is specialized. 
5. $\mathbf{x}$ represents a monochrome image, i.e., the brightness values of $n$ pixels.
   1. If the vector has many 0 elements, it means most of the image is likely white (represented by a 0). 
6. $\mathbf{x}$ is the daily rainfall in a location over one year.
   1. We can assume the elements represent the quantity of rain in mm. If many of the elements are 0, the location is in a place where it doesn't rain often, like a desert or something similar. 

### 1.3: Total Score from Course Record

The weight vector w will have the following values : 
For w_1 to w_8 (first category), the value will be : 10 * (1/8) * 0.25
For w_9 (second cat), the value will be : (5/6) * 0.35
For w_10 (third cat), the value will be : (5/8) * 0.4

Here's the explanation : 
- First, we want every grades to be on the same 0-100 scales, as it is the value for the finale grade s. 
  - For the first cat, that means multiplying by 10. 
  - For the second, that means multiplying them by 5/6, as 120 (the max for the cat) * 5/6 = 100 (the max for s). 
  - For the third cat, similarly, we multiply it by 5/8, as 160 * 5/8 = 100. 
- Then, only for the first category, we need to account for the fact that there are multiple grade in the category, 8 in total. Each grade being equal to the other, we divide its value by 1/8. 
- Finally, we account for the weight of each category. That's the % given in the instruction. 
  
So, the final w vector will be 

`weights = [10*1/8*0.25,10*1/8*0.25,10*1/8*0.25,10*1/8*0.25,10*1/8*0.25,10*1/8*0.25,10*1/8*0.25,10*1/8*0.25,5/6*0.35,5/8*0.4]`

or

`weights = [0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.29166667, 0.25]`


## Questions Chapter 3

### Step 1.3, linear combination with fixed coef and intercept. 
Looking at the plot, do you think the chosen values for `c` and the `intercept` are good estimates for the data? Explain briefly.

In the plot, we can see that the line goes constantly (far) under the actual values, and it generally goes in the wrong direction. Its slope is negative, where as the actual values seem to have more of a positive tendancies, as in, their y value go up with the x value. The line doesn't touch or even come close to any points. 

The interecept could be decent if we squint our eyes and the slop was different, but it could certainly be modified so that it fits the data better. 

Because of this, the currect c and intercept values are not at all a good fit. 

### Step 1.6 : testing initial and new parameters
The MSE is a value that we're trying to minimize. A lower value for the MSE means that we're improving the performances of the model. Here, the initial parameter had the slope going down since it was negative, where as the datapoints seemed to show a mostly positive arrangement. Add


### Step 2.4 : Reflection and Discussion on the drawbacks of the naive approach
The naive approach we use here is severely limited compared to more advanced approach, such as the gradient descent. 