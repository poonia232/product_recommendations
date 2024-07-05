# product_recommendations
Product recommendations models for e-commerce apps

Complete blog here: 

Building a recommendation model
Recommendation is essentially filtering the right set of products for any given user. There are two broad types of filtering:

1. **Content-Based Filtering:** Filters products based on user demographics (age, city, etc.), order history, app interaction (products viewed, marked favorite), etc.
2. **Collaborative Filtering:** Filters products based on the similarity of users to other users and the similarity of products to other products.

### Content-Based Filtering

Recommending products based on product features and user preferences is straightforward. Businesses with a strong understanding of their assortment can use comprehensively planned rules, supported by intuitive dashboards, to boost customer experience. Examples of this approach include:

- Deals and offers, sale offers
- Popular items, new arrivals
- Recommendations based on browsing history:
    1. Recommend the same product again after a set period (e.g., 10 days)
    2. Products the user viewed but didn't purchase
    3. Top deals on search results that didn't yield

> An important challenge is the cold start problem, where recommendations must be made to new users with little data. Addressing this is crucial, as early orders are where the biggest drop-off happens in customer acquisition funnels.
> 

### Collaborative Filtering

Collaborative filtering leverages relationships and similarities between products and users to recommend products. 

### User-User Collaborative Filtering

This method is item-agnostic and relies on clustering users based on similar behaviors. Once users are grouped into clusters based on their past behavior and demographics, products bought by others in the same cluster are recommended.

**Objective:** Predict the top 5 products (ranked according to user rating for a product) for any user from the set of products they haven’t purchased yet.

**Dataset:** [Amazon Ratings](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings)

**Code:** [User - User collaborative filter](https://github.com/poonia232/product_recommendations/blob/main/User_col_filter.ipynb)

**Implementation:**
Machine learning algorithms using matrix decompositions can predict customer ratings for products not yet bought based on other customers of similar nature and their ratings. Once we know all the products that user would give high rating to, we can recommend these products to our user.

To do this, we need to build a m x n matrix, where m is the number of customers and n is the number of products. This will be a sparse matrix since most products would not be rated by most users but that is not an issue. When we use a ML model which first factors this matrix using NMF and then reconstructs it, it will predict the rating user would give to the products they haven’t rated yet.



>**Matrix factorization to predict missing values: **
The idea here is that when you decompose a matrix with some missing values, it reduces dimensions to preserve the most important trends in dataset. Then you re-compose the matrix from factors and now you have a complete matrix with values even for the places which didn’t have value earlier. 
>

Then, we can use Non-Negative Matrix Factorization (NMF) to decompose a user-item matrix into factors, then reconstruct the matrix to predict user-product affinities. While other matrix factorization techniques like SVD or LightFM could be used, NMF from Scikit was chosen for simplicity.

The below code snippet is how product recommendation can be built using NMF from Scikit-learn. For the set of 1 Lac product ratings, the model was able to reasonably predict the ratings any user would give for any particular product with a rmse (root mean squared error) of 0.05 on 1-5 rating which is not too bad. 

For actual use, you would want to test various models, compare and identify the best configuration your use case. Hyperparameter tuning like n_components also has impact on the kind of recommendations you would be getting.


**Item-Item Collaborative Filtering:** 

This method recommends products similar to ones a user has already liked or purchased. For example, recommending more chocolates to someone who already buys chocolates. Machine learning models can cluster products based on attributes and descriptions.

>**Product clustering using description: **
The product description is well the biggest detail about the predict. We can simply cluster products based on this but first the text has to be converted to machine readable TF-IDF vectors
>
**Objective:**  Predict the nearest similar product suggestions based on product description text

**Dataset:** [DMart Products](https://www.kaggle.com/datasets/chinmayshanbhag/dmart-products)

**Code:** [item - item collaborative filter](https://github.com/poonia232/product_recommendations/blob/main/Item_col_filter.ipynb)

**Implementation:**

Use K-means clustering on TF-IDF vectors of product descriptions to find related products. While word embeddings or HDBScan could be used for better contextual understanding, K-means with TF-IDF vectors was chosen for simplicity

The model worked decently well on the dataset of D-mart products. But using k-means with Tf-iDF means that models performance is highly influenced by the model hyperparameters like number of clusters for k-means and vocabulary length for Tf-IDF. For a production system you really want to test it out.

