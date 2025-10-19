# 🎬 Movie Recommendation System

Welcome to the **Movie Recommendation System** project! This project aims to build an efficient movie recommendation engine using a collaborative filtering approach and matrix factorization via **Singular Value Decomposition (SVD)**. The goal is to provide personalized movie recommendations based on users' past ratings and preferences by leveraging machine learning techniques and data analysis.

## 📋 Project Overview

In the age of digital entertainment, providing personalized recommendations is key to improving user experience on streaming platforms. This project employs a collaborative filtering method, focusing on matrix factorization using SVD to uncover hidden patterns in user-movie interactions. We address the problem of missing ratings using various imputation techniques and optimize the model's performance through careful evaluation and tuning.

## 🛠️ Technologies Used

- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, tqdm
- **Machine Learning Models**:
  - Collaborative Filtering using **SVD**
  - Imputation Techniques: Zero Imputation, Mean Imputation, KNN Imputation
- **Data Handling**: Feature scaling, outlier detection, missing value imputation
- **Visualization**: Data exploration and result analysis using Matplotlib and Seaborn

## 📊 Dataset

We use the **MovieLens 100k** dataset, which contains 100,000 ratings from 943 users on 1,682 movies. The dataset includes the following files:

- **u.data**: User ratings (UserID, ItemID, Rating, Timestamp)
- **u.item**: Movie metadata (ItemID, Movie Title, Genre)
- **u.user**: User information (UserID, Age, Gender, Occupation)

The dataset is **sparse**, with many missing ratings that need to be imputed for accurate predictions.

## 🧪 Methodology

### Data Exploration:
- Analyzed the structure of the dataset and checked for missing values.
- Created a user-item matrix with users as rows and movies as columns.
- Performed outlier detection and feature scaling for consistent analysis.

### Imputation of Missing Values:
To handle the missing ratings in the user-item matrix, we explored various imputation techniques:
1. **Zero Imputation**: Filled missing values with zero (baseline approach).
2. **Mean Imputation**: Replaced missing values with the mean rating of each user.
3. **KNN Imputation**: Used K-Nearest Neighbors to estimate missing values based on similar users.
4. **SuperCharged KNN**: Enhanced KNN imputation by incorporating additional user features and movie genres.

### SVD Decomposition:
- Applied **Singular Value Decomposition (SVD)** on the imputed matrices to extract latent factors.
- Truncated the SVD components to reduce dimensionality and improve generalization.
- Reconstructed the user-item matrix for prediction.

### Model Evaluation:
- Evaluated the reconstructed matrix using metrics like **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)**.
- Tested different values of `k` (number of components) to find the optimal trade-off between complexity and accuracy.
- Visualized the performance of different imputation techniques across various values of `k`.

### Recommender System:
- Developed a recommendation function to suggest the top-rated movies for a given user based on predicted ratings.
- Integrated the final matrix with user and movie data to provide personalized movie recommendations.

## 📈 Results

- The **KNN Imputation** method combined with SVD provided the best performance, with the lowest MAE and MSE.
- The optimal number of components (`k`) was found to be around **12**, indicating that the majority of the information could be captured with relatively few factors.
- The imputation method significantly impacted the model's accuracy, highlighting the importance of handling missing data effectively.

### Key Metrics:
| Imputation Method   | Optimal k | MAE   | MSE   |
|---------------------|-----------|-------|-------|
| Mean Imputation     | 9         | 0.7841| 0.6172|
| Zero Imputation     | 13        | 2.4144| 5.8271|
| KNN Imputation      | 14        | 0.7538| 0.5829|
| SuperCharged KNN    | 13        | 0.7765| 0.6012|

## 🎯 Usage

To get movie recommendations for a user, use the following function in the notebook:

```python
recommend_movies(user_id=124, user_item_matrix=user_item_train, reconstructed_matrix=final_matrix, n=5)
