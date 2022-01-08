# Netflix Challenge - Movie rating prediction

### Data Mining (CSE2525) 2020-2021

The goal of this team project is to develop a recommendation algorithm for movies. The data for this project comprises 910,190 ratings (on a 1 to 5 scale) that were given by 6,040 users to 3,706 movies. Next to the ratings, the data contains some information on the users (gender, age, and profession) and information on the movies (title and year of release). We develop an algorithm that, based on this data, predicts as good as possible what rating a particular user will give to a particular movie.

### Demographics and content-based approaches
We also applied a hybrid approach that builds upon CF to improve its predictability by combining information
extracted from demographic data (figure below) and content (movie) data.

![Statistics](resources\dataset_stats.png)

### Matrix factorization and SGD
SGD-based factorization has been a successful method for recommendation algorithms and it was used by the winners
of the **actual** Netix challenge. We used a hybrid approach: content-based Collaborative Filtering (item-item: between items calculated using people's ratings of the items - i.e. movies),
in combination with Stochastic Gradient Descent.

### Cross-referencing with IMDb dataset
We turned towards the extensive IMDb dataset which provided us with the opportunity to take factors such as the genre or the metascore of a given movie into
account when building content-based methods. This resulted in ~ 15% improvement in score. 

### Results
For each algorithm, we performed hyper-parameter tuning to choose the values for certain parameters that minimize the RMSE score.
The most effective method found was the hybrid **SGD-enhanced-content-CF**.

![Results](resources\results.png)


### Contributors
* Luca Pantea
* Nikolaos Efthymiou