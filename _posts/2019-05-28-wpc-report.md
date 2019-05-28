---
layout: post
title: Predicting Drinking Water Well Failures in Tanzania
subtitle: Full Report
---

Hundreds of millions of people do not have access to clean drinking water, to the detriment of their health, education, and livelihoods. While it is important to install new wells, it is just as important to ensure existing wells remain functional. The objective of this project was to develop a model that can predict the functional status of wells in Tanzania, using a dataset consisting of 40 variables and nearly 75,000 waterpoints. I explored the data, identified instances of data leakage, imputed missing data, created new features, fit several machine learning models including ensembles, and cross-validated their accuracy. The random forest model had the best accuracy, correctly classifying the functional status for 78% of the waterpoints without the leaky predictors and 81% with the leaky predictors. My best model was within 2% of the best performing model in the drivendata.org competition.

[Link to GitHub Repository](https://github.com/kykar/water_pump_condition)  
[Link to drivendata.org Competition](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/)

Introduction
------------

Access to clean drinking water is a basic human right, but more than 1 in 3 people in Sub-Saharan Africa lack access (UN, 2018). This contributes to various health problems, including diarrhea that causes more than half a million deaths per year - disproportionately affecting children. Time spent collecting water or overcoming sickness negatively impacts people's ability to provide for their families and children's school attendance and performance (WHO, 2017). Additionally, women and girls are responsible for collecting water in 80% of households, an arduous and time-consuming task without a nearby improved water supply point (waterpoint) (UN, 2019).

Efforts to increase access to clean drinking water are impeded by frequent water pump failures that go unaddressed. Various sources estimate that around one-third of waterpoints in Sub-Saharan Africa are non-functional at any given time (Danert, 2010; Majuru, Suhrcke, & Hunter, 2016; Skinner, 2009). Therefore, it is important to be able to identify non-functional wells so that they can be repaired. The aim of this project was to predict the functional status of waterpoints in Tanzania, a country in East Africa.

The data for this project were sourced from Taarifa and the Tanzanian ministry of Water, and provided by drivendata.org for one of their data science competitions. The data consists of 59,400 waterpoints/observations in the training set (waterpoint status known/labeled) and 14,850 observations in the test set (waterpoint status unlabeled). The dataset consists of 40 variables (features) not including the waterpoint status (target variable). These features are described in further detail in the Exploratory Data Analysis section. The status of waterpoints in the training set is shown in the figure below. 54% of the waterpoints were functional, 38% were non-functional, and 8% were functional but in need of repair.

<img src="wpc_report_files/figure-markdown_github/status_group-1.png" style="display: block; margin: auto;" />

The main steps I performed included data exploration, feature engineering, fitting several different models (random forest, XGBoost, multinomial logistic regression, and k-nearest neighbors), cross validation, down selecting features, tuning model parameters, and creating ensemble models. More details are provided in the Methods section below, which is followed by the Exploratory Data Analysis, Results, and Conclusion sections.

Methods
-------

#### Data Exploration and Preprocessing

I began by exploring the data to get a better understanding of the various features and their potential usefulness for predicting the functional status of waterpoints in Tanzania. I made an initial pass through each of the variables looking at summary statistics, plotting distributions for the numeric features, and looking at summary tables for categorical features to get a better understanding of their meaning and potential usefulness in predicting waterpoint status; I removed redundant features and features with little to no variance. Through exploration I also identified features that required log transformations to normalize the data, features that required imputation, instances of data leakage, and other potential issues with the data.

During the initial pass I did not consider the features against the target variable (e.g. I did not plot features broken down by status group), to avoid biasing the feature selection process; feature selection should occur within the resampling (cross-validation) procedure, not using the entire training set (Hastie, Tibshirani, & Friedman, 2009; Kuhn & Johnson, 2018). There are more details on the resampling procedure and feature selection process in the following subsections. After I down-selected the features to include in the modeling process using the methods outlined above and below, I explored the variables further, including visualizing their relationship with the target variable.

There were a significant number of hierarchical features with overlapping levels, making them highly correlated. I will refer to these as grouped features. For example, `source`, `source_type`, and `source_class` are all directly related to the water source, but they range in the amount of detail in the levels - they have 10, 7, and 3 unique levels, respectively. There were 7 feature groups with 17 total features, not including the location feature group which had 9 features.

I did not want to include all the features within a group in the model and feature selection process due to issues arising from their correlation, such as multicollinearity in the case of logistic regression. I made an inital selection of one variable from each group based on the number of unique levels (not too many or too few) and the number of waterpoints in each level; for example, if a certain feature had levels with less than 100 datapoints and that level was combined with another level in a different feature within the group, the less detailed feature was chosen.

While some of the features were obviously correlated due to their hierarchical nature, other features needed to be tested for correlation. Since the features included a mix of numeric and categorical data, I used a custom function to calculate correlations. This function used Pearson's correlation for numeric pairs, chi-squared test for categorical pairs, and the linear regression correlation coefficient for numeric-categorical pairs. Features with correlation coefficients greater than |0.5| were not included in the model building process at the same time. Fortunately, there were very few feature pairs that exceeded this threshold.

#### Model Evaluation

Before discussing my modeling methods, I will explain the methods I used to measure model performance. The primary metric I used to evaluate model performance was the overall accuracy, mostly because this is the only metric provided by drivendata.org when submitting predictions for the unlabeled test set. As such, I used this metric during the model building process, but this was not the only metric I considered. Since the consequences for people lacking access to clean drinking water are much higher (i.e. possible death) than the consequences of sending a repair team to a functioning well, sensitivity to predicting non-functional wells is a high priority, so I also considered this metric and plotted ROC curves.

To choose between different models and variations thereof, I needed an accurate method of assessing their accuracy. If I only measured accuracy based on how well the models perform on the data used to fit the model, and selected and tuned the models accordingly, I would have ended up with an overly complex model that was overfit and would have much lower accuracy when making predictions on new data. In order to avoid this, I needed to fit the models to a portion of the data, a training set, and then measure their performance on a different portion of the data, a validation set. However, researchers have shown that judging model performance on a single test set is less than ideal (Kuhn & Johnson, 2018).

An alternative to using a single validation set is to resample the dataset, creating several different training and validation sets, then using statistics to get a realistic estimate of model performance (Kuhn & Johnson, 2018). I employed a resampling technique that is common used to estimate the accuracy of predictions -- k-fold cross-validation (Hastie et al., 2009). Usually, k is 5 or 10, with a larger value of k decreasing the bias of the estimate, and I chose to use k=10 for model assessment. There are other resampling techniques, such as bootstrap methods and leave-one-out cross-validation (k-fold where k=n), but for relatively large sample sizes 10-fold cross-validation should have reasonably low bias, acceptable variance, and faster computation time compared to these other techniques (Kuhn & Johnson, 2018).

In 10-fold cross-validation the dataset is divided into 10 equal parts, and the model is fit 10 times. In each iteration the model is fit to 9 parts of the data and the accuracy or error of the fit is measured on the 10<sup>th</sup> part (validation set). There are 10 iterations, so that each of the 10 parts is used as the validation set. For a more detailed explanation of k-fold cross validation see (Hastie et al., 2009) pages 241-249 or (Kuhn & Johnson, 2018) pages 69-71.

10-fold cross validation should give a reasonable approximation of model performance on new unseen data. As a final test of the real-world generalizability of my final models, I used them to predict the waterpoint status on the unlabeled test set (i.e. waterpoint status unknown). I submitted these predictions to drivendata.org which returned a final accuracy score. I did not use the test set for model selection or model tuning, only for the final performance evaluation.

#### Model Selection

In order to select the best model for predicting waterpoint status, I followed the model selection guideline provided by Kuhn & Johnson (2018): 1. Start with several models that are the most flexible (but least interpretable) -- models that have a high likelihood of being the most accurate across a variety of domains. These models establish the "performance ceiling." I chose to use a random forest classifier and gradient boosted decision tree classifier (xgboost) to establish the performance ceiling. 2. Apply simpler more interpretable models for comparison. I chose to use multinomial logistic regression and k-nearest neighbors models for this step. 3. Use the simplest model that closely approximates the performance of more complex models.

When comparing model performance I selected the most parsimonious (i.e. simplest) model that was within one standard error of the model with the best accuracy, which is a common rule of thumb (Hastie et al., 2009). I provide a brief overview of the candidate algorithms below.

**Random Forest**
A random forest classifier is the average of a "forest" of decision tree classifiers. Decision tree classifiers are essentially made up of nested `if-then` statements that aim to partition the data into smaller groups that contain a larger proportion of one class after each `if-then` statement. These classifiers are simple, interpretable, and can handle categorical and missing data, but they are very noisy and unstable (Kuhn & Johnson, 2018). Bagging (bootstrap aggregation) reduces the variance of decision trees by averaging many of them into an ensemble or "forest" (Hastie et al., 2009).

The random forest further improves on the bagged trees by introducing randomness: a limited number of predictors/features are randomly sampled at each split/node in the tree. By introducing randomness, the trees are de-correlated and variance of the estimate is further reduced. Random forest is popular because it performs well, requires very little tuning, is robust to noisy features, provides estimates of variable importance, can handle missing data, and it rarely overfits the data. For more information on random forests, see (Hastie et al., 2009) pages 587-602 or (Kuhn & Johnson, 2018) pages 198-203.

**XGBoost**
Boosting is a relatively new technique whose theory and algorithms were developed in the 1980's and 1990's, respectively (Kuhn & Johnson, 2018). Boosting algorithms take many weak classifiers whose error rate is only slightly better than random guessing and combines them into a strong classifier (EoSL p.337). Decision trees are a good base learner for boosting because they can be made weak learners by restricting their depth and they are easily ensembled together.

Gradient boosted trees have some basic similarities with random forests, but they have significant differences: in random forests the trees are independent, deep, and equally weighted in the ensemble, whereas with gradient boosting the trees are dependent on past trees, shallow, and unequally weighted in the ensemble. Gradient boosted trees and random forests offer similar predictive performance, but random forests are easier to tune and can be computed faster since parallel processing is straightforward due to the independent trees (Kuhn & Johnson, 2018). For more information on boosting, see (Kuhn & Johnson, 2018) pages 203-208 & 389-392).

XGBoost is a newer variation on the gradient boosting technique and was originally released in 2014. XGBoost quickly gained popularity due to its winning performance in Kaggle machine learning competitions; of the 29 Kaggle challenge winning solutions in 2015, 17 solutions used XGBoost while the next most popular method was deep neural nets which were used in 11 solutions. This method is applicable to a wide range of problems, has a faster computation time than deep learning methods with similar performance in many instances. For more information on XGBoost, see (Chen & Guestrin, 2016).

**Multinomial Logistic Regression**
Logistic regression is a simple and interpretable classification technique, but it is unlikely to perform as well as the previous two models. It is a variation on linear regression where the logit function, *l**o**g*(*p*/(1 − *p*)), is used to transform the continuous outcome variable to a probability distribution between 0 and 1 for a binary outcome variable. Multinomial logistic regression extends logistic regression to outcome variables with three or more classes. For more information on logistic regression, see (Hastie et al., 2009) pages 119-128 or (Kuhn & Johnson, 2018) pages 282-287.

**k-Nearest Neighbors**
The k-nearest neighbors (KNN) predicts the class of a sample based on the majority vote of the classes of the k-nearest datapoints. The nearest datapoints are determined by the Euclidian distance or another distance metric. Thus, features must be numeric or categorical variables that are able to be converted to numeric, and they must be centered and scaled if the units of any of the features are different. KNN is simple and has good performance on certain datasets (e.g. data with irregular decision boundary), but it is not interpretable for high-dimensional data, prone to overfitting with smaller k, can be unstable, and it does not take categorical features. For more information on KNN, see (Hastie et al., 2009) pages 463-468 or (Kuhn & Johnson, 2018) pages 350-353.

**Ensemble Model**
I created two ensemble models, which combine the predictions of the models above. The first was a simple majority vote ensemble model. In the event of a tie the prediction of the top performing model was selected, so only a unified dissenting vote could overrule the top model. The second ensemble was a stacked ensemble where the predictions from the four models above were used as input variables for fitting another random forest model.

**Final Models**
These are the final models that I used to predict the target variable on the test set for submission to drivendata.org: the two best individual models above, the ensemble model, as well as the best model with the leaky predictors included (for comparison purposes).

#### Feature Selection and Parameter Tuning

I reduced the number of potential features using the methods outlined in the Data Exploration section; I removed features that were redundant, had little to no variance, or that appeared to be data leakage and made an initial selection of a single variable among those belonging to a group (though I later substituted the grouped variables for one another).

I worked to select only the most predictive features in order to have a simpler more parsimonious model, even though Random forest and XGBoost essentially have feature selection built in, i.e. unimportant predictors are (usually) automatically not included in the tree building process. To select the most important features I used recursive feature elimination, using the `rfe()` function in the `caret` package, which employs the algorithm in the figure below, copied from (Kuhn, 2019).

![](../images/rfe_algo.png)

For each of the features that belonged to a group and remained after the recursive feature selection process, I substituted the other variables within the feature group one-by-one and reevaluated the model performance. Of the grouped features, I selected the simplest feature that didn't negatively impact model accuracy.

It is important to tune the model parameters in order to maximize performance on the validation set and not overfit to the train set. In order to tune model parameters I used the `train()` function from the `caret` package. This function repeatedly fits the model using a specified or auto-generated grid of tuning parameters. For the random forest algorithm (using "method=rf") the only tuning parameter was `mtry`, which is the number of variables to randomly sample at each split. The only tuning parameter for the KNN algorithm was `k`, the number of neighbors. The only tuning parameter for the multinomial logistic regression was `decay`, the weight decay for the penalty parameter. XGBoost has seven tuning parameters, `nrounds` (\# boosting iterations), `max_depth` (max tree depth), `eta` (shrinkage), `gamma` (minimum loss reduction), `colsample_bytree` (subsample ratio of columns), `min_child_weight` (minimum sum of instance weight), `subsample` (subsample percentage) (Kuhn, 2019).

`ntrees` is not considered a model performance parameter because performance will quickly plateau with a sufficient number of trees (e.g. &gt;200), but more trees will not improve the accuracy, nor will they cause overfitting until an excessive amount of trees are used (e.g. &gt;2000). However, I tuned the number of trees in order to use fewer trees to get to the performance plateau with faster computation time.

Exploratory Data Analysis
-------------------------

The dataset consisted of 40 features (predictor variables), including the waterpoint name and ID. The target (outcome variable) of interest was the waterpoint status (`status_group`). The majority of the data was recorded between 2011 and 2013, but some of the data was recorded as early as 2002. I've organized the features into four categories: location features, funder and installer, waterpoint features, and payments and management. I explore the features in these categories in more detail below.

There were a number of features that were either redundant or their meaning was unclear, so they were not included in the modeling process. This includes: `wpt_name` (redundant with `id`), `num_private` (unknown meaning, low variance), `public_meeting` (unknown meaning, low variance), `recorded_by` (zero variance), and `scheme_name` (unknown meaning, likely redundant).

#### Location Features

There were eight features describing the waterpoint location (`latitude`, `longitude`, `subvillage`, `ward`, `lga`, `region`, `region_code`, and `district_code`) and three features related to location (`population`, `altitude`, `basin`). Subvillage, ward, lga, region, and district are all geographic subdivisions, from smallest to largest. Although region is a subdivision of district in reality, there are more unique regions codes (27) than district codes (20). Additionally, the number of unique region names (21) is smaller than the number of regions codes. Because of these discrepancies, I decided to use the latitude and longitude location features in all the models except the logistic regression. I included `lga` as the location variable for the logistic regression, since it is very unlikely that waterpoint status would be linearly dependent on GPS coordinates. About 3% of the longitude data is missing (zeros), but none of the latitude data is missing. The latitude longitude coordinates of the waterpoints colored by functionality are shown in the figure below.

<img src="{{site_url}}/img/blog/map-1.png" style="display: block; margin: auto;" />

Nearly half of the population data was missing (recorded as zeros or ones), so I tried two different imputation methods. The first simply imputed the median population of the region (`region_code`) where the waterpoint is located. However, because of the large amount of missing population data, about one-fourth of the regions had a median population of zero. Thus, even after imputation, about one-third of the population data was still zero.

The second method used the k-nearest neighbors (KNN) algorithm, with latitude and longitude as the only inputs and the log<sub>10</sub> transform of population as the target. Using 10-fold cross-validation, the R<sup>2</sup> was 0.58 and the RMSE was 0.31. It is not straightforward to interpret the RMSE of a log transformed variable; it is not simply 10<sup>0.31</sup> = 2.0. For some context, the mean population is 15 and if you were to add and subtract the RMSE from this value you would get 31 and 8 people, respectively. While the imputation error is likely better than that of imputing the overall median or medain by region, there is still significant error. In the future I will include publicly available population density data in the imputation to further improve accuracy.

Since missing numerical data is recorded as zeros in this dataset, we cannot differentiate between missing `gps_height` data and waterpoints that are actually at sea level. In the future, I will verify the accuracy of the gps heights by checking it against elevation data for a given latitude and longitude. The `basin` feature is comprised of the geographic water basins, which only provides very low-resolution location information.

#### Funder and Installer

The dataset includes information on who funded the well (`funder`) and who installed it (`installer`). There were 1898 unique funders and 2146 unique installers recorded. However, they were not truely unique as the same funder/installer was sometimes recorded under different names. For example, "Government", "Gover", "GOVER", "Govt", "Central government." I first converted all levels to lowercase strings, then identified the government related levels with the regex `"(gov)|(dist)|(coun)"`. I then examined the strings with the detected pattern and developed a regex to exclude misidentified strings, `"(village)|(comm)|(china)|(belgian)|(finland)|(irish)|(ital)|(japan)|(iran)|(egypt)|(methodist)"`. I similarly identified community/village funders and installers with the regex `"(comm)|(vill)"`, while excluding `"(committe)|(bank)"`. I retained the top 10 and 20 funders and installers, and changed the rest to "other." There is significant overlap between the top funders and installers. The figure below shows the top 10 installers broken down by waterpoint condition.

<img src="{{site_url}}/img/blog/installer-1.png" style="display: block; margin: auto;" />

#### Waterpoint Features

There are a number of waterpoint features in this dataset, including the construction year, permit, water source, extraction type, waterpoint type, and water quality. A little over one-third of the construction year data was missing. In the furure I will impute missing construction year data. The figure below shows the construction year data broken down by functionality.

<img src="wpc_report_files/figure-markdown_github/construction_year-1.png" style="display: block; margin: auto;" />

I created a new variable, `years_op` (i.e. years operational), that was the time elapsed between the construction year and the year this data was recorded, since it was recorded over an 11 year timeframe. This was intended to help narrow down the number of years that the waterpoint was operational, but since we do not have data on the actual year of failure this is a source of uncertainty. This new variable made it possible to predict which wells have failed since the data was recorded, by adding the number of years elapsed since the data was recorded to `years_op` and re-running the model. In addition to giving updated predictions on waterpoint failures, tweaking this variable and re-running the model will validate the stability and usefullness of the model: I expect that the number of non-functional waterpoints will increase and waterpoints previously predicted to be non-functional should not change to functional when `years_op` is increased.

Another waterpoint feature was `permit`. While not explicitly defined, this feature most likely represents whether a permit was recieved for the installation of a waterpoint. The table below shows permit broken down by the functional status of the well. Functional status does not vary much by permit, so this feature is likely to have little predictive power, but it was still included in the recursive feature elimination variable selection process.

<table class="table table-condensed" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
functional
</th>
<th style="text-align:right;">
needs repair
</th>
<th style="text-align:right;">
non-functional
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
False
</td>
<td style="text-align:right;">
0.52
</td>
<td style="text-align:right;">
0.08
</td>
<td style="text-align:right;">
0.41
</td>
</tr>
<tr>
<td style="text-align:left;">
True
</td>
<td style="text-align:right;">
0.55
</td>
<td style="text-align:right;">
0.07
</td>
<td style="text-align:right;">
0.38
</td>
</tr>
<tr>
<td style="text-align:left;">
unknown
</td>
<td style="text-align:right;">
0.55
</td>
<td style="text-align:right;">
0.10
</td>
<td style="text-align:right;">
0.35
</td>
</tr>
</tbody>
</table>
The next waterpoint feature group was the water source, which included `source`, `source_type`, and `source_class`, with 10, 7, and 3 unique levels respectively. My inital selection was `source_type`, but the more detailed `source` variable is broken down by functional status in the figure below.

<img src="{{site_url}}/img/blog/source-1.png" style="display: block; margin: auto;" />

Another waterpoint feature group was the extraction type, meaning the method of extracting water from the water source. This group included `extraction_type`, `extraction_type_class`, and `extraction_type_group`, with 18, 13, and 7 unique levels, respectively. My inital selection was `extraction_type_class`, which is broken down by functional status in the figure below.

<img src="wpc_report_files/figure-markdown_github/extraction_type-1.png" style="display: block; margin: auto;" />

The next waterpoint feature group was the waterpoint type, meaning the method of dispensing the water. This group included `waterpoint_type` and `waterpoint_type_group`. The only difference was that the latter combined "community standpipe" with "community standpipe multiple." The `waterpoint_type` was the feature that I initally selected in this group, and it is shown broken down by status group in the figure below.

<img src="{{site_url}}/img/blog/waterpoint_type-1.png" style="display: block; margin: auto;" />

Additionally, two feature groups were flagged as data leakage, meaning they provide information that would not realistically be available at the time of prediction. The first "leaky" feature group is `quantity` and `quantity_group`. These features qualitatively reveal how much water flows from the waterpoint. In order to obtain quantity data one would also observe the functional status of the well. For example, waterpoints with quantity of "dry" are almost entirely non-functional. Presumably, when making new predictions in the future, quantity data would not be available for waterpoints with an unknown functional status. As such, I have developed my models without the quantity features. For comparison purposes, I added `quantity` to my final model to see how much this leaky predictor improves model accuracy.

The second leaky feature group is water quality, including `water_quality` and `quality_group`. One of the levels of this feature is "unknown" and waterpoints in this category are predominantly non-functional. Water quality could not be assessed without a functioning well (except when the source is a river, lake, etc.), thus unknown water quality leaks information about the functionality of the well. It was also not included except for comparison purposes at the end. The majority of waterpoints had "good" water quality (50,818), 5,195 have "salty" water, 1,876 had unknown water quality, and the remaining 1,511 had either milky, colored, or fluoride water quality.

#### Payments and Management

There are two related payment features in the dataset: payment amount in Tanzanian Shillings (`amount_tsh`) and the type or frequency of payment (`payment_type`), which could be by the bucket, monthly, or yearly for example. At first glance it appeared that 70% of the `amount_tsh` data was missing (recorded as zeros) but when accounting for plausible zeros in the `payment_type` categories of "never pay," "unknown," and "on failure," less than 10% of the data appeared to be erroniously recorded as zero. A smoothed histogram of non-zero payment amounts in Tanzanian Shillings (Tsh) broken down by payment type is shown below. Note that one US dollar was equivalent to about 1,500 Tsh around the time most of these data were recorded.

<img src="{{site_url}}/img/blog/amount_tsh2-1.png" style="display: block; margin: auto;" />

The median payment per bucket was 20 Tsh (~$0.01), the median monthly payment was 300 Tsh (~$0.20), and the median annual payment was 2000 Tsh (~$1.33). Thus, the relationship between payment amount and payment type is as one might expect. The log<sub>10</sub> transform of non-zero values of `amount_tsh` were saved as a new feature (`amount_log`) because its distribution was closer to normal (zeros were retained).

Note that the limited information provided on this dataset shows `amount_tsh` to be the "total static head," which is essentially a measure of water pressure. If this was the true meaning, this variable should be highly correlated with the water quantity as well as the waterpoint status, i.e. if there is zero water pressure there would be zero water flow, so `quantity` should be dry and `status` should be "non-functional." However, the relationship between `amount_tsh` and these two variables was not as expected. The correlation coefficents between log<sub>10</sub>(`amount_tsh`) and `quantity` and `status_group` are 0.18 and 0.21, respectively. The correlation coefficents between log<sub>10</sub>(`amount_tsh`) and `payment_type` is 0.78. Additionally, total static head is typically measured in feet or meters, and many of the values for `amount_tsh` are astronomical compared to even high-horsepower waterpumps, nevermind handpumps. As such, I am confident that `amount_tsh` is the payment amount in Tanzanian Shillings, not the total static head.

There were three management related features which were assumed to represents who manages the well, including the payment process -- though the exact meaning of these variables is unclear. These features are `scheme_management`, `management`, and `management_group`, with 10, 9, and 5 unique levels respectively. I selected `management` as the inital feature to use for model building.

#### Correlations

The figure below shows the correlation matrix for the 13 variables initally selected for modeling, none of which exceed an absolute correlation coefficient of 0.5. The full correlation matrix is not shown because it would be too large, and correlation of the grouped variables is expected. The following variables were not included in the inital model due to correlation coeffieients in excess of 0.5: `amount_log` (*r* = 0.78 with `payment_type`) and `scheme_management` (*r* = 0.73 with `management`). Additionally, the transformed variable `years_op` proved useful because the variable it was created from, `construction_year`, was correlated with `pop_log` (*r* = 0.90) and `gps_height` (*r* = 0.66). The grouped variables were highly correlation, as expected.

<img src="{{site_url}}/img/blog/corr-1.png" style="display: block; margin: auto;" />

Results
-------

#### Random Forest

**Recursive Feature Elimination**
The results of the recursive feature elimination are shown in the figure below, with the error bars representing the 95% confidence interval in this and all other plots. There is a significant increase in accuracy as the number of variables included in the random forest model increases from 4 to 10. Though the 12 variable model has the highest accuracy, its accuracy is not significantly better than that of the 10 variable model. As such, I chose to use the 10 variable random forest model going forward.

<img src="{{site_url}}/img/blog/rf_rfe-1.png" style="display: block; margin: auto;" />

The variable importance for the 12 variable model is shown in the figure below. `permit` and `source_type` were removed for the 10 variable random forest model.

<img src="{{site_url}}/img/blog/rf_imp-1.png" style="display: block; margin: auto;" />

**Parameter Tuning**
The RFE results above were obtained with the defualt of $mtry = sqrt(\# of variables) = 3`, which is the number of variables randomly sampled at each node, and with`ntrees=500`. I used a search grid of`mtry=c(2,3,4,5)`. For 2 and 5 accuracy was slightly reduced, but was essentially the same for 3 and 4, and not significantly different for any value of`mtry`, so I kept it at 3. I used values of`ntrees=c(51,101,201,301,501)`. Similarly, There was not a significant difference for any value of`ntrees`, but there was a slight upward trend. I chose`ntrees=201\` even though it had lower accuracy compared to 101 or 301, as that dip is likely due to variance, and 201 should provide an adequate number of trees with a faster compute time than larger forests. Overall, you can see that these parameters have very little effect on the accuracy.

<img src="{{site_url}}/img/blog/rf_mtry_trees-1.png" style="display: block; margin: auto;" />

**Variable Substitutions**
After establishing the 10-variable baseline, I substituted grouped variables for one another to see if further improvements in accuracy were possible. The figure below compares the accuracy of these variable substitutions.

<img src="{{site_url}}/img/blog/rf_tests-1.png" style="display: block; margin: auto;" />

For the "ext" substitution, I replaced `extraction_type_group` with `extraction_type_class`, which resulted in a nearly significant decrease in accuracy. For "fund" I added `funder20`, which slightly improved accuracy. For the "inst" substitution, I replaced `install20` with `install10`, which did not affect accuracy. For the "man" substitution, I replaced `management` with `management_group` or `scheme_management`, neither of which affected accuracy. For the "pay" substitution, I replaced `payment_type` with `amount_log`, which significantly reduced accuracy. For the "pop" substitution, I replaced `pop_log3` with `pop_log`, which improved accuracy by an almost significant amount. For the "pop2" substitution, I replaced `pop_log3` with `pop_log2`, which also improved accuracy. For the "year" substitution, I replaced `years_op` for `construction_year`, which had a negligible increase in accuracy.

Though none of the substitutions resulted in a statistically significant improvement in accuracy, adding `funder20` and substituting `pop_log` for `pop_log3` showed promise. I made both of these changes for the "final" model, which showed a significant improvement over the "baseline"model, which is shown in the data leakage figure below.

**Data Leakage**
The accuracy of the baseline, final (with 200 and 500 trees), and leaky predictor models are compared in the figure below. The final model is significantly better than the baseline model, but there is not a difference between the final model with 200 trees or 500 trees. Including the suspected leaky predictors, `quantity` and `water_quality`, significantly improves the model performance as expected. According to the mean decrease in the Gini coefficient, quantity becomes the most important variable in the leaky model while quality is the least important. This confirms that quantity is likely to be a leaky predictor, and quality may not be a leaky predictor but it also isn't a very useful predictor.

<img src="{{site_url}}/img/blog/rf_leak-1.png" style="display: block; margin: auto;" />

#### All Model Comparison

I focused on the random forest model in terms of the analysis and presenting the results above because it had higher accuracy than the other models; I only provide a brief overview of the other model results in comparison to the random forest results, as well as the final submission results in this section. In the figure below I show the accuracy for the final random forest (rf), random forest with leaky predictors (leak), XGBoost (xgb), multinomial logistic regression (multinom), k-nearest neighbors (knn), majority vote ensemble (vote), and stacked ranfom forest ensemble (stack) models. The 10-fold cross validation accuracy estimates with 95% confidence intervals are represented by blue dots, while the test set accuracy of the final submissions are represented by red dots.

<img src="{{site_url}}/img/blog/all_models-1.png" style="display: block; margin: auto;" />

Of the individual models without leaky predictors, the random forest had the highest CV-estimated accuracy. In fact, the individual random forest model had better accuracy on the test set than either of the ensemble models, though the stacked random forest ensemble was within one ten-thousandth of the individual random forest, at 0.7800 and 0.7801 respectively. The random forest model with the leaky predictors had the best performance overall, at 0.8125 accuracy on the test set. This is within 2% of the best performing submission to the drivendata.org competition, which has an accuracy of 0.8286. The test set accuracy is within the confidence interval for the random forest and XGBoost models, and just above it for the model with leaky predictors.

The XGBoost model was the next best individual model with a test set accuracy of 0.7601. This accuracy was acheived with tuning parameters of `nrounds=250, max_depth=5, eta=0.4, gamma=0, colsample_bytree=0.8, min_child_weight=1, subsample=0.75`, which had a CV-estimated accuracy of 0.758. The tuning grid had 500 different variations for these values (except `gamma` and `min_child_weight` which were held constant). The worst tuning parameters had a CV-estimated accuracy of 0.672, demonstarating that tuning parameters have a large impact on the performance of the XGBoost model and performance may be improved further through additional tuning. In comparison, the random forest model is much more user frinedly because it only has one tuning parameter, `mtry`, which had little effect on the estimated accuracy in this case.

Because of the consequences resulting from a lack of potable water, it is more important to identify wells that require repair than to identify (or misidentify) functioning wells. Thus, the sensitivity (a.k.a. recall or true positive rate) for predicting non-functional wells is an important model characteristic. The random forest model has a sensitivity of 0.72 for non-functional and 0.87 for functional, thus it is better at predicting functional wells, which is not ideal. The ROC curve for non-functional is shown in the figure below, with the red dot representing the current model sensitivity VS specificity.

<img src="{{site_url}}/img/blog/roc-1.png" style="display: block; margin: auto;" />

Similarly, the XGBoost model has a sensitivity of 0.69 for non-functional and 0.87 for functional, so it is slightly worse at predicting non-functional wells. In the future I could change the prediction cutoff values to a value that increases sensitivity while maintaining an acceptable number of false positives. It is also worth noting that neither the random forest nor the XGBoost models predicted "functional, needs repair" waterpoints with a sensitivity above 0.30, due to the significant class imbalance. In the future I can take steps to address the class imbalance or simply combine this class with one of the others.

#### Updated Predictions

After adding the number of years between `date_recorded` and 2019 to `years_op`, it increased from an average of 14 years to 21 years. Predicting waterpoint status using the updated values of `years_op` in the final random forest model resulted in 4667 prediction class changes. 1986 waterpoints changed from functional to non-functional, 473 changed from needs repair to non-functional, and 51 changed from functional to needs repair. These changes reflect the expected behavior--older wells are more likely to fail--but not all the changes follow this behavior. 1374 waterpoints changed from non-functional to functional, 757 changed from needs repair to functional, and 26 changed from non-functional to needs repair.

Unfortunately, I do not have a way of verifying the accuracy of the updated predictions; while it is logical that older wells are more likely to fail according to some monotonically decreasing function, the model is likely to capture greater complexities and nuances. For example, a certain installer may have changed their methods or materials over the years, causing reliability and longevity to fluctuate over time. A linear/logistic model might better represent our expectation that waterpoints are more likely to be non-functional with increasing age, but it would not capture such nuances. Additionally, if this example is true, it would be better to use the `construction_year` feature instead of `years_op` in future models.

Conclusion
----------

The random forest was the single best model, with 0.7801 accuracy on the test set, and it even outperformed the ensemble models. When I included the leaky predictors in the random forest model accuracy rose to 0.8125, which was within 2% of the top performing submission to the drivendata.org competition. Interestingly, imputing population using group means or KNN did not improve model accuracy significantly. In the case of grouped variables, substituting different variables within a group had very little effect on accuracy.

I attempted to identify additional wells that are likely to have failed since the data was originally collected, but waterpoint status predictions changed from non-functional or needs repair to functional almost as much as they changed from functional to non-functional or needs repair; this unexpected behavior calls into question the reliability of these new predictions, and the generalizability of the model over time. Thus, additional work is needed to validate the updated predictions and develop a model that better accounts for changes over time.

In the future, I would like to address the "needs repair" class imbalance. I would also like to use the dataset to optimize resource allocation for repairing waterpoints. For example, create plans that account for limited time and resources by targeting areas with large numbers of non-functional wells and incorporating the predicted probabilities (not just the predicted class) in order to target wells that are most like to be non-functional. Additionally, I will include publicly available population density data to improve the accuracy of the population imputation in the future.

It is worth noting that identifying non-functioning wells is only a small part what is needed to address waterpoint reliability in Sub-Saharan Africa. Funders and installers should be held to high standards, not only in terms of the quality of their installations but also their long-term sustainability. It is important for such projects to get community buy in long before breaking ground. These projects should have maintenance and repair plans in place, use pumps with parts that are locally available, and build local capacity for service and repair.

References
----------

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining - KDD '16, 785-794. <https://doi.org/10.1145/2939672.2939785>

Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition (2nd ed.). Retrieved from <https://www.springer.com/us/book/978038784857>

Kuhn, M. (2019). The caret Package. Retrieved from <https://topepo.github.io/caret/recursive-feature-elimination.html>

Kuhn, M., & Johnson, K. (2018). Applied Predictive Modeling (1st ed. 2013, Corr. 2nd printing 2016 edition). New York: Springer.
