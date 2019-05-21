---
layout: post
title: Predicting Failures of Drinking Water Wells in Tanzania
subtitle: A drivendata.org Competition
---

Hundreds of millions of people do not have access to clean drinking water, to the detriment of their health, education, and livelihoods. While it is important to install new wells, it is just as important to ensure existing wells remain functional. The objective of this project was to develop a model that can predict the functional status of wells in Tanzania, using a dataset consisting of 40 variables and nearly 75,000 waterpoints. I explored the data, identified instances of data leakage, imputed missing data, created new features, fit several machine learning models including ensembles, and cross-validated their accuracy. The random forest model had the best accuracy, correctly classifying the functional status for 78% of the waterpoints without the leaky predictors and 81% with the leaky predictors. My best model was within 2% of the best performing model in the drivendata.org competition.

[Link to GitHub Repository](https://github.com/kykar/water_pump_condition) 
[Link to drivendata.org Competition](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/)

Full report coming soon

