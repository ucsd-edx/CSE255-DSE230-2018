# Whale Classification from echo-location clicks

This project primarily involves dealing with the development of a classifier system using under-water recordings of echo-location clicks. It currently looks to classify two types of Beaked Whales: The **Cuvier's** and the **Gervais**

## Data Files:
---

* **whale_data_15mb.np** : Consists of a small subset of the data after filtering out the incomplete and misclassified data. It is used as the dataset for extracting and visualizing Eigen vectors.
* **processed_data_15mb.np** : The processed data files that consist of the data required for building the model i.e. Output of the **Data_Processing_Whales.ipynb**.
* **processed_data_150mb.np** : This is the same as **processed_data_15mb.np** , except that the Eigen Vector Analysis was done on a larger subset of Data. 
* **xgboost_output_full_data.pk** : Pickle file that saves the classifcation predictions of the XGBoost model on the entire Test Dataset.

## Data Pre-processing (Data_Processing_Whales.ipynb)
---

**Input**: whale_data_15mb.np <br>
**Output**: processed_data_15mb.np

The following steps are carried out in this file:
* Calculating Mean and Covariance for the input data
* Extracting and Analyzing the Eigen Vectors
* Writing the processed data to the output file with following columns (zero-indexed):
  * Col 0-9: projections on first 10 eigen vectors
  * Col 10: rmse
  * Col 11: peak2peak
  * Col 12: label (`0 if row.species==u'Gervais' else 1`)

The code for the same can be found [here](Sections/Section4-Classification/XGBoost/Data_Processing_Whales.ipynb). The preprocessed is a numpy array with `4175` rows (for the 15mb file) 

## XGBoost Analysis
---

### Train/Validation/Test Split
The data is shuffled and divided as follow:
* Training: 70%
* Validation: 15%
* Testing: 15%

### Parameters for XG Boost
* Maximum Depth of the Tree = 3
* Step size shrinkage used in update to prevents overfitting = 0.3
* Evaluation Criterion: `Maximize Logliklihood`
* Maximum Number of Iterations = 100, 1000


**NOTE:** 
* Originally the data analysis was done on 16 GB with data for over 6 million clicks. The data files were preprocessed on PySpark (10 nodes) cluster.
* To avoid heavy computation time, smaller subsets of data have been provided along with the data generated after different stages.
* The file **xgboost_output_full_data.pk** has been provided so that visualization and analysis on the entire test dataset can be carried out.
