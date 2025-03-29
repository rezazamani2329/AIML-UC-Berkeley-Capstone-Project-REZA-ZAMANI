# Big Picture of the Project and its process:  
    


### 1- Using AIML to evaluate marketing strategy

 Suppose you take part in a meeting to evaluate the probability of default payment by customers 
 
 Speaker represents the strategy and now is your turn to give suggestion   
 
  You can suggest this: Oh, let see what AI says 


### 2- Structure designer of the project and Python code writer 

 Here is the answer of AIML for your suggestion, please follow it. You will find some good ideas

 This code has user-friendly structure. It is classification issue in machine learning, as capstone project for `UC Berkeley Professional Certificate in AIML`, from `Berkeley Engineering` and `Berkeley HAAS`, and written by `Reza Zamani`

### 3- Process of the project
1- Business Understanding (Problem statement, purposes, questions, methodology, steps, phases of project)  </span>

 <span style="color: Red; "> 2- Data understanding </span> 
 
 <span style="color: Red; "> 3- Feature Engineering </span> 
 
  <span style="color: Red; "> 4- Train/Test split and handling imbalanced dataset <span style="color: blue; ">  </span> 
  
  <span style="color: Red; ">5- Machine learning models and their evaluation before hyperparameter tuning  </span>
  
  <span style="color: Red; ">6- Machine learning models after hyperparameter tuning  </span>
  
  <span style="color: Red; ">7- Comparison of Models performance, evaluation, best model, and SHAP</span>
  
  <span style="color: Red; ">8- Conclusion</span>

## **4- Stages of Project** 

#### <span style="color: blue; "> **1-First Phase:** </span> 

<span style="color: blue; "> **Data Understanding** </span> 
- General checking: missing data, duplicate, incorrect data, ...
- Target Variable Visualization
- Numerical Features Visualization
- Categorical Features Visualization
- Relationship between features, and their relationship with target
- Correlation problem

<span style="color: blue; "> **Feature Engineering** </span> 

- <span style="color: blue; "> Using algorithms, before hyperparameter tuning </span> 

- <span style="color: blue; "> Choosing the best algorithm with different criteria </span> 

####  <span style="color: green; "> **2-Second Phase:** </span> 

- <span style="color: green; "> Tuning the hyperparameters </span> 

- <span style="color: green; "> Choosing best parameters for each algorithm </span> 

- <span style="color: green; "> Evaluating the algorithms with different criteria (accuracy, time, f1, precision, recall, ROC_AUC </span> 

- <span style="color: green; "> Choosing the best algorithm with different criteria </span> 

#### <span style="color: orange; "> **3-Third Phase:** </span> 

- <span style="color: orange; "> SHAP, and deployment </span> 

- <span style="color: orange; "> Policy recommendation </span> 


# 1- Business Understanding 

###   **1-1-Main Research Question**  
-  Can Machine Learning models accurately predict whether credit card customers will default or not

###  **1-2-Other Research Questions**   
- What are the main factors affecting the probability of default by customers

- what are the relative importance of factors affecting the probability of default by customers

###  **1-3-Problem Statement** 
Customers and also managers of financial institutions- a credit card financial institutions - want to have a better understanding of the factors that impact credit card payment default and get a solution in place that can predict with high accuracy whether customers will default on payments.

To solve this problem we use a dataset that contains credit card payment information and it is our task to determine what factors impact payment default as well as build a model or an ensemble of models that can predict default.

From the standpoint of risk management, the predictive accuracy of the predicted chance of default will be more valuable than the binary outcome of classification - credible or not credible clients. We must determine which clients will fall behind on their credit card payments. Financial dangers are demonstrating a trend regarding commercial bank credit risk as the financial industry has improved dramatically. As a result, one of the most serious risks to commercial banks is the risk prediction of credit clients. The current project is being created in order to analyses and predict the above-mentioned database. This research aims to identify credit card consumers who are more likely to default in the next month.

###  **1-4-Main Goal** 
- Using AIML models, predict whether credit card customers will default or not

###  **1-5-Other Goals** 
- Determine main features have the highest impact on credit card default

- Understand the relationship between features together

- Understand the mechanism through which features affect default payment as target variable

###  **1-6- Methodology** 
- **Classification**
###  <span style="color: red; "> **1-7-Stages of Project** </span> 

###  **1-8-Methods (AIML algorithms)**

-  **Logistic Regression**

-  **K-Nearest Neighbors (KNN)**

- **Decision Trees**

- **Support Vector Machine (SVM)**

- **Random Forest**

- **AdaBoost**

- **XGBoost**

- **Multi Layer Perceptron (MLP)**

#  **2- Data Understanding** 
##  **2-1-Load Dataset**  
##  **2-2- Understanding the Features** 

***Input variables:*** 

* **ID:** Unique ID of each client
* **LIMIT_BAL:** Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
* **Gender:** 1 = male; 2 = female
* **Education:** 1 = graduate school; 2 = university; 3 = high school; 4 = others
* **Marital status:** 1 = married; 2 = single; 3 = others.
* **Age:** Age in years

**History of past payment.**

From April to September of 2005, we tracked historical monthly payment records. The payback status is measured using the following scale: -2=no spending, -1=paid in full, and 0=use of revolving credit (paid minimum only).

1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.

* **PAY_0:** Repayment status in September 2005

* **PAY_2:** Repayment status in August 2005

* **PAY_3:** Repayment status in July 2005

* **PAY_4:** Repayment status in June 2005

* **PAY_5:** Repayment status in May 2005

* **PAY_6:** Repayment status in April 2005

**Amount of bill statement (NT dollar).**
* **BILL_AMT1:** Amount of bill statement in September 2005

* **BILL_AMT2:** Amount of bill statement in August 2005

* **BILL_AMT3:** Amount of bill statement in July 2005

* **BILL_AMT4:** Amount of bill statement in June 2005

* **BILL_AMT5:** Amount of bill statement in May 2005

* **BILL_AMT6:** Amount of bill statement in April 2005

**Amount of previous payment (NT dollar).**
* **PAY_AMT1:** Amount of previous payment in September 2005

* **PAY_AMT2:** Amount of previous payment in August 2005

* **PAY_AMT3:** Amount of previous payment in July 2005

* **PAY_AMT4:** Amount of previous payment in June 2005

* **PAY_AMT5:** Amount of previous payment in May 2005

* **PAY_AMT6:** Amount of previous payment in April 2005
 

**Output variable (desired target):**  

* **default.payment.next.month:** Default payment (1=yes, 0=no)
 
###   **2-3- General check of data, info, missing and duplicate** 
-Missing data, outlier and duplicated data are checked. 
- **Handling Missing Values:**
   - `No missing values`. 
###   **2-4- checking Variables correctness** 
 **Data correcting:**
   - 9 Features have some incorrect data:  `EDCATION', 'SEX', 'MARRIAGE' and 'PAY_1' to 'PAY_6'
   - we correct mistakes
###  **2-5-Data Wrangling** 
   - rename most of features to be easily followed
   - change variables education, sex, and marriage from numerical to categorical data to follow easily and make sense 
   - change target variable in this phase from numerical to categorical, later after data visualization we will change it again to numerical
###  **2-6- Target variable Analysis** 
- we have imbalanced dataset, then our project in imbalanced dataset classification
  
- target`         `Percent
  
-No`              `0.78

-Yes`             `0.22


![TARGET_IMBALANCE](Images/TARGET_IMBALANCE.PNG)

###  **2-7- Numerical variables visualization** 
  
- **Numerical features:**
- `LIMIT_BAL`,`AGE`,
- `'PAY_SEPT`, `PAY_AUG', 'PAY_JUL', 'PAY_JUN', 'PAY_MAY', 'PAY_APR`,
- `BILL_AMT_SEPT', 'BILL_AMT_AUG', 'BILL_AMT_JUL', 'BILL_AMT_JUN', 'BILL_AMT_MAY','BILL_AMT_APR`,
- `'PAY_AMT_SEPT', 'PAY_AMT_AUG', 'PAY_AMT_JUL', 'PAY_AMT_JUN', 'PAY_AMT_MAY', 'PAY_AMT_APR'`,
- **using different methods for visualization:**
- #### **`scatterplot, histplot, violinplot, catplot, pariplot, kdeplot, boxplot, barplot`**
- We got various general point for numerical features and their effect on target variable (default payment)

###  *2-7-1- Visualization of "LIMIT_BAL" feature (balance limit)* 
 
#### <span style="color: blue; "> Points for "LIMIT_BAL" feature and its effect on target: </span> 
1. <span style="color: red; ">  As balance limit increases, the probability of default decreases   </span>  
2.   <span style="color: red; "> The average of balance limit for people who has default is around half of people do not have default.
3.    <span style="color: red; "> Most of people have balance limit smaller than $300k dollars  </span>
4.    <span style="color: red; "> Maximum number of balance limit is below 100k dollar</span>
5.    <span style="color: red; "> Very few balance limit are above 500k </span>
6.    <span style="color: red; ">  When balance limit is less than equal to 50k or smaller default ratio is high  </span>
7.    <span style="color: red; "> Default ratio is exceptionally high when balance limit is 550000 and 600000 dollars </span>
</div>

![Limit_ball_distribution](Images/Limit_ball_distribution.PNG)

![BALANCE_TARGET](Images/BALANCE_TARGET.PNG)

![Limit_ball_risk_of_default_0_1000k](Images/Limit_ball_risk_of_default_0_1000k.PNG)

![Limit_ball_risk_of_default_0_100k](Images/Limit_ball_risk_of_default_0_100k.PNG)



###  *2-7-2- Visualization of "age" feature* 
 
#### <span style="color: blue; "> Points for "age" feature and its effect on target: </span> 
1.  <span style="color: red; "> Maximum default for male is in the age 29  </span>
2. <span style="color: red; "> maximum default for female is in the age 27  </span>
3. <span style="color: red; "> Age group 31-40 has the lowest probability for default (20%)   </span>
4. <span style="color: red; "> Age group 71-80 has the highest probability for default (more than 33%)   </span>
5. <span style="color: red; "> Age group 21-30 has the second lowest probability for default   </span>
#### 6. <span style="color: red; "> important point: probability of default 31-40 < 21-30 < 41-50 < 51-60< 61-70 < 71-79 </span>

![AGE_DEFAULT](Images/AGE_DEFAULT.PNG)

AGE_DEFAULT
###  *2-7-3- Relationship between age and other factors* 
 
#### <span style="color: blue; "> Points for relationship between "age" and "balance limit" </span> 
1. <span style="color: red; "> age between 20-30 has the minimum balance   </span>  
2.   <span style="color: red; "> age between 70-79 has the maximum balance 
3. <span style="color: red; "> at first the balance increases as age goes up, but then decrease, and in retirement it increase again   </span>

![AGE_BILL](Images/AGE_BILL.PNG)

   
### *2-7-4- Visualization of history of payments* 
 <div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 
 
#### <span style="color: blue; "> Points for "history of payment" feature and its effect on target: </span> 
1. <span style="color: red; "> **Remarkable ratio** of people has**paid the minimum** only </span>  
2.   <span style="color: red; ">Someone who pays only the minimum have a higher risk of default
3. <span style="color: red; "> Customers ratio with **2 month default**  is high   </span>  
4. <span style="color: red; "> Customers ratio with **more than eight months default**  is high  </span>  
5. <span style="color: red; "> All months have approximately similar KDE density function ratio with **more than eight months default**   </span>  

 <div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 
 
#### <span style="color: blue; "> Points for distribution of "history of payment" feature </span> 
1. <span style="color: red; "> For all six months, Interquartile (25) is on time payment </span>  
2. <span style="color: red; "> For all six months, Interquartile (75) is 2 months delay in payment </span>  
3. <span style="color: red; "> For all six months, more than 5 months delay in payment is very low </span>

![payment_history_kde2](Images/payment_history_kde2.PNG)
![payment_history_kde1](Images/payment_history_kde1.PNG)

### 2-7-4-1- Probability of default with attention to history of payments 
##### We focus here only on first and last months, as they are more important than others in affecting target variable 
#### <span style="color: green; ">Points for history of payment on September (last month):

- 1-	Default probability with full payment is 16%
- 2-	Default probability for people who pay only part of debpt is 13%
- 3-	Default probability with one month delay is 33%
- 4-	Default probability with two months delay is 69%, which shows there is a sharp increase from one month delay to two months delay 
- 5-	Default probability with three months delay is 75% and in maximum of default between all conditions 
- 6-	Default probability with four months to 8 months delay is 50% or more
- 7-	Default probability with more than 8 months is 13% 
- 8-	Generally, we see that first month delay is threshold point, as after that we see remarkable increase in default in second months. 
- 9-	Generally, fully payment has the minimum probability of default
  
#### <span style="color: green; ">Points for history of payment on April (first month):
- 1-	Default probability with full payment is 19%
- 2-	Default probability with having only part of payment is 0%
- 3-	Default probability with one month delay is 51%
- 4-	Default probability with three to seven months delay are more than 60% and even in seven month delay it reaches to more than 80%
- 5-	Default probability with eight months delay is 100%
- 6-	Generally, probability of default is increasing from full payment to eight months delay. 
- 7-	Generally, we see that there is a sharp increase in the probability of default in second months delay, which shows month one and two are critical for default or not.
   
![DEFUALT_SEPTEMBER](Images/DEFUALT_SEPTEMBER.PNG)

![DEFUALT_APR](Images/DEFUALT_APR.PNG)


### *2-7-4-2- pairplot to have big picture*   

### *2-7-5- Visualization of Monthly Bill Amount* 

#### *2-7-5-1- Distribution of Monthly Bill Amount* 
#### <span style="color: red; "> *2-7-5-2- Relationship between Bill amount and target variable* </span>  

#### <span style="color: red; ">2-7-5-3- Probability of default with attention to Bill amount </span> 
- We have information of bill amount for six month (from April to September). Here we give probability of default only for two of them, September and April, which are more important with attention to SHAP and feature importance.

![Bill_amnt_default_september_0_1000k](Images/Bill_amnt_default_september_0_1000k.PNG) 


![Bill_amnt_default_september_0_100k](Images/Bill_amnt_default_september_0_100k.PNG)  


![Bill_amnt_default_april_0_1000k](Images/Bill_amnt_default_april_0_1000k.PNG)  


![Bill_amnt_default_april_0_100k](Images/Bill_amnt_default_april_0_100k.PNG)  

 
####  *2-7-5-4- pairplot of Bill Amount to have big picture and relationship between them* 
 
1. <span style="color: red; "> main part of bill amount (Inter Quantile Range (IQR) for all six months is between 35k to 70k </span>  
2.   <span style="color: red; ">as the number of previous months increases, the amount of 25%, 50% and 75% decreases 
3.   <span style="color: red; ">From pariplot, it seems some of them have correlation
![pariplot_billamount](Images/pariplot_billamount.PNG)

![BILL_AMNT_DESCRIBE](Images/BILL_AMNT_DESCRIBE.PNG) 
 
###  *2-7-6- Visualization of pay amount (Previous payment)*  
 <div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 
 
1. <span style="color: red; "> main part of bill amount (Inter Quantile Range (IQR) for all six months is between 4000 to 5000 </span>  
2.   <span style="color: red; "> 25%, 50% and 75% quartiles decreases, as the previous month increases  
3.   <span style="color: red; "> mean of previous payment for six months does not have clear trend, but mean of previous payment is among 4800 to 5700 

#### *2-7-6-1- Distribution of pervious payment (pay amount)* 
![previous_payment_KDE2](Images/previous_payment_KDE2.PNG)
![previous_payment_KDE1](Images/previous_payment_KDE1.PNG)

![PAY_AMNT_DESCRIBE](Images/PAY_AMNT_DESCRIBE.PNG)

## <span style="color: red; ">2-7-6-2- Probability of default with attention to pay amount </span>  

- We have information of bill amount for six month (from April to September). Here we give probability of default only for two of them, September and April, which are more important with attention to SHAP and feature importance.

![pay_amnt_default_september_0_1000k](Images/pay_amnt_default_september_0_1000k.PNG)

![pay_amnt_default_september_0_100k](Images/pay_amnt_default_september_0_100k.PNG)

![pay_amnt_default_april_0_400k](Images/pay_amnt_default_april_0_400k.PNG)

![pay_amnt_default_april_0_100k](Images/pay_amnt_default_april_0_100k.PNG)


  
#### *2-7-6-3- pariplot of previous payment to have big picture and relationship between them* 
1. <span style="color: red; "> main part of bill amount (Inter Quantile Range (IQR) for all six months is between 4000 to 5000 dollars</span>  
2.   <span style="color: red; "> 25%, 50% and 75% quartiles decreases, as the previous month increases  
3.   <span style="color: red; "> mean of previous payment for six months does not have clear trend, but mean of previous payment is among 4800 to 5700 dollars


###  *2-7-6-4- Relationship between bill amount and previous payment in each month*  
 <div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 
 
1. <span style="color: red; "> When Previous payment is low but Monthly Bill is high there is high possibility of default the payment. </span>  
2. <span style="color: red; "> approximately the distribution of default or payment for all six months are the same, when paying attention to the both  Previous payment and bill amount </span>  

![pay_and_bill2](Images/pay_and_bill2.PNG)
![pay_and_bill1](Images/pay_and_bill1.PNG)


###  *2-7-7- Correlation between  numerical features* 
- We find some features have correlation with each other

###  **2-7-7-1-  using **VIF** to check the correlation conditions**
- using VIF, we find that four features (`BILL_AMT_AUG`, `BILL_AMT_MAY`, `BILL_AMT_JUL`, `BILL_AMT_JUN` have high correlation, then remove them from dataset
  
### **2-8-Categorical variable Analysis**
- Categorical features are: `EDUCATION`, `SEX`, and `MARRIAGE`
- For categorical data, we check distribution of each feature (with counplot), their effect on target variable with heatmap, and their relationship with default payment.  
###  *2-8-1- Visualization of "gender" feature* 
1. <span style="color: red; "> 60% of sample is female and 40% is male   </span>  
2.   <span style="color: red; "> around 24% of male has default, and around 21% of female has default
3.    <span style="color: green; "> Male has higher rate of default than Female  </span>  

![GENDER_TARGET](Images/GENDER_TARGET.PNG)


###  *2-8-2- Visualization of "education" feature* 
1. <span style="color: red; "> 45% of customers have university degree, after that 35% have graduate school.   </span>  
2.   <span style="color: red; "> It seems people with high school has higher level of default

![EDUCATION](Images/EDUCATION.PNG)
 
### *2-8-3- Visualization of "Marriage" feature*  
1. <span style="color: red; "> 53% of customers are single and 45% are married. </span>  
2.   <span style="color: red; "> it seems Married people have higher level of default than single

![MARRIAGE](Images/MARRIAGE.PNG)


### *2-8-4- Heatmap for categorical variables* 

![Heatmap-categorical_variables](Images/Heatmap-categorical_variables.PNG)

### *2-8-5- Probability of default with attention to categorical variables*
 
1. <span style="color: red; "> people with high school degree have the highest default (25%), followed by university degree (23%), and Graduate School (19%) and others (7%)  </span>  
2.   <span style="color: red; "> single people (22%) has the lowest level default, followed by married (23%) and others (24%)
3.   <span style="color: red; "> default of male (24%) is higher than female (21%)

![DEFAULT_SEX_EDUCATION_MARRIAGE](Images/DEFAULT_SEX_EDUCATION_MARRIAGE.PNG)

### *2-8-6- Relationship between limit balance, education, gender, and marriage* 
1. <span style="color: red; "> Maximum balance limit in account belongs to people with graduate school degree   </span>  
2. <span style="color: red; "> Minimum balance limit in account belongs to people with high school degree   </span>
3. <span style="color: red; "> Balance limit for female is higher than male   </span>  
4. <span style="color: red; "> it shows that male is risk lover than female </span>
5.  <span style="color: red; "> "Married" group has the maximum limit balance </span>  
6. <span style="color: red; "> "Others" group has the minimum limit balance </span>

![EDCUATION_BILL_AMNT](Images/EDCUATION_BILL_AMNT.PNG)

![GENDER_BILL_AMNT](Images/GENDER_BILL_AMNT.PNG)

![MARRIAGE_BILL_AMNT](Images/MARRIAGE_BILL_AMNT.PNG)
  
# **3- Engineering Features** 
- 1- check the dataset info to be sure have all data or do not have extra 

- 2- Split the data into features and target

- 3- Identify `categorical and numerical columns`

- 4- `Encode the target variable` with `LabdelEncoder()`

- 5- `OneHotEncoder` fo categorical and `StandardScaler()` for numerical variables 

#  **4- Train/test split and Handling Imbalanced Datasets** </span>  
##  **4-1-Train/Test Split** 
With data prepared, we split it into a train and test set.

##  <span style="color: red; "> **4-2-SMOTE method for Imbalanced Dataset** </span>  
<div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 
 Our target variable in imbalance, then we should use the approach attention to this.

There are different approaches to split imbalance classification. Here are some of them 

1. **Oversampling**
- SMOTE (Synthetic Minority Over-sampling Technique)
- ADASYN (Adaptive Synthetic Sampling)
- Random Oversampling (Simply duplicates minority class instances)
2. **Undersampling**
- Random Undersampling
- Tomek Links (Removes borderline samples)
- NearMiss (Keeps only the hardest-to-classify examples)
3. **Hybrid Methods**
- SMOTE + Tomek Links
- SMOTE + Edited Nearest Neighbors (ENN)
4. **Class Weighting**
5. **Algorithm-Level Methods**
6. **Data Augmentation**

### We choose SMOTE

#### 1. <span style="color: red; ">  **At first we split data into train and test, then applied SMOTE, then model does not have data leakage**  </span>  
#### 2. <span style="color: red; ">  **Moreover, in this project, we find that if we apply SMOTE before splitting the data, we will have overfitting or underfitting, I Learn it from my supervisor in UC Berkeley**</span>  
#### 3. <span style="color: red; ">  **As we applying train/test at first, then applying SMOTE, we do not have data leakage and also check and find there is not any overfitting**</span>  

# <span style="color: red; ">  **5- Machine learning Models** </span> 
Strategy:

- 1-At first we define and fit each model
- 1- in each model, we see accuracy
- 2- in each model, we see classification report
- 3- check train score, test score, fitting time, precision, recall and f1
- 4- create dataframe to see all score together
- 5- with attention to all criteria, choose the best model in this step
- 6- we do not tune the models in this section, we will tune the models in next section
#### <span style="color: red; ">  **5-1-Decistion Tree** </span> 
#### <span style="color: red; ">  **5-2-XGBoost** </span> 
#### <span style="color: red; ">  **5-3-KNN (K-Nearest Neighbors)** </span> 
#### <span style="color: red; ">  **5-4-Logistic regression** </span> 
#### <span style="color: red; ">  **5-5-Random Forest** </span>
#### <span style="color: red; ">  **5-6-AdaBoost** </span> 
#### <span style="color: red; ">  **5-7-SVM (Support Vector Machine)** </span> 
#### <span style="color: red; ">  **5-8-MLP (Multi Layer Perceptron)** </span> 

Evaluation: 


![EVALUATION_BEFROE_TUNED](Images/EVALUATION_BEFROE_TUNED.PNG)



## <span style="color: red; ">  **5-9-Best Model** </span> 
### <span style="color: green; ">  **in this step, best model is chosen before tuning the hyperparameters** </span>

1- **Accuracy**: 

- `XGBoost` has maximum score, followed by `random forest` and `decision trees`  

2- **Precision**: 

- `XGBoost` has maximum score, followed by `random forest` and `decision trees`  

3- **Recall**: 

- `Logistic regression` has maximum score, followed by `MLP` and `SVM`  

4- **f1**: 

- `Random forest` has maximum score, followed by `AdaBoost` and `MLP`

5- **ROC_AUC**: 

- `Random forest` has maximum score, followed by `AdaBoost`, `XGBoost` and `SVM`
  
6- **Fit Time**: 

- `KNN`, `Decision Tree`, and `XGBoost` have the minimum Average Fit Time 

- `SVM` and `MLP` have the maximum Average Fit Time
    
## <span style="color: green; "> **Best model before hyperparameter tuning**: <span style="color: red ; ">**Random Forest**   </span> 
- **It has maximum ROC_AUC**
- **It has the second maximum accuracy**
- **It has the maximum f1**
- **It has the maximum Recall**
- **It has also relatively low fit time**


#  <span style="color: red; "> **6- Models with hyperparameter tuning (best parameters)** </span>  

####  <span style="color: red; "> **6-1 Decision Tree (best parameters)** </span>  
#### <span style="color: red; ">  **6-2-KNN (best parameters)** </span> 
#### <span style="color: red; ">  **6-3-SVM (best parameters)** </span> 
#### <span style="color: red; ">  **6-4- logistic Regression( best parameters)** </span> 
#### <span style="color: red; ">  **6-5-Random Forest ( best parameters)** </span> 
#### <span style="color: red; ">  **6-6-AdaBoost (best parameters)** </span> 
#### <span style="color: red; ">  **6-7-XGBoost (best parameters)** </span> 
#### <span style="color: red; ">  **6-8-MLP (Multi Layer Perceptron) (best parameters)** </span> 

 **Here are the results:**
- 1- Best Params for Decision Tree classification is : {'model__criterion': 'log_loss', 'model__max_depth': 10, 'model__max_features': 'sqrt'}
  
- 2- Best Params for KNN classification is: {'model__n_neighbors': 2, 'model__p': 1, 'model__weights': 'distance'}

- 3- Best Params for SVM classification is : {'model__C': 1, 'model__kernel': 'rbf'}
  
- 4-Best Params for Logistic Regression classification is: {'model__C': 1, 'model__max_iter': 100, 'model__solver': 'liblinear'}
  
- 5- Best Params for Random Forest classification is : {'model__max_depth': 10, 'model__max_features': 'sqrt', 'model__n_estimators': 50}

- 6- Best Params for AdaBoost classification is : {'model__learning_rate': 1, 'model__n_estimators': 100}

- 7 - Best Params for XGBoost classification is: {'model__learning_rate': 0.1, 'model__max_depth': 10, 'model__n_estimators': 100}

-8-  Best Params for MLP classification is: {'model__activation': 'relu', 'model__hidden_layer_sizes': (5,)}



# <span style="color: red; ">  **7- Comparison of tuned models (with best parameters)** </span> 

- At first we define pipeline to find best parameters for each model  </span> 
- Then fit the models and make prediction
- After that we have evaluation with different criteria. Here is the results  </span> 

![Classifiers_Comparison](Images/Classifiers_Comparison.PNG)

![EVALUATION_TUNED_MODELS](Images/EVALUATION_TUNED_MODELS.PNG)


*******************************************************
KNN Accuracy: 0.86

KNN Classification Report:
               precision    recall  f1-score   support

           0       0.93      0.77      0.84      4613
           1       0.81      0.94      0.87      4733

    accuracy                           0.86      9346
    macro avg      0.87      0.86      0.86      9346
    weighted avg   0.87      0.86      0.86      9346

*******************************************************
Decision Tree Accuracy: 0.74

Decision Tree Classification Report:
               precision    recall  f1-score   support

           0       0.72      0.78      0.75      4613
           1       0.77      0.70      0.73      4733

    accuracy                           0.74      9346
    macro avg       0.74      0.74     0.74      9346
    weighted avg    0.74      0.74     0.74      9346

*******************************************************
SVM Accuracy: 0.71

SVM Classification Report:
               precision    recall  f1-score   support

           0       0.67      0.82      0.74      4613
           1       0.78      0.61      0.68      4733

    accuracy                           0.71      9346
    macro avg      0.72      0.71      0.71      9346
    weighted avg   0.72      0.71      0.71      9346

*******************************************************
Logistic Regression Accuracy: 0.63

Logistic Regression Classification Report:
               precision    recall  f1-score   support

           0       0.65      0.52      0.58      4613
           1       0.61      0.73      0.67      4733

    accuracy                           0.63      9346
    macro avg      0.63      0.63      0.62      9346
    weighted avg   0.63      0.63      0.62      9346

*******************************************************
Random Forest Accuracy: 0.79

Random Forest Classification Report:
               precision    recall  f1-score   support

           0       0.76      0.86      0.81      4613
           1       0.84      0.73      0.78      4733

    accuracy                           0.79      9346
    macro avg      0.80      0.80      0.79      9346
    weighted avg   0.80      0.79      0.79      9346
*******************************************************
AdaBoost Accuracy: 0.75

AdaBoost Classification Report:
               precision    recall  f1-score   support

           0       0.71      0.83      0.77      4613
           1       0.80      0.66      0.73      4733

    accuracy                           0.75      9346
    macro avg      0.76      0.75      0.75      9346
    weighted avg   0.76      0.75      0.75      9346
*******************************************************
XGBoost Accuracy: 0.86

XGBoost Classification Report:
               precision    recall  f1-score   support

           0       0.83      0.90      0.86      4613
           1       0.89      0.82      0.85      4733

    accuracy                           0.86      9346
    macro avg      0.86      0.86      0.86      9346
    weighted avg   0.86      0.86      0.86      9346

*******************************************************
MLP Accuracy: 0.70

MLP Classification Report:
               precision    recall  f1-score   support

           0       0.67      0.78      0.72      4613
           1       0.75      0.62      0.68      4733

    accuracy                           0.70      9346
    macro avg      0.71      0.70      0.70      9346
    weighted avg   0.71      0.70      0.70      9346
*******************************************************
- We also plotted ROC curve** </span> 
![ROC_AUC](Images/ROC_AUC.PNG)

- Here is Confusion Matrix** </span> 
![confusion_matix1](Images/Confusion_matrix1.PNG)
![confusion matix2](Images/Confusion_matrix2.PNG)

### <span style="color: red; ">  **-Best Model after tuning the hyperparameters** </span> 

<div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 
    
1- **Accuracy**: 

- `KNN` and `XGBoost`   have maximum  score, followed by `Random Forest`. 

2- **Precision**: 

- `XGBoost` has maximum score, followed by `random forest` and `KNN`  

3- **Recall**: 

- `KNN` has maximum score, followed by `XGBoost`  

4- **f1**: 

- `KNN` has maximum score, followed by `XGBoost`

5- **ROC_AUC**: 

- `XGBoost` has maximum score, followed by `KNN`.

## <span style="color: green; "> **Best model** after hyperparameter tuning: <span style="color: red; ">**XGBoost**   </span> 
- **It has the maximum Precision**
-  **It has the maximum ROC_AUC**
- **It has the maximum accuracy**
- **It has the second maximum Recall**
- **It has the second maximum f1**

## <span style="color: green; "> **Second Best model** after hyperparameter tuning: <span style="color: red; ">**KNN**   </span> 
- **It has the maximum f1**
- **It has the maximum Recall**
- **It has the second maximum accuracy**
- **It has the second maximum ROC_AUC**
- **It has the third maximum Precision**


## <span style="color: red; ">  **7-7-feature importance and SHAP** </span> 
##### <span style="color: red; ">  **7-7-1-Feature Importance with the best model (XGBoost)** </span> 
##### <span style="color: red; ">  **7-7-2-Feature Importance with the second best model (KNN)** </span> 

![Feature importance KNN](Images/Feature_importance_KNN.PNG)
![Feature importance XGBoost](Images/Feature_importance_XGBoost.PNG)

##### <span style="color: red; ">  **7-7-3-SHAP  for the best model (XGBoost)** 
![SHAP_XGBoost](Images/SHAP_XGBoost.PNG)
                                                                                    
##### <span style="color: red; ">  **7-7-3-SHAP  for the second best model (KNN)** </span> 
![SHAP_KNN](Images/SHAP_KNN.PNG)

<div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 

# <span style="color: red; ">  **7-8- Main results from SHAP and their interpretations** </span> 
### <span style="color: red; ">  **7-8-1-Main results from SHAP and feature importance** </span> 

<div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 


- <span style="color: red; "> **1-3-Top Three features affecting the target (risk of default):** <span style="color: GREEN; ">
- <span style="color: red; "> **AGE**, 
- <span style="color: red; "> **PAY_SEPT**,
- <span style="color: red; "> **LIMIT_BAL** 
### **Other findings:**
- 4- **History of payment**: it is generally important as its elements have relatively high raking in SHAP and **three among top five ranking** (`PAY_SEP`, `PAY_ARP`, `PAY_AUG` ) are among top five features
- 5- **PAY_AMT**: It is not so important feature to affect risk of payment, but among different months, **April** has the **highest impact**, followed by **September**

- 6- **BILL-AMT**: It is not so important feature to affect risk of payment, but bill amount of **September** has the **highest impact**, followed by April

- 7- **Among six months**: **Information of September is most important followed by April**. We have the feature for each six months (history of payment, last payment, and last bill). Among all six previous months, September is most important, as PAY_AMT, BILL_AMNT_SEP, and PAY_AMNT_SEP have higher ranking from other months in similar situation. After September, April in the second important month. 

- 8- **SEX**: This feature has low impact on risk of default (target variable). Impact of male and female on target variable are low. 

- 9- **MARRIAGE**: This feature has low impact on risk of default (target variable). Impact of married and single on target variable are low. 

- 10- **EDUCATION**: This feature has low impact on risk of default (target variable). Among different groups, university degree and graduate school have higher ranking than others. 
</div>

## <span style="color: red; ">  **7-8-2- Interpretation of results**

For interpretation, we use SHAP, feature importance, and data understanding </span> 

<div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 

## <span style="color: red; "> **1- PAY_SEPT** <span style="color: black; ">
**Here is the risk of default in PAY_SEP as we explain in data understanding. Numbers are from -1 to 9, which comes back to 10 different conditions:** 
- ##### <span style="color: GREEN; "> **Number in feature`                `Means `                      `Risk of default (percent)**   
- **`    `-1`                     `paid in full`                               `17%**                            
- **`    `0`                      `paid minimum only`                        `13%**                             
- **`    `1`                      `1 moth delay`                             `34%**                              
- **`    `2`                      `2 moths delay`                            `69%**                              
- **`    `3`                      `3 moths delay`                            `76%**                            
- **`    `4`                      `4 moths delay`                            `68%**                            
- **`    `5`                      `5 moths delay`                            `50%**                             
- **`    `6`                      `6 moths delay`                            `55%**                             
- **`    `7`                      `7 moths delay`                            `78%**                            
- **`    `8`                      `8 moths delay`                            `58%**                             
- **`    `9`                      `more than 8 moths delay`                  `13%**                              

#### **Average risk of default is 48%** in PAY_SEP feature 
#### **we can categorize them with attention to the mean risk, form low risk to very high risk** 

- <span style="color: GREEN; ">**`           `Condition`              `Risk level (from low to very high)`            `Risk level (from 1 to 6)**
- **`                `paid in full`                        `Low Risk`                                    `1**
- **`                `paid minimum only`                 `Low Risk`                                      `1**
- **`                `1 moth delay`                      `Lower Middle Risk`                             `2**
- **`                `2 moths delay`                     `High Risk`                                     `5**
- **`                `3 moths delay`                     `Very High Risk`                                `6**
- **`                `4 moths delay`                     `High Risk`                                     `5**
- **`                `5moths delay`                      `Upper Middle Risk`                             `4**
- **`                `6 moths delay`                     `Upper Middle Risk`                             `4**
- **`                `7 moths delay`                     `Very High Risk`                                `6**
- **`                `8 moths delay`                     `Upper Middle Risk`                             `4**
- **`                `more than 8 moths delay`            `Low Risk`                                      `1**

#### **with attention to level of risk we can see each level of risk with attention to the conditions:**

#### `Low risk` of default:   `paid in full`  , `paid minimum only`, `more than 8 moths delay`
#### `Lower Middle Risk` of default: `one month delay`
#### `Upper Middle Risk` of default: `5, 6, 8 months delay`
#### `High Risk`  of default: `2, 4 months delay`
#### `Very High Risk`  of default: `3 , 7 months delay`


- risk of default is around 13% for full payment, but when it reaches to one month delay the risk of default increase sharply to 33%  and if delay in payment reach to two months, level of risk increases to high risk. This shows that with starting to have delay in payment, the risk of default starts to increase.
- risk of default increases sharply when delay in payment increases from one to two months delays, which shows month first is very critical. 
- from two months delay to 8 month delay, the level of risk in upper middle or high risk, which shows these months have higher level of risk. 
- If someone has default for two months, it would be possible to continue default or delay for eight months. There is a sharp increase in probability of default from one month delay to two months delay in payment, which shows one month delay in critical month, if a customer passes this critical point, the risk of default increases sharply from 33% in first month delay to 69% in second month delay. 

## <span style="color: red; "> **2- LIMIT_BAL** <span style="color: GREEN; "> 
- This feature shows credit limit (in dollar)
- It is an important feature affecting default.
- Default risk with attention to credit limit (LIMIT_BAL) is as following: 

#### **1- Probability of default by Credit Limit group, divided by 100k, from 0 to 1000k**
- <span style="color: GREEN; ">**Credit Limit**`                  `**Risk of default**
- **0-100k**`                              `**29%**
- **100k-200k**`                           `**19%**
- **200k-300k**`                           `**16%**
- **300k-400k**`                           `**14%**
- **400K-500K**`                           `**12%**
- **500k-600k**`                           `**13%**
- **600k-1000k**`                          `**8%**
  
  - **Points**
  - Credit limit is from 0-1000k
  - Maximum risk of default with attention to credit limit: 0-100k (29%)
  - Minimum risk of default with attention to credit limit: 600k-1000k (8%)
  - 100k-600k: The risk of default is 10%-20%
  - Trend (0-1000k, General): As credit limit goes up, risk of default goes down.
    
#### **2- Probability of default by Credit Limit group, divided by 100k, from 0 to 1000k**
- <span style="color: GREEN; ">**Credit Limit**`                  `**Risk of default**
- **0-10k**`                                `**40%**
- **10k-20k**`                              `**35%**
- **20k-30k**`                              `**35%**
- **30k-40k**`                              `**40%**
- **40K-50K**`                              `**26%**
- **50k-60k**`                              `**28%**
- **60k-70K**`                              `**29%**
- **70K-80K**`                              `**23%**
- **80K-90K**`                              `**25%**
- **90K-100K**`                             `**26%**

  - **Points**
  - In the range of 0-100k credit limit: 
  - Maximum risk of default with attention to credit limit: 0-10k (40%)
  - Minimum risk of default with attention to credit limit: 70K-80K (23%)
  - 30% < risk <= 40% : credit limit in the range of 0-40k
  - 20% <risk <30%: credit limit in the range of 40k-100k
  - in the range of 0-100k for credit limit, 40k is critical point, credit limit =>40k has lower risk than credit limit <40k
  - Trend (0-100k, General): As credit limit goes up, risk of default goes down.

    
  - **General Points**
  - There are two critical amount of credit limit (dollar):  40k, and 600k
   - 40k: In range 10-100k, risk of default is 30%-40% before 40k, but after 40k, risk of default is 20%-30%
   - 600k: In range 100k-1000k, risk of default is more than 10% before 600k, but after that is less than 10%
     
      
## <span style="color: red; "> **3- AGE**  <span style="color: GREEN; ">
- Age is an important feature affecting the default payment
- I categorize age into 21-30, 31-40, ...., 71-80 groups
- Age group of 31-40 has the minimum probability of default, followed by age group of 21-30, 41-50 
- Age group of 71-80 has the maximum probability of default, followed by age group 61-70. 

#### The probability of default for each group is as following: 
- <span style="color: red; ">**AGE_GROUP `       ` Risk of default**
- **21-30`               `22%**
- **31-40`               `20%**
- **41-50`               `23%**
- **51-60`               `25%**
- **61-70`               `26%**
- **71-79`               `33%**
  
#### `Low risk` of default: `31-40` 
#### `High risk` of default: `71-80` 
#### `Middle risk` of default: `51-60`
#### `Lower middle risk` of default: `21-30`, `41-50`
#### `Upper middle risk` of default: `61-70`

 - **Points**
  - Minimum risk: 31-40 age 
  - Maximum risk: 71-79 age
  - Trend (General, from 30-80): as age goes up, the risk of default goes up too. 
     

## <span style="color: red; "> **4- BILL-AMT**  <span style="color: GREEN; ">
- Bill amount shows the amount of debt (in dollar)
- Bill amount in **September** has the highest impact among six months, followed by April
- Then we here analyze only bill amount in September 

#### **1- Probability of default by bill amount group, divided by 100k, from 0 to 1000k**
- <span style="color: red; ">**Bill Amount Group`       ` Risk of default in September**
- **0-100k**`                        `**22%**
- **100k-200k**`                     `**19%**
- **200k-300k**`                     `**18%**
- **300k-400k**`                     `**23%**
- **400K-500K**`                     `**24%**
- **500k-600k**`                     `**43%**
- **600k-1000k**`                    `**20%**

   - **Points**
  - Maximum risk belongs to bill amount group of 500k-600k (43%)
  - Minimum risk of default belongs to bill amount group of 200k-300k (18%)
  - risk < 20%: bill amount in the range of 100k-300k
  - 40% <risk < 20% :  bill amount in the range of 0-100k, 300k-500k, and 600k-1000k
  - 40% > risk: bill amount in the range of 500k-600k

#### **2- Probability of default by bill amount group, divided by 10k, from 0 to 100k**
- <span style="color: red; ">**Bill Amount Group`       ` Risk of default in September**
- **0-10k**`                          `**22%**
- **10k-20k**`                        `**25%**
- **20k-30k**`                        `**24%**
- **30k-40k**`                        `**23%**
- **40K-50K**`                        `**22%**
- **50k-60k**`                        `**21%**
- **60k-70k**`                        `**21%**
- **70k-80k**`                        `**21%**
- **80k-90k**`                        `**23%**
- **90k-100k**`                       `**20%**

     - **Points**
  - Maximum risk belongs to bill amount group of 10k-20k (25%)
  - Minimum risk of default belongs to bill amount group of 90k-100k (20%)
  - Risk of default among different bill amount groups are close together. 

## <span style="color: red; "> **5- PAY-AMT**  <span style="color: GREEN; ">
- Pay amount shows the amount of payment (dollar)
- Pay amount in **April** has the highest impact among six months, followed by September
- Then we here analyze only bill amount in April 

#### **1- Probability of default by pay amount group, divided by 100k, from 0 to 400k**
- <span style="color: red; ">**Pay Amount Group`       ` Risk of default in April**
- **0-100k**`                          `**20%**
- **100k-200k**`                       `**11%**
- **200k-300k**`                       `**17%**
- **300k-400k**`                       `**20%**

    - **Points**
  - Maximum risk belongs to pay amount groups in the range of 0k-100 and 300k-400k (20%)
  - Minimum risk of default belongs to pay amount group of 100k-300k (11%)
  - (max / min) risk = 2, which is very high  

#### **2- Probability of default by pay amount group, divided by 10K, from 0 to 100K**
- <span style="color: red; ">**Pay Amount Group `       ` Risk of default in April**
- **0-10k**`                           `**21%**
- **10k-20k**`                         `**14%**
- **20k-30k**`                         `**13%**
- **30k-40k**`                         `**12%**
- **40K-50K**`                         `**11%**
- **50k-60k**`                         `**13%**
- **60k-70k**`                         `**9%**
- **70k-80k**`                         `**6%**
- **80k-90k**`                         `**13%**
- **90k-100k**`                        `**19%**
   
    - **Points**
  - Maximum risk of default: pay amount in the range of 0-10k
  - Minimum risk of default: pay amount in the range of 70k-80k
  - (max risk / min risk) = 3.5, which is very high
  - low risk range: people who pay debt in the range of 60k-80k in April have the minimum risk and are really safe groups. 

## <span style="color: red; "> **6- Information of which month is more important?**  <span style="color: GREEN; ">
- We have six months and three feature for each month, with attention to all information from SHAP, feature importance and data understanding, two months have more impact on the risk of default:
- 6-1- **September** (last month)
- 6-2- **April** (first month)

- This finding shows that if you want to check the behavior of customer check the first and last month behavior

## <span style="color: red; "> **7- SEX**  <span style="color: GREEN; ">

- It is not so important factor to affect risk of default. The risk of default with attention to gender is as following: 

- <span style="color: red; ">**Gender`            ` Risk of default** 
- **Female**`                  `**21%**
- **Male**`                    `**24%**

## <span style="color: red; "> **8- MARRIAGE**  <span style="color: GREEN; ">
-It is not so important factor to affect risk of default. The risk of default with attention to Marriage condition is as following: 

- <span style="color: red; ">**Marriage condition`            ` Risk of default** 
- **Single**`                     `**21%**
- **Married**`                    `**23%**
- **others**`                     `**24%**

## <span style="color: red; "> **9- EDUCATION**  <span style="color: GREEN; ">
- It is not so important factor to affect risk of default. The risk of default with attention to Education is as following:
- <span style="color: red; ">**Education`            ` Risk of default** 
- **High School**`                `**25%**
- **University**`                 `**24%**
- **Graduate School**`            `**19%**
- **Others**`                     `**7%**


</div>


# <span style="color: red; ">  **8- Conclusion** </span> 
## <span style="color: red; "> **8- 1- Steps of the project (what we have done?)** </span>
### 1- <span style="color: red; "> **Business understanding** </span>
### 2- <span style="color: red; "> **Data understanding** </span> 
### 3- <span style="color: red; "> **Feature Engineering** </span>
### 4- <span style="color: red; "> **train/test split and handling imbalanced dataset** </span>
### 5- <span style="color: red; "> **Machine learning models before hyperparameter tuning** </span> 
### 6- <span style="color: red; "> **Models with hyperparameter tuning (best parameters)** </span>
### 7- <span style="color: red; "> **Comparison of tuned models and best model** </span> 
- Using confusion matrix, ROC curve, recall, precision and f1, we find that among models with best parameters, **XGBoost** classifier is the best and **KNN** is the second best model. 
  
- From SHAP and feature importance we find that three factors are very important and have the highest impact on risk of default. They are: 
- LIM_SEP
- BILL_AMT
- AGE
  
### <span style="color: red; "> **8-2- Deployment** </span>
Now that we have settled on our models and findings, it is time to deliver the information to the client.  I am organizing my work as a basic report that details my primary findings.  Keep in mind that my audience are financial companies interested in fine-tuning their strategy in debt market and are sensitive to the default of payment by customers.


<div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 


### <span style="color: red; "> **1- Why we can trust to AIML model and its results and predictions**
- <span style="color: black; "> Results of AIMl models are reliable, as our best and second best models have higher accuracy, recall, precision, f1 and ROC_AUC. All criteria are more than 80%, which shows results are acceptable. 

### <span style="color: red; "> **2- What are the most important factors affecting the risk of default?** 
- <span style="color: black; "> Three top factors are affecting the risk of default are
- `AGE`,
- `PAY_SEPT`,
- and `LIMIT_BAL`

### <span style="color: red; "> **3- How `AGE` is affecting the `risk of default`, can you divide people with age on high, low and middle risk of payment?** 
- I categorize age into 21-30, 31-40, ...., 71-80
- The probability of default for each group is as following: 
- AGE_GROUP `       ` Risk of default
- 21-30`               `0.22
- 31-40`               `0.20
- 41-50`               `0.23
- 51-60`               `0.25
- 61-70`               `0.26
- 71-79`               `0.33
  
- `Low risk` of default: `31-40` 
- `High risk` of default: `71-80` 
- `Middle risk` of default: `51-60`
- `Lower middle risk` of default: `21-30`, `41-50`
- `Upper middle risk` of default: `61-70`

 - **Points**
  - Minimum risk: 31-40 age 
  - Maximum risk: 71-79 age
  - Trend (General, from 30-80): as age goes up, the risk of default goes up too.
    
### <span style="color: red; "> **4- How `PAY_SEPT` (history of last month) is affecting the `risk of default`?** 
- We can categorize different situation of history of payment in September into following five level of risk:
- `Low risk` of default:   `paid in full`  , `paid minimum only`, `more than 8 moths delay`
- `Lower Middle Risk` of default: `one month delay`
- `Upper Middle Risk` of default: `5, 6, 8 months delay`
- `High Risk` of default: `2, 4 months delay`
- `Very High Risk` of default: `3 , 7 months delay`

 - **Points**
- risk of default is around 13% for full payment, but when it reaches to one month delay the risk of default increase sharply to 33%  and if delay in payment reach to two months, level of risk increases to high risk. This shows that with starting to have delay in payment, the risk of default starts to increase.
- risk of default increases sharply when delay in payment increases from one to two months delays, which shows first month is very critical. 
- from two months delay to 8 month delay, the level of risk in upper middle or high risk, which shows these months have higher level of risk. 
- If someone has default for two months, it would be possible to continue default or delay for eight months. There is a sharp increase in probability of default from one month delay to two months delay in payment, which shows one month delay in critical month, if a customer passes this critical point, the risk of default increases sharply from 33% in first month delay to 69% in second month delay. 

### <span style="color: red; "> **5- How `LIMIT_BAL` (bill limit) is affecting the `risk of default`?** 
 
- LIMIT_BALL feature shows credit limit
- It is an important feature affecting default.
  #####
- Default risk of credit limit (LIMIT_BAL) in the range of 0-1000k: 
  - Maximum risk of default with attention to credit limit: 0-100k (29%)
  - Minimum risk of default with attention to credit limit: 600k-1000k (8%)
  - 100k-600k: The risk of default is 10%-20%
  - Trend (0-1000k, General): As credit limit goes up, risk of default goes down.
###### 
- Default risk of credit limit (LIMIT_BAL) in the range of 0-100k: 
  - Maximum risk of default with attention to credit limit: 0-10k (40%)
  - Minimum risk of default with attention to credit limit: 70K-80K (23%)
  - 30% < risk <= 40% : credit limit in the range of 0-40k
  - 20% <risk <30%: credit limit in the range of 40k-100k
  - in the range of 0-100k for credit limit, 40k is critical point, credit limit =>40k has lower risk than credit limit <40k
  - Trend (0-100k, General): As credit limit goes up, risk of default goes down.
######    
- **General Points**
  - There are two critical amount of credit limit (dollar):  40k, and 600k
   - 40k: In range 10-100k, risk of default is 30%-40% before 40k, but after 40k, risk of default is 20%-30%
   - 600k: In range 100k-1000k, risk of default is more than 10% before 600k, but after that is less than 10%
     

### <span style="color: red; "> **6- How `History of payment` is affecting the `risk of default`?** 

- Previous behavior of each customer can act as proxy for next behavior. Behavior of customer in last month (`PAY_SEP`) and in six month before (`PAY_ARP`) are very important. It means if someone started in the last month to default or from the first month of our data started to default, he/she has higher probability to default again.

### <span style="color: red; "> **7- How `BILL-AMT` is affecting the `risk of default`?** 
- Bill amount shows the amount of debt, and it has little impact on risk of default with attention to SHAP and feature important. 
- Bill amount in September has the highest impact among six months, followed by April
- Then we here analyze only bill amount in September
#####  
- Default risk with attention to Bill amount in the range of 0-600k: 
  - Maximum risk belongs to bill amount group of 500k-600k (43%)
  - Minimum risk of default belongs to bill amount group of 200k-300k (18%)
  - risk < 20%: bill amount in the range of 100k-300k
  - 40% <risk < 20% :  bill amount in the range of 0-100k, 300k-500k, and 600k-1000k
  - 40% > risk: bill amount in the range of 500k-600k

#####
- Default risk with attention to Bill amount in the range of 0-100k: 
  - Maximum risk belongs to bill amount group of 10k-20k (25%)
  - Minimum risk of default belongs to bill amount group of 90k-100k (20%)
  - Risk of default among different bill amount groups are close together. 

### <span style="color: red; "> **8- How `PAY-AMT` is affecting the `risk of default`?** 
- Pay amount shows the amount of payment, and it has little impact on risk of default with attention to SHAP and feature important. 
- Pay amount in April has the highest impact among six months, followed by September
- Then we here analyze only bill amount in April 
#####
- Default risk with attention to Bill amount in the range of 0-400k: 
  - Maximum risk belongs to pay amount groups in the range of 0k-100 and 300k-400k (20%)
  - Minimum risk of default belongs to pay amount group of 100k-300k (11%)
  - (max / min) risk = 2, which is very high  
#####
- Default risk with attention to Bill amount in the range of 0-100k: 
  - Maximum risk of default: pay amount in the range of 0-10k
  - Minimum risk of default: pay amount in the range of 70k-80k
  - (max risk / min risk) = 3.5, which is very high
  - low risk range: people who pay debt in the range of 60k-80k in April have the minimum risk and are really safe groups. 


### <span style="color: red; "> **9- Which `Month` has more valuable information about the `risk of default`?** 
- We have three data for each month (history of payment, bill amount and pay amount). Generally two important months are: 
-1 **`September`** (`last month`)
-2 **`April`** (`first month`)
- You, as financial institution, should track these two months, they represent most characters and behavior of customer
- This finding shows that if you want to check the behavior of customer check the first and last month behavior

### <span style="color: red; "> **10- How `Gender (SEX) ` is affecting the `risk of default`?** 
- It has little impact on risk of default. The risk of default, with attention to gender, is 21% for female and 24% for male. Female has lower risk than male.

### <span style="color: red; "> **11- How `MARRIAGE` is affecting the `risk of default`?** 
- It has little impact on risk of default. The risk of default, with attention to marriage condition, is 21% for single, 23% for married, and 24% for others. Single has the lowest risk of default.  

### <span style="color: red; "> **12- How `EDUCATION` is affecting the `risk of default`?** 
- It has little impact on risk of default. The risk of default, with attention to education level, is 25% for high school, 24% for university, 19% for graduate school, and 7% for others. High school degree has highest risk of default, followed by university degree. 

### <span style="color: red; "> **8-3- Suggestions for next studies and programming** </span>

<div style="background-color: LightPink; padding: 10px; border-radius: 5px;"> 

#### 1- **`More features`**
- we did not have information about `revenue`, `kind of job`, `industry are working`, `assets in exchange market`, `value of house`, 'owner or tenant', `value of cars`, `investments in banks` or `other financial institutions`, while all of them can affect the risk of default. Then using them as new features can improve the results.
#### 2- **`Other methods for imbalanced datasets`:**
-  Using other methods for imbalance datasets, such as `ADASYN (Adaptive Synthetic Sampling)`,  or `Hybrid Techniques`



