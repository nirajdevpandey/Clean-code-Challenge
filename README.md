### Clean-code-Challenge
The repository contains the solution of BlueYonder GmbH's challenge for data scientist position. Implementing a regression model on bike-sharing data-set to predict count of future rentals.

Table of Contents
=================
* Data-set description 
* Data Summary
* Feature Engineering
* Missing Value Analysis
* Correlation Analysis
* Visualizing Distribution Of Data
* Visualizing Count Vs (Month,Season,Hour,Weekday,Usertype)
* Fitting the model 
* Results

#### Data-set description 
Bike sharing systems are a means of renting bicycles where the process of obtaining membership, rental, and bike return is automated via a network of kiosk locations throughout a city. Using these systems, people are able rent a bike from a one location and return it to a different place on an as-needed basis. Our target is to predict the remtal count given the independent variables

#### Data Summary

Here you can see what is inside the data

#### Simple Visualization Of Variables number Count
all 4 Seasons seem to have eqaul count

It is quit obvious that there would be less holiday and more of working day in 2 years of time. 

Working day is having sattistics. `remember` in plots `0:False`and `1:True`

Weather Count is as follows 
`1:spring`
`2:summer`
`3:fall`
`4:winter`

Which nonth had the highest demand 

Which was the pick hour for renting the bike

What temperature was best preferred for the ride



#### Feature Engineering
You see! the columns "season","holiday","workingday" and "weather" should be of "categorical" data type.But the current data type is "int" for those columns. Let us transform the dataset in the following ways so that we can get started up with our `EDA` (Exloratory Data Analysis). 

### Missing Value Analysis
Let's see if there is any `missing` on `NA` values in the entire dataset. SO, we dont have any missing value in the dataset. Yeeey...!!

#### Correlation Analysis

To understand how a dependent variable is influenced by features (numerical) is to get a correlation matrix between them. Lets plot a correlation plot between "count" and ["temp","atemp","humidity","windspeed"].

>temp and humidity features has got positive and negative correlation with count respectively.Although the correlation between them are not very prominent still the count variable has got little dependency on "temp" and "humidity".

>windspeed is not gonna be really useful numerical feature and it is visible from it correlation value with "count"

>"atemp" is variable is not taken into since "atemp" and "temp" has got strong correlation with each other. During model building any one of the variable has to be dropped since they will exhibit multicollinearity in the data.

>"Casual" and "Registered" are also not taken into account since they are leakage variables in nature and need to dropped during model building.

### Visualizing Count Vs (Month,Season,Hour,Weekday,Usertype)
Looking at the following plot we can get some useful information. 

>It is quiet obvious that people tend to rent bike during summer season since it is really conducive to ride bike at that season.Therefore June, July and August has got relatively higher demand for bicycle.

>On weekdays more people tend to rent bicycle around 7AM-8AM and 5PM-6PM. As we mentioned earlier this can be attributed to regular school and office commuters.

>Above pattern is not observed on "Saturday" and "Sunday".More people tend to rent bicycle between 10AM and 4PM.















