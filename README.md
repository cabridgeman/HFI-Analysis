# Human Freedom Index Analysis
Summary statistics, supervised and unsupervised machine learning analysis of G20 states using the Human Freedom Index with Python, pandas, scikit-learn, NumPy and Matplotlib

## Data Preparation

The HFI features 83 indicators about personal and economic freedom from 165 states (Cato Institute, 2023). The published datasets include both raw count statistics for certain metrics and pre-weighed proportional scores from 0 to 10 (where a higher score means ‘better’ or ‘freer from (something negative’). This report shall use the pre-weighed 0 to 10 variables as it allows for simple combining and comparison of freedom metrics without the need to calculate the proportionality of any given statistic within the context of its country and other countries. Moreover, this ensures graphs retain a consistent scale across comparisons.
```
#first step, reading the data we chose & downloaded.

HFi = pd.read_csv("hfi_cc_2022.csv")
print(HFi)
```

Cross-referencing with the dataset's documentation allowed for a refined selection of relevant safety data points. Crime statistics were able to be selected from both personal and economic freedom measures.
```
#specify even further, just want the statistics to do with crime. 

HFi_20_crime = HFi_20[["pf_ss_homicide", "pf_ss_disappearances", "pf_ss_disappearances_violent", "pf_ss_disappearances_fatalities", "pf_ss_disappearances_torture", "ef_legal_police", "countries"]]
print(HFi_20_crime)
```

This analysis uses the latest publication with findings up until 2020, published in 2022. However, as the scope of the analysis primarily concerns itself with exploring the safest countries in the G20 currently, the data is segmented to include only the latest 2020 measures. Profiles which do not constitute states, like the European Union and African Union, were omitted.
```
#specifying that we only want the data from 2020. 

HFi_20 = HFi[HFi["year"]==2020]
print(HFi_20)
```
```
#segment our data to only include G20 countries, since those are the ones we're focusing on. 

G20 = ["Argentina", "Australia", "Brazil", "Canada", "China", "France", "Germany", "India", "Indonesia", "Italy", "Japan", "Mexico", "Russian Federation", "Saudi Arabia", "South Africa", "Korea, Rep.", "Turkey", "United Kingdom", "United States"]

print(G20)

#put these stats into a dataframe that we can work with.

G20_data = pd.DataFrame(G20)
g20 = HFi_20_crime.loc[(HFi_20_crime["countries"].isin(G20))]
g20
```

## Findings and Figures
### Summary Statistics
Although the median rate for *freedom from homicide* scores positively across the G20 at 9.61, and with the upper quartile at 9.79 and lower at 8.34, it features several outliers who do not follow the rest of the G20: notably, Brazil at 3.93, Mexico at 1.75, and South Africa at 0.27. *Disappearances* in the G20, although more spread with an upper quartile at 9.61 and lower at 7.16, do not feature any outliers. Its median, at 8.80, does sit lower than its *homicide* counterpart. *Freedom from torture* does not exhibit results as positive as above, with its median at 7.75 and lowest quartile at 4.04; its maximum observation at 9.43 is bested by the *freedom from disappearance’s* top quartile at 9.61. *Torture’s* standard deviation is 2.71. For this, the results were demonstrated with a boxplot. 

```
#boxplots for 3 variables, and statistics for all variables
boxplot = g20.boxplot(column = ["pf_ss_homicide", "pf_ss_disappearances", "pf_ss_disappearances_torture"], figsize=(9,4))
boxplot.set_ylabel("Frequency")
boxplot.set_title("Homicide, Disappearances and Torture Across G20")
```

![fig3](https://github.com/cabridgeman/HFI-Analysis/assets/113062621/a091145d-e276-48e5-b1a8-1b9a72fd7069)  
---
The histograms demonstrate a near-universal positive *freedom from violent disappearances* score across the G20. It has the highest median at 10, with close second being *freedom from fatal disappearances* at 9.87 and *reliability of police* at 6.41. *Fatal disappearances* report an upper quartile of 10 and a lower of 9.72, thus following a similar positive picture as *violent disappearances*. The same cannot be reported for the *reliability of police*, however. With a standard deviation of 1.79, it fluctuates from a lowest quartile at 5.18 and minimum observation of 2.47 to highest of 7.52 and maximum of 8.74, respectively. Its median is 6.41.



```
#histogram for violent disappearances
hist3 = g20["pf_ss_disappearances_violent"].hist(figsize=(8,4))
hist3.set_xlabel("Rating for Violent Disappearances")
hist3.set_ylabel("Frequency")
hist3.set_title("Violent Disappearances Across G20")
```


![fig4](https://github.com/cabridgeman/HFI-Analysis/assets/113062621/15139cea-c8ea-43e4-8faa-b42e5e74260b)


```
#histogram for fatal disappearances
hist2 = g20["pf_ss_disappearances_fatalities"].hist(figsize=(8,4))
hist2.set_xlabel("Rating for Fatal Disappearances")
hist2.set_ylabel("Frequency")
hist2.set_title("Fatal Disappearances Across G20")
```
![fig5](https://github.com/cabridgeman/HFI-Analysis/assets/113062621/ab4fb5c9-7174-46da-b6d4-0bd41cabc411)

```
#histogram for reliability of police
hist = g20["ef_legal_police"].hist(figsize=(8,4))
hist.set_xlabel("Rating for Reliability of Police")
hist.set_ylabel("Frequency")
hist.set_title("Reliability of Police Across G20")
```

![fig6](https://github.com/cabridgeman/HFI-Analysis/assets/113062621/64c0d6e3-71ac-4e2c-b274-32a7fde9ff0f)  
---

### Supervised Methods
The secondary research question focused on how *police presence* impacts the safety of individual countries. We chose to use the supervised machine learning of linear regression. Before testing individual states, a scatter plot was created showing the overall ranking of each G20 country. The safest countries over all the data analysed are Japan, Australia, Canada and the Republic of Korea.

```
#scatter plot.

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(g20_with_ranking["Ranking"], g20_with_ranking["countries"])
plt.xlabel("Ranking (0-10)")
plt.ylabel("Country")
plt.grid()
plt.title("2020 Safety Rankings Among G20 Countries")
plt.show()
```

![fig7](https://github.com/cabridgeman/HFI-Analysis/assets/113062621/42b7fe07-f786-452f-93d6-76bfcc2416df)  

After the scatterplot, the calculations for linear regression were carried out. First, we calculated the score for the target dataset (Rankings) against the rest of the data. The score was 0.9833, which means 98.33% of the variation in the general dataset can be explained by the independent variable. Since in this case, the independent variable is an average of each country’s general data, this high score makes sense. To decide which variable to make the dependent, we first created scatterplots of each to see where there was a trend.

```
#scatterplot for police presence & overall rankings

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(g20_with_ranking["ef_legal_police"], g20_with_ranking["Ranking"])
plt.axis([2, 11, 2, 11])
plt.xlabel("Police Presence (Scale: 1-10)")
plt.ylabel("Ranking")
plt.title("The Impact of Police Presence on Safety in the G20")
plt.grid()
plt.show()
```
As can be seen, there is not a clear relationship between the two factors, and so we can conclude that police presence does not have much of an impact on freedom from violent disappearances. The one outlier in this data is Turkey, with a mid-level police presence (5.57), and the lowest score on freedom from violent disappearances (7.59).  
  
![fig8](https://github.com/cabridgeman/HFI-Analysis/assets/113062621/56ed6d40-d621-417f-9c80-b4edcf9e4064)  

The next category is *freedom from disappearances*. There is more of a relationship here; when police presence increases, freedom from disappearance also increases.

```
#scatterplot for police & disappearances
#scatter plot cont.

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(g20_with_ranking["ef_legal_police"], g20_with_ranking["pf_ss_disappearances"])
plt.axis([2, 11, 2, 11])
plt.xlabel("Police Presence (Scale: 1-10)")
plt.ylabel("Freedom From Disappearances (Scale: 1-10)")
plt.title("The Impact of Police Presence on Disappearances")
plt.grid()
plt.show()

```



![fig9](https://github.com/cabridgeman/HFI-Analysis/assets/113062621/32f13d5b-8a59-4608-ad1a-3ea6e9d1a191)  

The third category is the *freedom from instances of torture within disappearances*. This relationship is less clear - a trend appears to exist, however, there are multiple outliers, with China and South Africa being the clearest.

```
#scatterplot for police & torture
#scatter plot cont.

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(g20_with_ranking["ef_legal_police"], g20_with_ranking["pf_ss_disappearances_torture"])
plt.axis([2, 11, 2, 11])
plt.xlabel("Police Presence (Scale: 1-10)")
plt.ylabel("Freedom from Torture Related to Disappearances (Scale: 1-10)")
plt.title("The Impact of Police Presence on Instances of Torture Related to Disappearances")
plt.grid()
plt.show()
```

![fig10](https://github.com/cabridgeman/HFI-Analysis/assets/113062621/80dbc5af-c6c0-47d9-89f9-0abaf0fb3f55)  

The fourth category is *freedom from fatal disappearances*. There is significantly less of a trend in this case, with the exception of one outlier: Turkey.

```
#scatterplot for police & fatalities
#scatter plot cont.

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(g20_with_ranking["ef_legal_police"], g20_with_ranking["pf_ss_disappearances_fatalities"])
plt.axis([2, 11, 2, 11])
plt.xlabel("Police Presence (Scale: 1-10)")
plt.ylabel("Freedom from Fatality Related to Disappearances (Scale: 1-10)")
plt.title("The Impact of Police Presence on Instances of Fatalities (Related to Disappearances)")
plt.grid()
plt.show()
```


![fig11](https://github.com/cabridgeman/HFI-Analysis/assets/113062621/d2db57e7-6d56-4259-9d2e-3e5f4aeacf38)  

The final category is *freedom from homicide*. These two variables appeared to have more of a trend, and as such, we hypothisised that it would be best for our linear regression analysis. To be sure, however, we performed a linear regression calculation for each relationship. It was discovered that this final category did have the most significant relationship, with *police presence* able to explain approximately 56.28% of the variations in *freedom from homicide*.   
  
*(One linear regression example)*

```
#linear regression between homicide and police presence - police presence as the indepedent, 
#homicide as dependent.

import sklearn.linear_model
model = sklearn.linear_model.LinearRegression()
X = np.c_[g20_with_ranking["ef_legal_police"]]
y = np.c_[g20_with_ranking["pf_ss_homicide"]]
# Train the model
model.fit(X, y)
```


```
#scatterplot - for homicides - some effect from police

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(g20_with_ranking["ef_legal_police"], g20_with_ranking["pf_ss_homicide"])
plt.axis([2, 11, 2, 11])
plt.xlabel("Police Presence (Scale: 1-10)")
plt.ylabel("Freedom from Homicide Rating (Scale: 1-10)")
plt.title("The Impact of Police Presence on Freedom from Homicide (Scale: 1-10)")
plt.grid()
plt.show()

```
![fig12](https://github.com/cabridgeman/HFI-Analysis/assets/113062621/19a51392-babd-4d75-8e50-0dca8df1cb77)  
---


Using these calculations, we created a new scatterplot with a line of best fit. This demonstrates a positive trend in the majority of the G20 countries: when *police presence* increases, *homicide* rates decrease.

```
t0, t1 = model.intercept_[0], model.coef_[0][0]
g20_with_ranking.plot(kind='scatter', x="ef_legal_police", y='pf_ss_homicide', figsize=(6,4))
plt.axis([0, 11, 0, 11])
X=np.linspace(0, 60000, 1000) #generate 1000 values between 0 and 60000 to create the line
plt.plot(X, t0 + t1*X, "r")
plt.title("Impact of Police Presence on Freedom from Homicide in G20 Countries")
plt.xlabel("Police Presence (Scale: 1-10)")
plt.ylabel("Freedom from Homicide (Scale: 1-10)")
plt.grid()
plt.show()
```
![fig13](https://github.com/cabridgeman/HFI-Analysis/assets/113062621/f5a013ad-7787-47dc-9fe5-59928c90a338)  
---

### Unsupervised Methods
It is hypothesized that the supervised methods section will be more useful, as much of our data is labelled, however we still carried out the Kmeans clustering calculations to test this hypothesis. Whilst a silhouette coefficient test, completeness score, calinski-harabasz coefficient and homogeneity scores will tell us the optimal number of clusters and how accurate they are.

```
# This confirms the shape of the dataset
n_samples, n_features = newg20.shape
print("number of rows:", n_samples)
print("number of features:", n_features)

# This gives us the number of different values in the target dataset
n_digits = len(np.unique(Rankings))

# n_digits will be used as the number of clusters 
print("number of different values for the target:", n_digits)

# this will allow the create the KMeans Model
kmeans = cluster.KMeans(n_clusters=19)

# Once created, now we can fit the model to the data
kmeans.fit_predict(data)
print(kmeans.get_params())
```

Silhouette coefficient scores illustrate how compact a cluster is. If it is more compact, then the data is more accurately organised in that cluster, and is more clearly related to its own cluster in comparison to the other clusters. This is demonstrated through a score between -1 and 1, with 1 being perfectly compact, and -1 being not compact at all. The lowest silhouette score came out at 0.59737, and the highest at 0.76043. These are relatively high scores, meaning the clusters are (decently) compact.

```
n_samples, n_features = newg20.shape
n_digits = len(np.unique(newg20))
Y2 = LabelEncoder().fit_transform(newg20)
array_silhouette = []
array_completeness = []
array_homogeneity = []
array_Calinski_harabasz = []
clusters = []
for k in range(2, 20):
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(newg20)
    array_silhouette.append(metrics.silhouette_score(newg20, kmeans.labels_))
    array_completeness.append(metrics.completeness_score(Y2, kmeans.labels_))
    array_homogeneity.append(metrics.homogeneity_score(Y2, kmeans.labels_))
    array_Calinski_harabasz.append(metrics.calinski_harabasz_score(newg20, kmeans.labels_))
    clusters.append(k)
    print(k)
    print("silhouette_score = ", metrics.silhouette_score(newg20, kmeans.labels_))
    print("completeness_score = ", metrics.completeness_score(Y2, kmeans.labels_))
    print("homogeneity_score = ", metrics.homogeneity_score(Y2, kmeans.labels_))
    print("Calinski_harabasz Coefficient:", metrics.calinski_harabasz_score(newg20, kmeans.labels_))
```
Homogeneity scores illustrate how similar points are in a specific cluster. The more similar they are, the higher the score will be. This is also on a scale of 0 to 1. These scores were not as high, with the lowest being 0.113 and the highest being 0.6195. This indicates that the clusters were not as homogenous, and so likely were not all from the same class.  

The completeness score indicates whether all data points from a specific class are from the same cluster. This score was 1.0 for nearly all of the clusters – again, this makes sense, since we were analysing all of the data as a whole, rather than specific classes. As such, this score was not as relevant, but still useful to ensure the clustering was carried out correctly.  

```
spoints = array_silhouette
hpoints = array_homogeneity
cpoints = array_Calinski_harabasz
copoints= array_completeness

plt.plot(clusters, hpoints, linestyle = 'dotted')
plt.plot(clusters, spoints, linestyle = 'dotted')
plt.plot(clusters, copoints, linestyle = 'dotted')
plt.title("Kmeans Clustering Scores for Safety in G20")
plt.ylabel("Score")
plt.xlabel("Number of Clusters")
plt.grid()
plt.show()
```


Finally, the Calinski-harabasz Coefficient is a sum which illustrates how far away a cluster is from the other clusters. The Calinski- harabasz Coefficient for our data were relatively spread out: they ranged from 254 to 3306.

We graphed each of the first three scores on a line plot. The green line represents Completeness score, the yellow line represents the Silhoueae score, and the blue line represents the homogeneity score. The Calinski-harabasz coefficient is not included below, as it has a scale from 0-4000, and for visualisation purposes, we can learn more without it in the figure. Since we analysed for 2 to 19 clusters, the highest number indicates the best number of clusters. According to the graph, the optimal number of clusters would have been 18, as this is where the silhoueae score and homogeneity score intersect.

![fig14](https://github.com/cabridgeman/HFI-Analysis/assets/113062621/39d4c44c-7426-40e8-b0a7-d70ef4c3ca2b)  
