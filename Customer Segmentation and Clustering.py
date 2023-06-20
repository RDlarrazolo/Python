#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation & Clustering
# - Identify the best possible cluster using KMeans unsupervised machine learning algorithm to fund the univariate, bivariate and multivariate clusters.
# - Summarize Statistics to identify the best marketing group
# 

# Importing various libraries necessary for data manipulation, visualization, clustering, and handling warnings during code execution.

# In[1]:


#importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings 
warnings.filterwarnings('ignore')


# Reading the data from a CSV file named 'Mall_Customers.csv' using pandas and stores it in a DataFrame called 'df'.

# In[2]:


df = pd.read_csv('Mall_Customers.csv')


# In[3]:


df.head()


# # Univariate Analysis

# An overview of the central tendency, spread, and distribution of the numerical variables in the dataset.

# In[4]:


df.describe()


# Generating a histogram with a KDE plot to visualize the distribution of annual incomes of customers in a dataset.

# In[5]:


sns.distplot(df['Annual Income (k$)']);


# In[6]:


df.columns


# Generates a separate plot for each column specified in the columns list, 
# showing the distribution of data in that column using a combination of a 
# histogram and a KDE.

# In[7]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns: 
    plt.figure()
    sns.distplot(df[i])


# Creates a KDE plot to visualize the distribution of annual income in a dataset called 'Mall_Customers.csv'. The plot is shaded to represent the density of data points, and separate curves are shown for different genders.

# In[11]:


sns.kdeplot(data=df, x='Annual Income (k$)', shade=True, hue='Gender')
plt.show()


# Creates KDE plots for the 'Age', 'Annual Income (k$)', and 'Spending Score (1-100)' columns

# In[14]:


columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.kdeplot(data=df, x=i, shade=True, hue='Gender')

plt.show()


# Creates individual box plots for the variables 'Age', 'Annual Income (k$)', and 'Spending Score (1-100)'. Each box plot is categorized by the gender of the customers.

# In[16]:


columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df, x='Gender',y=df[i])


# In[18]:


df['Gender'].value_counts()


# In[20]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis

# Creates a scatter plot using the seaborn library to visualize the relationship between the annual income and spending score of mall customers.

# In[23]:


sns.scatterplot(data=df, x='Annual Income (k$)',y='Spending Score (1-100)' );


# Preparing the dataset by dropping the 'CustomerID' column and then creating a pairplot to analyze the bivariate relationships between different variables. The scatter plots in the pairplot are colored based on the 'Gender' variable, allowing for visual exploration of the relationships between variables and gender in the dataset.

# In[27]:


df=df.drop('CustomerID',axis=1)
sns.pairplot(df,hue='Gender')


# Grouping the data by gender and calculating the average age, annual income, and spending score for each gender group.

# In[28]:


df.groupby(['Gender'])['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# Calculating and displaying the correlation matrix for age, annual income, and spending score, enabling bivariate analysis of their relationships.

# In[29]:


df.corr()


# In[32]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# # Clustering - Univariate, Bivariate, Miltivariate

# Initializing a K-means clustering algorithm with three clusters to analyze the variables of age, annual income, and spending score. It aims to group customers with similar annual income and spending score patterns into distinct clusters.

# In[41]:


clustering1 = KMeans(n_clusters=3)


# Performing clustering analysis on the 'Annual Income (k$)' variable.
# It uses the selected variable to train a clustering algorithm and find patterns or groups in the data based on annual income.

# In[42]:


clustering1.fit(df[['Annual Income (k$)']])


# Assigning cluster labels for each customer.

# In[44]:


clustering1.labels_


# Assigning cluster labels to the dataset based on the clustering algorithm and displaying the modified df with the added 'Income Cluster' column.

# In[45]:


df['Income Cluster'] = clustering1.labels_
df.head()


# Analyzing the frequency of different clusters in the 'Income Cluster' column, providing insights into the distribution of income clusters in the dataset.

# In[46]:


df['Income Cluster'].value_counts()


# Calculating the sum of squared distances between data points and their respective cluster centroids, providing an indication of how well the data points are grouped within the clusters. 

# In[51]:


clustering1.inertia_


# Calculating the inertia score for each clustering, which is a measure of how well the data points within each cluster are compact and separated from other clusters. The inertia scores are then stored in the 'inertia_scores' list for further analysis or visualization.

# In[55]:


inertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)


# In[56]:


inertia_scores


# Creating a line plot where the x-axis values range from 1 to 10, and the y-axis values are the scores stored in the "inertia_scores" variable. The plot is a visual representation of how these scores change across the range of x-axis values.

# In[57]:


plt.plot(range(1,11),inertia_scores)


# In[58]:


df.columns


# In[61]:


df.groupby('Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# # Bivariate Clustering

# Performing bivariate clustering on the 'Annual Income' and 'Spending Score' variables. It assigns cluster labels to each data point based on their similarity in terms of income and spending behavior. The resulting cluster labels are added as a new column in the DataFrame.

# In[70]:


clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
df['Spending and Income Cluster'] =clustering2.labels_
df.head()


# Performing K-means clustering on the 'Annual Income' and 'Spending Score' variables of the mall customer dataset. It calculates the inertia scores for different numbers of clusters (ranging from 1 to 10) and plots these scores on a graph. The inertia scores help us understand the optimal number of clusters to use for the dataset.

# In[71]:


intertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    intertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11),intertia_scores2);


# Calculating the cluster centers using a bivariate clustering algorithm and stores the coordinates of these centers in a DataFrame with columns labeled 'x' and 'y.'

# In[74]:


centers =pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']


# Creating a scatter plot showing the relationship between the annual income and spending score of customers.

# In[75]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x ='Annual Income (k$)',y='Spending Score (1-100)',hue='Spending and Income Cluster',palette='tab10')
plt.savefig('clustering_bivaraiate.png')


# Analyzing how gender is related to spending and income clusters in the mall customer dataset. It shows the proportion or percentage of each gender within each cluster, giving an understanding of gender distribution across different customer segments.

# In[76]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# Grouping the data based on the "Spending and Income Cluster" variable and calculates the average age, annual income, and spending score for each cluster.

# In[77]:


df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# # Mulivariate clustering

# Importing the necessary library for data preprocessing and creating a scaler object. This preprocessing step is essential for preparing the data before applying multivariate clustering techniques.

# In[79]:


from sklearn.preprocessing import StandardScaler


# In[80]:


scale = StandardScaler()


# In[81]:


df.head()


# Converting categorical variables in the dataset into numerical variables so that they can be used for multivariate clustering analysis. 

# In[82]:


dff = pd.get_dummies(df,drop_first=True)
dff.head()


# In[83]:


dff.columns


# Selecting specific columns from the dataset "dff" that contain information about customers' age, annual income, spending score, and gender.

# In[84]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)','Gender_Male']]
dff.head()


# Standardizing the variables

# In[85]:


dff = scale.fit_transform(dff)


# Applying scaling and standardization to the variables (age, annual income, and spending score) using the scale.fit_transform() function. The resulting scaled and standardized data is stored in a new DataFrame called dff. 

# In[86]:


dff = pd.DataFrame(scale.fit_transform(dff))
dff.head()


# Calculating the inertia scores for different numbers of clusters and plots them on a line graph to identify the optimal number of clusters for the dataset.

# In[87]:


intertia_scores3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    intertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11),intertia_scores3)


# In[88]:


df


# In[89]:


df.to_csv('Clustering.csv')


# # Analysis ( Target Cluster ) 

# Based on the analysis of the clustering results, the target cluster for marketing purposes would be Cluster 1. This cluster consists of customers who have both a high spending score and a high annual income. These customers are likely to be more affluent and willing to spend on products or services.
# 
# Additionally, it is worth noting that 54 percent of the shoppers in Cluster 1 are women. This information can be valuable for targeting marketing campaigns towards this specific demographic within the cluster.
# 
# To attract these customers, a marketing campaign can be designed that highlights popular items or services that align with the preferences of Cluster 1. By focusing on the specific needs and interests of this cluster, businesses can tailor their offerings and messaging to effectively capture their attention and encourage them to make purchases.
# 
# On the other hand, Cluster 2 also presents an interesting opportunity for marketing efforts. While the specific characteristics of Cluster 2 are not mentioned in the code, it is suggested that this cluster may consist of customers who exhibit distinct patterns or behaviors that make them suitable for targeted marketing.
# 
# For example, it is mentioned that Cluster 2 customers may be interested in sales events on popular items. This information can be utilized to create marketing campaigns that emphasize special offers, discounts, or promotions specifically tailored for Cluster 2 customers. By understanding the preferences and tendencies of this cluster, businesses can optimize their marketing strategies to maximize engagement and sales.
# 
# In summary, based on the clustering analysis, targeting Cluster 1 customers with high spending scores and high incomes, as well as designing marketing campaigns to appeal to their preferences, would be a strategic approach. Additionally, exploring opportunities to market to Cluster 2 customers who may be interested in sales events on popular items can also be advantageous.

# 
