#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset = pd.read_csv(r"C:/Users/srupa/Downloads/unified mentor/2ND PROJECT/Amazon Sales data.csv")
dataset.head()


# In[3]:


dataset.shape


#  ## Data Description
#   
#   we have 14 columns and 100 rows
#   
#     Region: Geographical area where the sale occurred.
#     Country: Specific country within the region where the sale took place.
#     Item Type: Category of the product sold.
#     Sales Channel: Medium through which the sale was made (Online or Offline).
#     Order Priority: Priority level assigned to the order (e.g., High, Critical, Low).
#     Order Date: Date when the order was placed.
#     Order ID: Unique identifier for each order.
#     Ship Date: Date when the order was shipped.
#     Units Sold: Number of units sold in the order.
#     Unit Price: Selling price per unit of the item.
#     Unit Cost: Cost incurred per unit of the item.
#     Total Revenue: Total sales revenue generated from the order.
#     Total Cost: Total cost incurred for the units sold.
#     Total Profit: Profit earned from the sale of the units.
# 
# 

# In[4]:


dataset.info()


# In[5]:


## here date columns are identified as the object 


# In[6]:


dataset['Order Date'] = pd.to_datetime(dataset['Order Date'], format='%m/%d/%Y')
dataset['Ship Date'] = pd.to_datetime(dataset['Ship Date'], format='%m/%d/%Y')


# In[7]:


# Check the data types to confirm the conversion
print(dataset.dtypes)


# In[8]:


dataset.describe()


# In[9]:


dataset.drop(columns = ["Order ID"], inplace = True)


# In[10]:


## checking for the null values
dataset.isnull().sum()


# In[11]:


## checking for the dublicate value
dataset.duplicated().sum()


# In[21]:


dataset.dtypes


# In[22]:


## checking for the correlation
numeric_cols = ["Units Sold","Unit Price","Unit Cost","Total Revenue","Total Cost","Total Profit"]               

numeric_cols


# In[23]:


dataset[numeric_cols].corr()


# ## we can see unit_cost and unit_price are highly correlated and total revenue is highly correlated with total cost and total profit 
# `

# In[24]:


dataset[numeric_cols].kurtosis()


# In[25]:


dataset[numeric_cols].skew()


# ## Data Visualization

# In[26]:


dataset["Region"].nunique() ## 7 different regions


# In[27]:


sns.countplot(data = dataset, x = "Region")


# In[28]:


dataset["Country"].nunique()


# In[29]:


dataset["Item Type"].nunique()


# In[30]:


dataset["Item Type"].value_counts()


# In[31]:


sns.countplot(data=dataset, x='Item Type')


# In[32]:


dataset["Sales Channel"].value_counts()


# In[33]:


sns.countplot(data = dataset, x = "Sales Channel")


# In[34]:


dataset["Order Priority"].value_counts()


# In[35]:


sns.countplot(data = dataset, x = "Order Priority")


# In[36]:


for col in numeric_cols:
    plt.subplot()
    plt.figure(figsize=(4, 3))
    plt.hist(dataset[col], bins=6, edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


# In[37]:


num_plots = len(numeric_cols)
num_cols = 3  # Number of columns in the subplot grid
num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
axes = axes.flatten()

# Plot each histogram in a different subplot
for i, col in enumerate(numeric_cols):
    axes[i].boxplot(dataset[col])
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True)

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()


# In[38]:


## there are outliers present in 3 columns
columns_with_outliers = ("Total Revenue", "Total Cost", "Total Profit")


# In[39]:


## using IQR for treating the outliers
q3 =  dataset["Total Revenue"].quantile(0.75)
q1 = dataset["Total Revenue"].quantile(0.25)
IQR = q3 - q1
IQR


# In[40]:


upper_limit = q3 + 1.5*(IQR)
lower_limit = q1 - 1.5*(IQR)

dataset["Total Revenue"] = np.where(dataset["Total Revenue"]> upper_limit, upper_limit, 
                                    np.where(dataset["Total Revenue"]< lower_limit,lower_limit,dataset["Total Revenue"]))


# In[41]:


# total cost column
q3 =  dataset["Total Cost"].quantile(0.75)
q1 = dataset["Total Cost"].quantile(0.25)
IQR = q3 - q1
IQR
upper_limit = q3 + 1.5*(IQR)
lower_limit = q1 - 1.5*(IQR)

dataset["Total Cost"] = np.where(dataset["Total Cost"]> upper_limit, upper_limit, 
                                    np.where(dataset["Total Cost"]< lower_limit,lower_limit,dataset["Total Cost"]))


# In[42]:


# Total profit column
q3 =  dataset["Total Profit"].quantile(0.75)
q1 = dataset["Total Profit"].quantile(0.25)
IQR = q3 - q1
IQR
upper_limit = q3 + 1.5*(IQR)
lower_limit = q1 - 1.5*(IQR)

dataset["Total Profit"] = np.where(dataset["Total Profit"]> upper_limit, upper_limit, 
                                    np.where(dataset["Total Profit"]< lower_limit,lower_limit,dataset["Total Profit"]))


# In[43]:


plt.boxplot(dataset["Total Profit"])


# In[44]:


## outliers are treated 


# In[45]:


# Plotting distribution of units sold
plt.figure(figsize=(8, 6))
plt.hist(dataset['Units Sold'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Distribution of Units Sold')
plt.xlabel('Units Sold')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[46]:


## scatter plot for camparing the unitsolds and total profit 


# In[47]:


# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(dataset['Total Revenue'], dataset['Total Cost'], color='skyblue', alpha=0.7)
plt.title('Total Revenue vs Total Cost')
plt.xlabel('Total Revenue')
plt.ylabel('Total Cost')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# In[48]:


sns.scatterplot(data = dataset, x = "Units Sold", y = "Total Profit", hue = "Item Type")


# In[49]:


most_sold = dataset.groupby('Item Type')['Units Sold'].sum().idxmax()
print(f"The most sold item type is: {most_sold}")


# In[50]:


highest_profit = dataset.groupby('Item Type')['Total Profit'].sum().idxmax()
print(f"The item type with the highest profit is: {highest_profit}")


# In[51]:


# Calculate total profit for each item type
item_total_profit = dataset.groupby('Item Type')['Total Profit'].sum()

# Plotting
plt.figure(figsize=(10, 6))
item_total_profit.plot(kind='bar', color='skyblue')
plt.title('Total Profit by Item Type')
plt.xlabel('Item Type')
plt.ylabel('Total Profit')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# ## we can see the highest sold product and product that make highest profit is Cosmetics

# In[52]:


sns.barplot(data=dataset, x='Item Type', y='Total Revenue', palette='Set2')
plt.title('Total Revenue by Item Type')
plt.xlabel('Item Type')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[53]:


highest_revenue_item = dataset.groupby('Item Type')['Total Revenue'].sum().idxmax()
highest_revenue_value = dataset.groupby('Item Type')['Total Revenue'].sum().max()

print(f"The item type generating the highest revenue is: {highest_revenue_item}")
print(f"Highest revenue generated: {highest_revenue_value}")


# In[54]:


plt.figure(figsize=(10, 6))
plt.plot(dataset['Order Date'], dataset['Units Sold'], marker='o', linestyle='-', color='b')
plt.title('Units Sold over Time')
plt.xlabel('Order Date')
plt.ylabel('Units Sold')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[55]:


# days to ship
dataset['Days to Ship'] = (dataset['Ship Date'] - dataset['Order Date']).dt.days

# average days to ship
average_days_to_ship = dataset['Days to Ship'].mean()

print(f"On average, products were shipped after {average_days_to_ship:.2f} days.")



# In[56]:


# distribution od days to ship
plt.figure(figsize=(8, 5))
plt.hist(dataset['Days to Ship'], bins=10, edgecolor='black')
plt.title('Distribution of Days to Ship')
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[57]:


# Find region with highest profit
highest_profit_region = dataset.groupby('Region')['Total Profit'].sum().idxmax()
highest_profit_value = dataset.groupby('Region')['Total Profit'].sum().max()

# Find region with highest units sold
highest_sale_region = dataset.groupby('Region')['Units Sold'].sum().idxmax()
highest_sale_value = dataset.groupby('Region')['Units Sold'].sum().max()

print(f"Region with highest profit: {highest_profit_region} (Profit: {highest_profit_value})")
print(f"Region with highest sale: {highest_sale_region} (Units Sold: {highest_sale_value})")


# In[58]:


# Calculate total profit for each Region
country_total_profit = dataset.groupby('Region')['Total Profit'].sum()

# Plotting
plt.figure(figsize=(8, 8))
plt.pie(country_total_profit, labels=country_total_profit.index, autopct='%1.1f%%', startangle=140)
plt.title('Total Profit Distribution by Region')
plt.show()


# ## region Sub-Saharan Africa has the highest profit and sale 

# In[59]:


#  sales channel with highest sales
highest_sales_channel = dataset.groupby('Sales Channel')['Units Sold'].sum().idxmax()
highest_sales_value = dataset.groupby('Sales Channel')['Units Sold'].sum().max()

print(f"The sales channel with the highest sales is: {highest_sales_channel} (Units Sold: {highest_sales_value})")


# ## the products were sold more on offline mode 

# In[60]:


region_total_profit = dataset.groupby('Region')['Total Profit'].sum()

# Find region with highest total profit
highest_profit_region = region_total_profit.idxmax()

print("Total Profit by Region:")
print(region_total_profit)


# In[61]:


# Calculate total profit for each sales channel
channel_total_profit = dataset.groupby('Sales Channel')['Total Profit'].sum()

# Plotting
plt.figure(figsize=(8, 8))
plt.pie(channel_total_profit, labels=channel_total_profit.index, autopct='%1.1f%%', startangle=140)
plt.title('Total Profit Distribution by Sales Channel')
plt.show()


# In[62]:


# Calculate total profit for each country
country_total_profit = dataset.groupby('Country')['Total Profit'].sum()

# Find country with highest total profit
highest_profit_country = country_total_profit.idxmax()
highest_profit_value = country_total_profit.max()

print(f"The country with the highest total profit is: {highest_profit_country} (Total Profit: {highest_profit_value})")


# ## country with highest profit is Djibouti

# In[63]:


## total profit change over time
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(dataset['Order Date'], dataset['Total Profit'], marker='o', color='orange', linestyle='-', linewidth=2)
plt.title('Total Profit Over Time')
plt.xlabel('Order Date')
plt.ylabel('Total Profit')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[64]:


# dropping order_date and ship_date since we have created the Days to Ship column which can represent both the columns
## dropping country column

dataset.drop(columns = ["Country", "Order Date", "Ship Date"], inplace = True)


# In[78]:


# unit_cost and unit_price are highly correlated and total revenue is highly correlated with total cost and total profit
dataset.drop(columns = ["Unit Cost", "Total Revenue"], inplace = True)


# In[79]:


dataset.columns


# In[80]:


columns_of_interest = ['Region', 'Item Type', 'Sales Channel', 'Order Priority', 
                        'Units Sold', 'Unit Price',  
                       'Days to Ship', 'Total Profit']

# Create a pairplot for pairwise relationships with Total Profit
sns.pairplot(dataset[columns_of_interest])
plt.show()


# ## seperating x and y variable 

# In[81]:


x = dataset.drop(columns = ["Total Profit"])
x


# In[82]:


y = dataset["Total Profit"]
y


# In[83]:


from sklearn.model_selection import train_test_split, GridSearchCV


# In[84]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = (42))
x_train.shape, x_test.shape


# In[85]:


## applying one hot encoding 


# In[86]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first',sparse_output=False,dtype=np.int32 )


# In[87]:


columns_to_encode = ['Region', 'Item Type', 'Sales Channel', 'Order Priority']


# In[88]:


x_train_new = ohe.fit_transform(x_train[['Region','Item Type',"Sales Channel", "Order Priority" ]])
x_train_new


# In[91]:


# Convert the encoded columns to a DataFrame
encoded_df = pd.DataFrame(x_train_new, columns=ohe.get_feature_names_out(columns_to_encode))

# Reset the index of the encoded DataFrame to match X_train
encoded_df.reset_index(drop=True, inplace=True)

# Drop the original columns from X_train
x_train_dropped = x_train.drop(columns=columns_to_encode)

# Reset the index of X_train_dropped to match encoded_df
x_train_dropped.reset_index(drop=True, inplace=True)

# Concatenate the encoded DataFrame with the rest of X_train
x_train_new = pd.concat([x_train_dropped, encoded_df], axis=1)


# In[92]:


x_train_new


# In[93]:


x_test_new = ohe.transform(x_test[['Region','Item Type',"Sales Channel", "Order Priority" ]])
x_test_new


# In[94]:


# Convert the encoded columns to a DataFrame
encoded_df = pd.DataFrame(x_test_new, columns=ohe.get_feature_names_out(columns_to_encode))

# Reset the index of the encoded DataFrame to match X_train
encoded_df.reset_index(drop=True, inplace=True)

# Drop the original columns from X_train
x_test_dropped = x_test.drop(columns=columns_to_encode)

# Reset the index of X_train_dropped to match encoded_df
x_test_dropped.reset_index(drop=True, inplace=True)

# Concatenate the encoded DataFrame with the rest of X_train
x_test_new = pd.concat([x_test_dropped, encoded_df], axis=1)


# In[95]:


x_test_new


# ## Model Building

# In[96]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[97]:


# Initialize the RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)



# In[98]:


# Train the model
rf_model.fit(x_train_new, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(x_test_new)


# In[99]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")


# In[70]:


## cross validation for getting best model 


# In[100]:


# Define the parameter grid for GridSearchCV
from sklearn.tree import DecisionTreeRegressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}



# In[101]:


# Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)


# In[103]:


# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit GridSearchCV to find the best parameters
grid_search.fit(x_train_new, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Train the RandomForestRegressor with the best parameters
best_rf = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_rf.predict(x_test_new)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")


# In[104]:


# Plotting y_pred vs y_test
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot(y_test, y_test, color='red', linestyle='--')  # Diagonal line for reference
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[105]:


from sklearn.tree import DecisionTreeRegressor


# Initialize the DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42, criterion="squared_error", max_depth=30, min_samples_split=2, min_samples_leaf=2)


# In[106]:


# Fit GridSearchCV to find the best parameters
dt.fit(x_train_new, y_train)

# Evaluate the model on the test set
y_test_pred = dt.predict(x_test_new)

# Calculate test metrics

test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Test Mean Absolute Error: {test_mae}")
print(f"Test R^2 Score: {test_r2}")


# In[107]:


# Plotting actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.5)
plt.plot(y_test, y_test, color='red', linestyle='--')  # Diagonal line for reference
plt.title('Actual vs Predicted (Decision Tree Regression)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Random Forest gave the best model with the accuracy of 93 percentage with params: {'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
# 
# 
# 

# ### saving the best model 

# In[108]:


from joblib import dump

# Save the best model to a file
dump(best_rf, 'best_random_forest_model.joblib')


# In[109]:


dump(ohe, "one_hot_encoding.joblib")


# In[ ]:




