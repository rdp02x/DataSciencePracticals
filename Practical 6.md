Loading the Dataset
```python
# Import necessary libraries
import pandas as pd

# Load the data into a dataframe
data = pd.read_csv('lendingdata.csv')
```

1. Determine the number of columns with 'object' data type
```python
object_columns = data.select_dtypes(include=['object']).columns
num_object_columns = len(object_columns)
print(f"Number of columns with 'object' data type: {num_object_columns}")
print(f"Columns with 'object' data type: {object_columns.tolist()}")
```

2. Find the total number of missing values in the dataset
```python
total_missing_values = data.isnull().sum().sum()
print(f"Total number of missing values in the data set: {total_missing_values}")
```

3. Identify and drop redundant columns
```python
# Assuming 'country_code' is redundant because 'country' is present
columns_to_drop = ['country_code']
data = data.drop(columns=columns_to_drop)
print(f"Columns after dropping redundant information: {data.columns.tolist()}")
```

4. Calculate the third quartile value (Q3) of the variable 'loan_amount'
```python
third_quartile_value = data['loan_amount'].quantile(0.75)
print(f"Third quartile value of the variable 'loan_amount': {third_quartile_value}")
```

5. Calculate the percentage split of different categories in 'repayment_interval'
```python
# Drop rows with missing values in 'repayment_interval'
repayment_data = data['repayment_interval'].dropna()

# Calculate the percentage split of different categories
percentage_split = repayment_data.value_counts(normalize=True) * 100
print("Percentage split of different categories in 'repayment_interval':")
print(percentage_split)
```

6. Find the minimum loan amount disbursed in the Agriculture sector
```python
min_loan_amount_agriculture = data[data['sector'] == 'Agriculture']['loan_amount'].min()
print(f"Minimum loan amount disbursed in the Agriculture sector: {min_loan_amount_agriculture}")
```
