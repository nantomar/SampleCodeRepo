import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

def df_data_type(df):
    # Define the required column data types and check if all the columns are in right format
    required_dtypes = {
        'fraud_bool': 'int64'.
        'income': 'float64',
        'name_email_similarity': 'float64',
        'prev_address_months_count': 'int64',
        'current_address_months_count': 'int64',
        'customer_age': 'int64',
        'days_since_request': 'float64',
        'intended_balcon_amount': 'float64',
        'payment_type': 'object',
        'zip_count_4w': 'int64',
        'velocity_6h': 'float64',
        'velocity_24h': 'float64',
        'velocity_4w': 'float64',
        'bank_branch_count_8w': 'float64',
        'date_of_birth_distinct_emails_4w': 'float64',
        'employment_status': 'object',
        'credit_risk_score': 'float64',
        'email_is_free': 'float64',
        'housing_status': 'object',
        'phone_home_valid': 'float64',
        'phone_mobile_valid': 'float64',
        'bank_months_count': 'float64',
        'has_other_cards': 'float64',
        'proposed_credit_limit': 'float64',
        'foreign_request': 'float64',
        'source': 'object',
        'session_length_in_minutes': 'float64',
        'device_os': 'object',
        'keep_alive_session': 'float64',
        'device_distinct_emails_8w': 'float64',
        'device_fraud_count': 'float64',
        'month': 'float64'
    }

    # Check if the column exists and if the current dtype matches the required dtype
    for column, required_dtype in required_dtypes.items():
        if column in df.columns:
            current_dtype = df[column].dtype
            if current_dtype != required_dtype:
                print(f"Column '{column}' dtype mismatch: {current_dtype} (expected {required_dtype})")
        else:
            print(f"Column '{column}' is missing from the DataFrame.")
    
    return df


def data_sanity(df):
  rowthresh=0.6
  colthresh=0.3

  if df is not None:
    # Check for duplicated rows
    duplicated_rows = df.duplicated().sum()
    print(f"Number of duplicated rows: {duplicated_rows}")

    # Check for missing values in rows and columns
    missing_rows = df.isnull().sum(axis=1)
    missing_cols = df.isnull().sum(axis=0)

    # Identify rows with more than thres
    miss_thresh_rows =missing_rows[missing_rows > rowthresh * df.shape[1]]
    print(f"Number of rows with more than 60% missing data: {len( miss_thresh_rows)}")

    # Identify columns with more than thresh
    miss_thresh_cols= missing_cols[missing_cols > colthresh* df.shape[0]]
    print(f"Number of columns with more than 30% missing data: {len(miss_thresh_cols)}")

    return miss_thresh_rows.index, miss_thresh_cols.index

def var_type(df):
    """
    This function var_type classifies the columns of a DataFrame into
     different types: boolean, categorical, numeric, string, and zero_variance.
      It checks the data type of each column and assigns it to the appropriate category
       based on its characteristics, such as unique values or variance. 
       The function returns a dictionary with the column types categorized accordingly.
    """
    boolean = []
    categorical = []
    numeric = []
    string = []
    zero_variance=[]

    for x in df.columns:
        if pd.api.types.is_numeric_dtype(df[x]):
        # checking data type of columns to find categorical and boolean cols
            if df[x].nunique() == 2:
                boolean.append(x)
            if df[x].var() == 0:
                zero_variance.append(x)
            else :
                numeric.append(x)

        elif df[x].dtype == 'object':
            if len(df[x].unique())<=10:
               categorical.append(x)
            else:
                string.append(x)
    return {"boolean":boolean,
            "categorical":categorical,
            "numeric":numeric,"string":string}


def bivariate_analysis(df, X, Y, print_summary=False):
    """
    This function performs bivariate analysis by creating a scatter plot,
    fitting a linear regression model, and plotting the regression line.

    Parameters:
    - df: pandas DataFrame containing the data.
    - X: string, name of the independent variable column.
    - Y: string, name of the dependent variable column.
    """

    # Prepare data for OLS regression
    X_data = sm.add_constant(df[X])  # Add a constant term for the intercept
    y_data = df[Y]

    # Fit the linear regression model
    model = sm.OLS(y_data, X_data).fit()

    if print_summary:
        print(model.summary())

    # Plot the regression line on the scatter plot
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X, y=Y, data=df)
    plt.plot(df[X], model.predict(X_data), color='red', label='Regression Line')
    plt.title(f'Regression Line: {X} vs {Y}')
    plt.legend()
    plt.show()

def correlation_check(df, var_list, threshold=0, print_map=False):
    """
    This function plots a correlation heatmap, but only shows correlations above a given threshold.

    Parameters:
    - df: pandas DataFrame containing the data.
    - numeric_cols: list of column names to include in correlation.
    - threshold: correlation threshold; only correlations with an absolute value above this threshold will be displayed.
    - print_map: boolean flag, if True the heatmap will be plotted.
    """
    # Compute correlation matrix for numeric columns
    correlation = df[var_list].corr()

    # Mask correlations below the threshold
    correlation_masked = correlation.applymap(lambda x: x if abs(x) >= threshold else None)

    # Print correlation values where the threshold is met (if needed)
    if threshold != 0:
        print(f"Variables with correlation >= {threshold} or <= {-threshold}:")
        # Find the column pairs with correlations above the threshold (ignoring NaN values)
        for col1 in correlation_masked.columns:
            for col2 in correlation_masked.index:
               if col1 != col2 :
                if pd.notna(correlation_masked.loc[col2, col1]):
                    print(f"{col1} and {col2}: {correlation_masked.loc[col2, col1]}")

    # If the print_map flag is True, plot the heatmap
    if print_map:
        plt.figure(figsize=(20, 15))
        sns.heatmap(correlation_masked, annot=True, cmap='coolwarm', linewidths=2)
        plt.tight_layout()
        plt.show()

def num_dist_plot(varlist, df, target):
    num_plots = len(varlist)
    ncols = 2
    nrows = (num_plots + ncols - 1) // ncols  # Round up to the nearest whole number

    # Create the subplots grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 5 * nrows))
    axes = axes.flatten()

    # Loop through the numeric features and plot a kernel density plot for each
    for i, feature in enumerate(varlist):
            ax = axes[i]
            sns.kdeplot(data=df[df[target] == 0][feature], fill=True, ax=ax, label='Not Fraud', color='blue',warn_singular=False)
            sns.kdeplot(data=df[df[target] == 1][feature], fill=True, ax=ax, label='Fraud', color='red',warn_singular=False)
            ax.set_xlabel(feature)
            ax.set_title(f"Distribution of {feature}")


    # Remove any empty subplots (if any)
    for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])


    plt.tight_layout()
    plt.show()

def negative_missing(df, varlist, Y):
    """
    This function checks for columns in `varlist` where negative values are used to represent missing data.
    It replaces negative values with NaN and then calculates the percentage of missing values for each column,
    grouped by the specified column `Y`.

    Parameters:
    - df: pandas DataFrame, the dataset
    - varlist: list of column names to check for negative values (representing missing data)
    - Y: string, the name of the column to group by (e.g., 'fraud_bool')

    Returns:
    - None: The function directly prints a DataFrame with the missing values percentages.
    """
    missing_vals = pd.DataFrame()
    # Loop through each column in the varlist to check and process missing values
    for var in varlist:


        # Replace negative values with NaN
      if pd.api.types.is_numeric_dtype(df[var]):
        df.loc[df[var] < 0, var] = np.nan

        # Calculate the percentage of missing values grouped by Y
        missing_vals_col = df.groupby(Y)[var].apply(lambda x: round(x.isna().sum()/len(x) * 100, 2))

        # Add the result to the missing_vals DataFrame
        missing_vals[var] = missing_vals_col

    # Reshape missing_vals DataFrame from wide to long format
    missing_vals = pd.DataFrame(missing_vals.T.stack())

    # Reset the index and rename columns for better clarity
    missing_vals.reset_index(inplace=True)
    missing_vals.rename(columns={'level_0': 'feature', 0: 'missing_vals'}, inplace=True)

    # Print the final table
    print(missing_vals)
