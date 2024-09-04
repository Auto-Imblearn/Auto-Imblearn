import pandas as pd
import numpy as np
import argparse
import os

def insert_missing_values(input_file, missing_proportion):
    # Load the dataset
    df = pd.read_csv(input_file, header=None)
    df_subset = df.iloc[:, :-1]

    # Get the total number of elements in the DataFrame
    total_elements = df_subset.size

    # Calculate the number of missing values to insert
    num_missing = int(total_elements * missing_proportion)

    # Randomly choose positions in the DataFrame to insert NaN values
    nan_indices = (
        np.random.choice(df_subset.index, num_missing, replace=True),
        #  np.random.choice(df_subset.columns, num_missing, replace=True)
        np.random.choice(np.arange(len(df_subset.columns)), num_missing, replace=True),
    )

    # Insert NaN values at the selected positions
    #  df_subset.values[nan_indices] = np.nan
    for i in range(num_missing):
        row, col = nan_indices[0][i], nan_indices[1][i]
        df_subset.iat[row, col] = np.nan


    df.iloc[:, :-1] = df_subset
    df.columns = df.columns.astype(str)

    # Generate the output file name by appending '-missing' to the input file name (without extension)
    output_file = os.path.splitext(input_file)[0] + '-missing.csv'

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

    missing_percentage = (df.isnull() | df.isin([' '])).mean() * 100
    print("\nPercentage of missing values in each column:")
    print(missing_percentage)

    print(f"\nMissing values inserted (excluding the last column). Output saved to {output_file}")

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Insert missing values into a dataset.')

    # Add arguments for the input file and missing proportion
    parser.add_argument('input_file', type=str, help='The name of the input CSV file.')
    parser.add_argument('--missing_proportion', type=float, default=0.1, help='Proportion of missing values to insert (default is 0.1 or 10%).')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    insert_missing_values(args.input_file, args.missing_proportion)

