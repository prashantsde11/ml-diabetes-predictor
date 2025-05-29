import pandas as pd  # Import pandas for data manipulation

def load_and_clean_data():
    # Load the Pima Indians Diabetes dataset from the given URL
    # Assign column names explicitly since the CSV doesn't have headers
    df = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        names=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
        ]
    )

    # Columns where zero values are invalid and should be treated as missing data
    cols_with_zeroes = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # Replace 0 values in these columns with pandas' NA to mark them as missing
    df[cols_with_zeroes] = df[cols_with_zeroes].replace(0, pd.NA)

    # Fill missing values (NA) with the median value of each column
    # Median is a robust statistic to handle missing data without skewing too much
    df.fillna(df.median(), inplace=True)

    # Return the cleaned DataFrame ready for modeling
    return df
