#!/usr/bin/env python
# coding: utf-8

# In[69]:


## Final Eval Score ####
import nbformat
import pandas as pd
from datetime import datetime
# File paths
problem_notebook_path = "/Users/rakeshdevarakonda/Documents/Auto_Eval/problem/problem.ipynb"
solution_notebook_path = "/Users/rakeshdevarakonda/Documents/Auto_Eval/solution/Solution.ipynb"
#user_email = 'user1@example.com'
# Get user email from Node.js (passed as an argument)
if len(sys.argv) < 2:
    print("User email not provided")
    sys.exit(1)
 
user_email = sys.argv[1]
attempt_id = '3'

# Task weightage list
task_weightage = {
    "load_the_dataset": 2,
    "process_store_data": 1,
    "find_unique_values": 1,
    "total_sales": 1,
    "check_missing_values": 1,
    "sales_distribution": 1,
    "top_customer_segment": 1,
    "regional_purchasing_behavior": 1,
    "high_spending_regions": 1,
    "popular_product_categories": 1,
    "avg_quantity_per_transaction": 1,
    "quantity_sales_relationship": 1,
    "category_quantity_sales_trends": 1,
    "highest_profit_segments": 1,
    "clean_and_calculate_shipping_time": 1,
    "calculate_discounted_price": 1,
    "calculate_revenue_per_day": 1,
    "identify_most_discounted_products": 2,
    "analyze_revenue_efficiency": 2,
    "anova_sales_by_category": 2,
    "ttest_sales_by_segment": 2,
    "treat_outliers_iqr": 1,
    "apply_outlier_treatment": 1,
    "compute_correlations": 1,
    "drop_var1": 1,
    "normalize_numeric_columns": 1,
    "one_hot_encode": 2
}

def extract_function_outputs_fixed(notebook_path, valid_functions):
    """
    Extracts function outputs from a Jupyter notebook, filtering only those
    in the valid_functions list (task_weightage keys).
    
    Returns a DataFrame containing function names and corresponding outputs.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    function_outputs = []
    current_function = None

    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell_outputs = []

            # Extract function name from the code cell
            lines = cell.source.split("\n")
            for line in lines:
                if line.strip().startswith("def "):
                    function_name = line.split("(")[0].replace("def ", "").strip()
                    if function_name in valid_functions:
                        current_function = function_name
                    else:
                        current_function = None
                    break
            
            # Extract outputs if function is valid
            if current_function:
                for output in cell.get('outputs', []):
                    if 'text' in output:
                        cell_outputs.append(output['text'].strip())
                    elif 'data' in output and 'text/plain' in output['data']:
                        cell_outputs.append(output['data']['text/plain'].strip())
                    elif 'traceback' in output:
                        cell_outputs.append("ERROR: " + "\n".join(output['traceback']))

                # Store function name and outputs if available
                if cell_outputs:
                    formatted_output = " ".join(cell_outputs).replace("\n", " ").strip()
                    function_outputs.append({"Function": current_function, "Output": formatted_output})

    return pd.DataFrame(function_outputs)

def compare_outputs(problem_df, solution_df, task_weightage):
    """
    Compares function outputs from problem and solution dataframes.
    Assigns a score based on task weightage if outputs match, otherwise assigns 0.
    
    Returns a dataframe with function names and scores.
    """
    scores = []

    # Convert solution dataframe to dictionary for quick lookup
    solution_dict = dict(zip(solution_df["Function"], solution_df["Output"]))

    for _, row in problem_df.iterrows():
        function_name = row["Function"]
        problem_output = row["Output"]
        solution_output = solution_dict.get(function_name, None)

        # Check if the outputs match and assign score accordingly
        if solution_output and problem_output.strip() == solution_output.strip():
            score = task_weightage.get(function_name, 0)
        else:
            score = 0

        scores.append({"Function": function_name, "Score": score})

    return pd.DataFrame(scores)

# Extract function outputs, considering only those present in task_weightage
problem_file = extract_function_outputs_fixed(problem_notebook_path, task_weightage.keys())
solution_file = extract_function_outputs_fixed(solution_notebook_path, task_weightage.keys())

# Compare outputs and calculate scores
score_df = compare_outputs(problem_file, solution_file, task_weightage)

score_df = score_df.rename(columns={'Function': 'method_name', 'Score': 'score_gained'})

# Convert dict to DataFrame
task_df = pd.DataFrame(list(task_weightage.items()), columns=['method_name', 'max_score'])

# Perform inner join
merged_df = score_df.merge(task_df, on='method_name', how='inner')
score_df = merged_df[['method_name', 'score_gained', 'max_score']]
# Define values
# Add columns to the DataFrame
score_df['UserEmail'] = user_email
score_df['attempt_id'] = attempt_id
score_df['timestamp'] = datetime.now()
score_df = score_df[['UserEmail','attempt_id','method_name','score_gained','max_score','timestamp']]


# In[70]:


import sys
import pandas as pd
import mysql.connector
 
# Get user email from Node.js (passed as an argument)
if len(sys.argv) < 2:
    print("User email not provided")
    sys.exit(1)
 


# In[71]:


# MySQL Connection Setup
db_config = {
    "host": 'arshniv.cuceurst1z3t.us-east-1.rds.amazonaws.com',
    "user": 'admin',
    "password": 'arshnivdb',
    "database":'vmharbor'
}


# In[72]:


def insert_results_into_db(df):
    try:
        # Connect to MySQL
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Insert each row into MySQL
        for _, row in df.iterrows():
            sql = """INSERT INTO assignment_results 
                     (UserEmail, attempt_id, method_name, score_gained, max_score, timestamp)
                     VALUES (%s, %s, %s, %s, %s, NOW())"""
            values = (
                row["UserEmail"], 
                row["attempt_id"], 
                row["method_name"], 
                row["score_gained"], 
                row["max_score"]
            )
            cursor.execute(sql, values)

        # Commit and close
        conn.commit()
        cursor.close()
        conn.close()
        print("Results inserted into database successfully.")

    except Exception as e:
        print("Error inserting results into database:", e)
#insert_results_into_db(score_df)


# In[73]:


insert_results_into_db(score_df)


# In[ ]:




