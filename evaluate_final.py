#!/usr/bin/env python
# coding: utf-8

# In[11]:


import nbformat
import pandas as pd
import base64
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
from zoneinfo import ZoneInfo
import io
import sys
import pandas.testing as pdt

# File paths and user metadata
problem_notebook_path = "/Users/rakeshdevarakonda/Documents/Auto_Eval/Linear_regression/Problem-Milestone1.ipynb"
solution_notebook_path = "/Users/rakeshdevarakonda/Documents/Auto_Eval/Linear_regression/Solution-Milestone1.ipynb"

user_email = 'user1@abc.com'
attempt_id = '23'
project = 'project1'

task_weightage = {
    "load_the_dataset": 2, "get_name_value_counts": 2, "get_city_value_counts": 2,
    "missing_value_check": 2, "missing_value_treatment": 2, "box_plot1": 2,
    "treat_outliers_iqr": 2, "box_plot2": 2, "clean_car_sales_data": 1, "corr": 2,
    "perform_anova": 5, "seller_type_influence_test": 5, "chi_square_test": 5,
    "drop_var1": 2, "city_region": 1, "drop_var2": 1, "encode_categorical_columns": 1,
    "save_file": 1, "load_the_cleaned_dataset": 1, "separate_data_and_target": 1,
    "split_into_train_and_test_normalize_features": 4, "multicollinearity": 4,
    "fit_the_model_ols": 10, "fit_the_model_on_the_training_data_2": 10,
    "ols_calculate_rmse_r2": 2, "test_the_sklearn_model": 2,
    "calculate_r_squared_sklearn": 3, "calculate_rmse_sklearn": 3,
    "drop_variables": 1, "target_transform": 2, "separate_data_and_target_new": 2,
    "split_into_train_and_test_normalize_features_2": 2,
    "fit_the_model_on_the_training_data_2": 2, "test_the_finalmodel": 2,
    "calculate_r_squared_finalmodel": 2, "calculate_rmse_finalmodel": 2
}

threshold_rules = {
    "ols_calculate_rmse_r2": {"r2": (0.60, 0.80), "rmse": (400000, 500000)},
    "calculate_r_squared_sklearn": {"r2": (0.60, 0.80)},
    "calculate_rmse_sklearn": {"rmse": (400000, 500000)},
    "calculate_r_squared_finalmodel": {"r2": (0.70, 0.90)},
    "calculate_rmse_finalmodel": {"rmse": (0.20, 0.50)}
}

def extract_function_outputs_with_call(notebook_path, valid_functions):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    output_dict = {}
    last_defined_func = None
    func_call_found = set()

    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        lines = cell.source.strip().split('\n')
        for line in lines:
            if line.strip().startswith("def "):
                last_defined_func = line.split("(")[0].replace("def ", "").strip()
                break
        for func in valid_functions:
            if func in cell.source and f"def {func}" not in cell.source:
                last_defined_func = func
                break

        if last_defined_func in valid_functions and last_defined_func not in func_call_found:
            combined_output = []
            for output in cell.get('outputs', []):
                if output.output_type == "execute_result" and "text/plain" in output.data:
                    combined_output.append(output.data["text/plain"])
                elif output.output_type == "stream" and "text" in output:
                    combined_output.append(output["text"])
                elif output.output_type == "error":
                    combined_output.append("\n".join(output["traceback"]))
                elif output.output_type == "display_data" and "text/plain" in output.data:
                    combined_output.append(output.data["text/plain"])
            if combined_output:
                output_dict[last_defined_func] = "\n".join(combined_output).strip()
                func_call_found.add(last_defined_func)
            elif last_defined_func not in output_dict:
                output_dict[last_defined_func] = None

    return pd.DataFrame([{"Function": func, "Output": out} for func, out in output_dict.items()])

def compare_outputs_with_patch(problem_df, solution_df, task_weightage):
    def clean_summary(text):
        lines = text.strip().splitlines()
        return "\n".join(" ".join(line.split()) for line in lines if not line.strip().startswith(("Date:", "Time:")))

    def parse_numeric_output(output):
        try:
            return float(output.strip())
        except:
            return None

    scores = []
    solution_dict = dict(zip(solution_df["Function"], solution_df["Output"]))

    for _, row in problem_df.iterrows():
        function_name = row["Function"]
        problem_output = row["Output"]
        solution_output = solution_dict.get(function_name)

        score = 0
        remark = "Missing or not executed"

        if solution_output and problem_output:
            try:
                if function_name == "fit_the_model_ols":
                    def split_output(output):
                        split_index = output.lower().find("model summary")
                        return (output[:split_index].strip(), output[split_index:].strip()) if split_index != -1 else (output.strip(), "")

                    prob_test, prob_summary = split_output(problem_output)
                    sol_test, sol_summary = split_output(solution_output)

                    test_data_match = prob_test == sol_test
                    summary_match = clean_summary(prob_summary) == clean_summary(sol_summary)

                    if test_data_match and summary_match:
                        score = task_weightage.get(function_name, 0)
                        remark = "Success"
                    else:
                        remark = "Mismatch in test_data or summary"
                elif function_name in threshold_rules:
                    numeric_value = parse_numeric_output(problem_output)
                    thresholds = threshold_rules[function_name]
                    success = False
                    if numeric_value is not None:
                        for metric, (low, high) in thresholds.items():
                            if low <= numeric_value <= high:
                                success = True
                                break
                    if success:
                        score = task_weightage.get(function_name, 0)
                        remark = "Success"
                    else:
                        remark = f"Value {numeric_value} out of range"
                else:
                    df1 = eval(problem_output, {"pd": pd, "np": np})
                    df2 = eval(solution_output, {"pd": pd, "np": np})
                    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
                        pd.testing.assert_frame_equal(df1, df2, check_dtype=False, atol=1e-6)
                        score = task_weightage.get(function_name, 0)
                        remark = "Success"
                    else:
                        if problem_output.strip() == solution_output.strip():
                            score = task_weightage.get(function_name, 0)
                            remark = "Success"
                        else:
                            remark = problem_output
            except Exception:
                if problem_output.strip() == solution_output.strip():
                    score = task_weightage.get(function_name, 0)
                    remark = "Success"
                else:
                    remark = problem_output
        elif problem_output:
            remark = problem_output

        scores.append({
            "method_name": function_name,
            "score_gained": score,
            "remarks": remark
        })

    return pd.DataFrame(scores)

def extract_images_from_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    images, errors = [], {}
    for cell in nb.cells:
        if cell.cell_type == 'code':
            func = None
            for line in cell.source.strip().split('\n'):
                if line.startswith("def "):
                    func = line.split("(")[0].replace("def ", "").strip()
            for output in cell.get('outputs', []):
                if output.output_type == 'display_data' and 'image/png' in output.data:
                    img_data = base64.b64decode(output.data['image/png'])
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                    images.append((func, np.array(img)))
                elif output.output_type == 'error' and func:
                    errors[func] = "\n".join(output["traceback"])
    return images, errors

def compare_images_with_func(problem_images, solution_images, threshold=0.99):
    image_results = {}
    for p_func, p_img in problem_images:
        matched = False
        for s_func, s_img in solution_images:
            if p_func == s_func:
                min_shape = (min(p_img.shape[0], s_img.shape[0]), min(p_img.shape[1], s_img.shape[1]))
                p_resized = Image.fromarray(p_img).resize(min_shape[::-1])
                s_resized = Image.fromarray(s_img).resize(min_shape[::-1])
                p_gray = np.array(p_resized.convert('L'))
                s_gray = np.array(s_resized.convert('L'))
                similarity, _ = ssim(p_gray, s_gray, full=True)
                matched = similarity >= threshold
                break
        image_results[p_func] = matched
    return image_results

# Step 1: Extract outputs
problem_outputs = extract_function_outputs_with_call(problem_notebook_path, task_weightage.keys())
solution_outputs = extract_function_outputs_with_call(solution_notebook_path, task_weightage.keys())

# Step 2: Compare outputs
output_scores = compare_outputs_with_patch(problem_outputs, solution_outputs, task_weightage)

# Step 3: Merge scores with metadata
task_df = pd.DataFrame(list(task_weightage.items()), columns=['method_name', 'max_score'])
merged_df = output_scores.merge(task_df, on='method_name', how='right')
merged_df['UserEmail'] = user_email
merged_df['attempt_id'] = attempt_id
merged_df['timestamp'] = datetime.now().astimezone(ZoneInfo("Asia/Kolkata"))
merged_df['project'] = project

# Step 4: Compare visual outputs
problem_images, problem_errors = extract_images_from_notebook(problem_notebook_path)
solution_images, _ = extract_images_from_notebook(solution_notebook_path)
image_comparisons = compare_images_with_func(problem_images, solution_images, threshold=0.99)

for plot_func in ["box_plot1", "box_plot2"]:
    if plot_func in task_weightage:
        if plot_func in image_comparisons and image_comparisons[plot_func]:
            merged_df.loc[merged_df["method_name"] == plot_func, "score_gained"] = task_weightage[plot_func]
            merged_df.loc[merged_df["method_name"] == plot_func, "remarks"] = "Success"
        elif plot_func in problem_errors:
            merged_df.loc[merged_df["method_name"] == plot_func, "score_gained"] = 0
            merged_df.loc[merged_df["method_name"] == plot_func, "remarks"] = problem_errors[plot_func]
        else:
            merged_df.loc[merged_df["method_name"] == plot_func, "score_gained"] = 0
            merged_df.loc[merged_df["method_name"] == plot_func, "remarks"] = "Image didn't match"

# Final output
score_df = merged_df[['UserEmail','attempt_id','method_name','score_gained','max_score','timestamp','remarks','project']]
#score_df
# Display output (if in notebook)
#score_df

import sys
import pandas as pd
import mysql.connector
 
# Get user email from Node.js (passed as an argument)
if len(sys.argv) < 4:
    print("User email not provided")
    sys.exit(1)
 


# In[23]:


# MySQL Connection Setup
db_config = {
    "host": 'arshniv.cuceurst1z3t.us-east-1.rds.amazonaws.com',
    "user": 'admin',
    "password": 'arshnivdb',
    "database":'autovmharbor'
}


# In[24]:


def insert_results_into_db(df):
    try:
        # Connect to MySQL
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Insert each row into MySQL
        for _, row in df.iterrows():
            sql = """INSERT INTO assignment_results 
                     (UserEmail, attempt_id, method_name, score_gained, max_score,timestamp,remarks,project)
                     VALUES (%s, %s, %s, %s, %s, now(), %s,%s)"""
            values = (
                row["UserEmail"], 
                row["attempt_id"], 
                row["method_name"], 
                row["score_gained"], 
                row["max_score"],
                row['remarks'],
                row['project']
            )
            cursor.execute(sql, values)

        # Commit and close
        conn.commit()
        cursor.close()
        conn.close()
        print("Results inserted into database successfully.")

    except Exception as e:
        print("Error inserting results into database:", e)


# In[25]:


insert_results_into_db(score_df)

