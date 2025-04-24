#!/usr/bin/env python
# coding: utf-8

# In[11]:


import nbformat
import base64
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import sys 
import sys
import pandas as pd
import mysql.connector

# File paths and metadata
problem_notebook_path = "/home/user3/project1/problem.ipynb"
solution_notebook_path = "/home/vmuser/project1/solution.ipynb"
user_email = sys.argv[1]
attempt_id = sys.argv[2]
project = sys.argv[3]

# Task weightage
task_weightage = {
    "read_csv": 3,
    "display_dataset_info": 2,
    "check_missing_values": 2,
    "compute_basic_statistics": 2,
    "plot_histograms": 2,
    "plot_correlation_heatmap": 2,
    "plot_outcome_counts": 2
}

# --- Extract outputs from notebook (execute_result + stream) ---
def extract_function_outputs_fixed(notebook_path, valid_functions):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    output_dict = {}
    current_func = None

    for cell in nb.cells:
        if cell.cell_type == 'code':
            lines = cell.source.split('\n')
            for line in lines:
                if line.strip().startswith("def "):
                    current_func = line.split("(")[0].replace("def ", "").strip()

            if current_func in valid_functions:
                for output in cell.get('outputs', []):
                    # Capture DataFrame output
                    if output.output_type == "execute_result" and "text/plain" in output.data:
                        output_dict[current_func] = output.data["text/plain"]
                    # Capture print output or df.info()
                    elif output.output_type == "stream" and output.name == "stdout":
                        output_dict[current_func] = output.text

    return pd.DataFrame([
        {"Function": func, "Output": out} for func, out in output_dict.items()
    ])

# --- Compare textual outputs ---
def compare_outputs(problem_df, solution_df, task_weightage):
    scores = []
    solution_dict = dict(zip(solution_df["Function"], solution_df["Output"]))

    for _, row in problem_df.iterrows():
        function_name = row["Function"]
        problem_output = row["Output"]
        solution_output = solution_dict.get(function_name, None)

        if solution_output and problem_output.strip() == solution_output.strip():
            score = task_weightage.get(function_name, 0)
            remark = "Success"
        else:
            score = 0
            remark = problem_output

        scores.append({
            "method_name": function_name,
            "score_gained": score,
            "remarks": remark
        })

    return pd.DataFrame(scores)

# --- Extract images per function ---
def extract_images_by_function(notebook_path, valid_functions):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    func_image_map = {}
    current_func = None

    for cell in nb.cells:
        if cell.cell_type == 'code':
            lines = cell.source.split('\n')
            for line in lines:
                if line.strip().startswith("def "):
                    current_func = line.split("(")[0].replace("def ", "").strip()

            if current_func in valid_functions:
                for output in cell.get('outputs', []):
                    if output.output_type == 'display_data' and 'image/png' in output.data:
                        img_data = base64.b64decode(output.data['image/png'])
                        img = Image.open(pd.io.common.BytesIO(img_data)).convert('RGB')
                        func_image_map.setdefault(current_func, []).append(np.array(img))

    return func_image_map

# --- Compare image lists by SSIM ---
def compare_images(imgs1, imgs2, threshold=0.95):
    for img1 in imgs1:
        for img2 in imgs2:
            min_shape = (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1]))
            img1_resized = Image.fromarray(img1).resize(min_shape[::-1]).convert('L')
            img2_resized = Image.fromarray(img2).resize(min_shape[::-1]).convert('L')
            similarity, _ = ssim(np.array(img1_resized), np.array(img2_resized), full=True)
            if similarity >= threshold:
                return True
    return False

# --- Run Evaluation Pipeline ---

# 1. Text outputs
problem_outputs = extract_function_outputs_fixed(problem_notebook_path, task_weightage.keys())
solution_outputs = extract_function_outputs_fixed(solution_notebook_path, task_weightage.keys())
output_scores = compare_outputs(problem_outputs, solution_outputs, task_weightage)

# 2. Merge with metadata
task_df = pd.DataFrame(list(task_weightage.items()), columns=['method_name', 'max_score'])
merged_df = output_scores.merge(task_df, on='method_name', how='outer')  # include all methods
merged_df['UserEmail'] = user_email
merged_df['attempt_id'] = attempt_id
merged_df['timestamp'] = datetime.now()

merged_df['project'] = project
merged_df['score_gained'] = merged_df['score_gained'].fillna(0).astype(int)
merged_df['remarks'] = merged_df['remarks'].fillna("Function not found")

# 3. Image comparison
problem_images_map = extract_images_by_function(problem_notebook_path, task_weightage.keys())
solution_images_map = extract_images_by_function(solution_notebook_path, task_weightage.keys())

for method in task_weightage:
    if method in problem_images_map and method in solution_images_map:
        match = compare_images(problem_images_map[method], solution_images_map[method])
        if match:
            merged_df.loc[merged_df["method_name"] == method, "score_gained"] = task_weightage[method]
            merged_df.loc[merged_df["method_name"] == method, "remarks"] = "Success"
        else:
            merged_df.loc[merged_df["method_name"] == method, "score_gained"] = 0
            merged_df.loc[merged_df["method_name"] == method, "remarks"] = "Image mismatch"

# 4. Finalize results
score_df = merged_df[['UserEmail','attempt_id','method_name','score_gained','max_score','timestamp','remarks','project']]
score_df = score_df.reset_index(drop=True)

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

