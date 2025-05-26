#import nbformat
import pandas as pd
import base64
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from difflib import SequenceMatcher
from datetime import datetime
from zoneinfo import ZoneInfo
from io import BytesIO
import sys
import pandas as pd
import mysql.connector

# File paths and metadata
problem_notebook_path = "/Users/rakeshdevarakonda/Documents/Auto_Eval/Auto_eval2/Retail_sales_classification_Tredence_problem.ipynb"
solution_notebook_path = "/Users/rakeshdevarakonda/Documents/Auto_Eval/Auto_eval2/Retail_sales_classification_Tredence_solution.ipynb"

user_email = "user@example.com"
attempt_id = "1"
project = "python_problem1"

task_weightage = {
    "read_dataset": 2,
    "df_shape": 0.5,
    "df_dtypes": 0.5,
    "drop_columns": 2,
    "outlier_treatment": 3,
    "transpose_1": 0.5,
    "treat_outliers_iqr": 6,
    "transpose_2": 0.5,
    "missing_value": 2,
    "remove_value": 2
}

# --- Function to Extract Outputs (from Calls) ---
def extract_function_outputs_including_calls(notebook_path, valid_functions):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    output_dict = {fn: {"text": "__NO_OUTPUT__", "image": None} for fn in valid_functions}
    
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell_source = cell.source
            for fn in valid_functions:
                if fn in cell_source:
                    for output in cell.get('outputs', []):
                        if output.output_type == "execute_result" and "text/plain" in output.data:
                            output_dict[fn]["text"] = output.data["text/plain"]
                        elif output.output_type == "stream" and output.name == "stdout":
                            output_dict[fn]["text"] = output.text.strip()
                        elif output.output_type == "display_data" and "image/png" in output.data:
                            image_data = base64.b64decode(output.data['image/png'])
                            output_dict[fn]["image"] = Image.open(BytesIO(image_data)).convert('RGB')
    return output_dict, {}

# --- Text Similarity ---
def similarity(a, b):
    norm_a = "\n".join([line.strip() for line in a.strip().splitlines()])
    norm_b = "\n".join([line.strip() for line in b.strip().splitlines()])
    return SequenceMatcher(None, norm_a, norm_b).ratio()

# --- Compare Outputs ---
def compare_outputs(problem_outputs, solution_outputs, task_weightage, threshold=0.95):
    scores = []
    for function_name in task_weightage:
        prob = problem_outputs.get(function_name, {})
        sol = solution_outputs.get(function_name, {})
        prob_out = prob.get("text", "").strip()
        sol_out = sol.get("text", "").strip()

        if prob_out not in ("", "__NO_OUTPUT__") and sol_out not in ("", "__NO_OUTPUT__"):
            sim = similarity(prob_out, sol_out)
            if sim >= threshold:
                score = task_weightage[function_name]
                remark = "Success"
            else:
                score = 0
                remark = "Mismatch"
        else:
            score = 0
            remark = "Missing output"

        scores.append({
            "method_name": function_name,
            "score_gained": score,
            "remarks": remark,
            "problem_output": prob_out,
            "solution_output": sol_out
        })
    return pd.DataFrame(scores)

# --- Full Evaluation ---
def evaluate_notebooks(problem_path, solution_path, task_weightage):
    problem_outputs, _ = extract_function_outputs_including_calls(problem_path, task_weightage.keys())
    solution_outputs, _ = extract_function_outputs_including_calls(solution_path, task_weightage.keys())

    output_scores = compare_outputs(problem_outputs, solution_outputs, task_weightage)
    task_df = pd.DataFrame(list(task_weightage.items()), columns=["method_name", "max_score"])
    merged_df = output_scores.merge(task_df, on="method_name", how="inner")

    # Metadata
    merged_df["UserEmail"] = user_email
    merged_df["attempt_id"] = attempt_id
    merged_df["timestamp"] = datetime.now(ZoneInfo("Asia/Kolkata"))
    merged_df["project"] = project

    # Special case: Image similarity for outlier_treatment
    prob_img = problem_outputs.get("outlier_treatment", {}).get("image")
    sol_img = solution_outputs.get("outlier_treatment", {}).get("image")
    if prob_img and sol_img:
        size = (400, 400)
        p_img = prob_img.resize(size).convert("L")
        s_img = sol_img.resize(size).convert("L")
        sim_score, _ = ssim(np.array(p_img), np.array(s_img), full=True)
        if sim_score >= 0.95:
            merged_df.loc[merged_df["method_name"] == "outlier_treatment", "score_gained"] = task_weightage["outlier_treatment"]
            merged_df.loc[merged_df["method_name"] == "outlier_treatment", "remarks"] = "Success"

    return merged_df[["UserEmail", "attempt_id", "method_name", "score_gained",
                      "max_score", "timestamp", "remarks", "project"]]

score_df = evaluate_notebooks(problem_notebook_path, solution_notebook_path, task_weightage)
score_df


 
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

