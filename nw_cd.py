#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install jellyfish


# In[4]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import jellyfish  # Use jellyfish for Jaro-Winkler similarity

# Dummy data
db1 = pd.DataFrame({
    'First_name': ['John', 'Alice'],
    'Second_name': ['A.', 'B.'],
    'Third_name': ['Doe', 'Smith'],
    'Address': ['123 Main St', '456 Elm St'],
    'Email': ['john@example.com', 'alice@example.com'],
    'City': ['NY', 'LA'],
    'Zip_code': [10001, 90001],
    'State_code': ['NY', 'CA']
})

db2 = pd.DataFrame({
    'First_name': ['Johnathan', 'Alice'],
    'Second_name': ['A.', 'B.'],
    'Third_name': ['Doe', 'Smith'],
    'Address': ['123 Main St', '456 Elm St'],
    'Email': ['johnathan@example.com', 'alice.smith@example.com'],
    'City': ['NY', 'LA'],
    'Zip_code': [10001, 90001],
    'State_code': ['NY', 'CA']
})

resolution = pd.DataFrame({'Resolution': ['Profile_matched', 'Profile_notmatched']})


# In[5]:


# Calculate Jaro-Winkler similarity for fields
def calculate_similarity(row1, row2):
    return {
        'First_name_sim': jellyfish.jaro_winkler(row1['First_name'], row2['First_name']),
        'Second_name_sim': jellyfish.jaro_winkler(row1['Second_name'], row2['Second_name']),
        'Third_name_sim': jellyfish.jaro_winkler(row1['Third_name'], row2['Third_name']),
        'Address_sim': jellyfish.jaro_winkler(row1['Address'], row2['Address']),
        'Email_sim': jellyfish.jaro_winkler(row1['Email'], row2['Email']),
        'City_sim': jellyfish.jaro_winkler(row1['City'], row2['City']),
        'Zip_code_sim': jellyfish.jaro_winkler(str(row1['Zip_code']), str(row2['Zip_code'])),
        'State_code_sim': jellyfish.jaro_winkler(row1['State_code'], row2['State_code']),
    }

similarity_data = [calculate_similarity(db1.iloc[i], db2.iloc[i]) for i in range(len(db1))]
similarity_df = pd.DataFrame(similarity_data)


# In[6]:


similarity_df


# In[7]:


# Combine similarity scores with the resolution
df = pd.concat([similarity_df, resolution], axis=1)


# In[8]:


df


# In[9]:


# Convert Resolution to a numeric label
df['Resolution'] = df['Resolution'].map({'Profile_matched': 1, 'Profile_notmatched': 0})


# In[10]:


df


# In[11]:


# Train-Test Split
X = df.drop(columns='Resolution')
y = df['Resolution']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


# Train Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)


# In[16]:


X_test


# In[13]:


# Predict with probabilities (confidence score)
y_pred_proba = rf.predict_proba(X_test)
y_pred_proba


# In[17]:


# Get feature importance (field-wise probability scores)
feature_importances = rf.feature_importances_


# In[18]:


feature_importances


# In[19]:


# Print results
print("Confidence Scores (Probabilities):", y_pred_proba)
print("Field-wise Probabilities (Feature Importances):", feature_importances)


# In[31]:


# New entity to be matched from tdb1
new_entity = {
    'First_name': 'Jonathan',
    'Second_name': 'A.',
    'Third_name': 'Doe',
    'Address': '789 Pine St',
    'Email': 'jonathan.doe@example.com',
    'City': 'NY',
    'Zip_code': 10002,
    'State_code': 'NY'
}

# tdb2 entity to compare with
tdb2_entity = {
    'First_name': 'Johnathan',
    'Second_name': 'A.',
    'Third_name': 'Doe',
    'Address': '123 Main St',
    'Email': 'johnathan@example.com',
    'City': 'NY',
    'Zip_code': 10001,
    'State_code': 'NY'
}

tdb2 = pd.DataFrame([tdb2_entity])

# Calculate Jaro-Winkler similarity for the new entity and the entity from tdb2
new_similarity_scores = {
    'First_name_sim': jellyfish.jaro_winkler(new_entity['First_name'], tdb2_entity['First_name']),
    'Second_name_sim': jellyfish.jaro_winkler(new_entity['Second_name'], tdb2_entity['Second_name']),
    'Third_name_sim': jellyfish.jaro_winkler(new_entity['Third_name'], tdb2_entity['Third_name']),
    'Address_sim': jellyfish.jaro_winkler(new_entity['Address'], tdb2_entity['Address']),
    'Email_sim': jellyfish.jaro_winkler(new_entity['Email'], tdb2_entity['Email']),
    'City_sim': jellyfish.jaro_winkler(new_entity['City'], tdb2_entity['City']),
    'Zip_code_sim': jellyfish.jaro_winkler(str(new_entity['Zip_code']), str(tdb2_entity['Zip_code'])),
    'State_code_sim': jellyfish.jaro_winkler(new_entity['State_code'], tdb2_entity['State_code']),
}

# Convert the similarity scores to a DataFrame (similar to how it was during training)
new_similarity_df = pd.DataFrame([new_similarity_scores])




# In[32]:


new_similarity_df


# In[33]:


# Use the trained Random Forest model to predict the match for the new entity
new_prediction_proba = rf.predict_proba(new_similarity_df)


# In[34]:


# Get confidence score for match or not match
confidence_score = new_prediction_proba[0]

# Print the confidence score
print("Confidence Score for the new entity match:", confidence_score)


# In[ ]:





# In[ ]:





# In[ ]:





# In[35]:


# After training the Random Forest model
importances = rf.feature_importances_

# Assign feature names to their importance values
feature_importance_df = pd.DataFrame({
    'Feature': new_similarity_df.columns,
    'Importance': importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)


# In[25]:


tdb2.iterrows


# In[ ]:




