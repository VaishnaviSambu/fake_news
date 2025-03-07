#!/usr/bin/env python
# coding: utf-8

# In[44]:


pip install numpy pandas scikit-learn


# In[45]:


get_ipython().system('pip install wordcloud')


# In[46]:


from wordcloud import WordCloud
print("WordCloud module imported successfully!")


# In[52]:


# Display Examples of Correctly and Misclassified News
test_data = x_test.reset_index(drop=True)
correctly_classified = test_data[(y_test == y_pred).values]
misclassified = test_data[(y_test != y_pred).values]

# Correctly classified examples
print("\nExamples of Correctly Classified News:")
for i in range(5):
    print(f"Text: {correctly_classified.iloc[i][:100]}...")
    print(f"Label: {y_test.iloc[correctly_classified.index[i]]}\n")

# Misclassified examples
print("\nExamples of Misclassified News:")
for i in range(min(5, len(misclassified))):
    print(f"Text: {misclassified.iloc[i][:100]}...")
    print(f"True Label: {y_test.iloc[misclassified.index[i]]}")
    print(f"Predicted Label: {y_pred[misclassified.index[i]]}\n")


# In[53]:


import matplotlib.pyplot as plt
import networkx as nx

# Define the classification flow
graph = nx.DiGraph()
graph.add_edges_from([
    ("Start", "Text Preprocessing"), 
    ("Text Preprocessing", "TF-IDF Vectorization"),
    ("TF-IDF Vectorization", "Model Training"),
    ("Model Training", "Prediction"),
    ("Prediction", "Evaluation")
])

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define the position and draw the graph
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_color='skyblue', 
        node_size=3000, font_size=10, font_weight="bold", ax=ax)

# Add a title
plt.title("Classification Process Flow", fontsize=16)
plt.show()


# In[54]:


# Pie Chart for Fake vs Real News
plt.figure(figsize=(6, 6))
df['label'].value_counts().plot.pie(autopct='%1.1f%%', colors=['red', 'green'], labels=['FAKE', 'REAL'], startangle=90)
plt.title('Fake vs Real News Distribution')
plt.ylabel('')
plt.show()


# In[55]:


from sklearn.metrics import classification_report

# Generate classification report
class_report = classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'], output_dict=True)
print("\nClassification Report:")
print(pd.DataFrame(class_report).transpose())


# In[56]:


import seaborn as sns

# Confusion matrix
confusion = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

# Plot Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[57]:


from wordcloud import WordCloud

# Word Cloud for FAKE News
fake_news = df[df['label'] == 'FAKE']['text'].str.cat(sep=' ')
fake_wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(fake_news)

plt.figure(figsize=(10, 6))
plt.imshow(fake_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for FAKE News')
plt.axis('off')
plt.show()

# Word Cloud for REAL News
real_news = df[df['label'] == 'REAL']['text'].str.cat(sep=' ')
real_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(real_news)

plt.figure(figsize=(10, 6))
plt.imshow(real_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for REAL News')
plt.axis('off')
plt.show()


# In[58]:


# Bar Chart for Precision, Recall, and F1-Score
metrics = ['precision', 'recall', 'f1-score']
fake_metrics = [class_report['FAKE'][metric] for metric in metrics]
real_metrics = [class_report['REAL'][metric] for metric in metrics]

x = np.arange(len(metrics))  # Label locations
width = 0.35  # Bar width

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, fake_metrics, width, label='FAKE')
rects2 = ax.bar(x + width/2, real_metrics, width, label='REAL')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, and F1-Score by Class')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.show()


# In[59]:


# Extract top influential features
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
sorted_coeff_indices = np.argsort(pac.coef_[0])
top_features_fake = feature_names[sorted_coeff_indices[:10]]
top_features_real = feature_names[sorted_coeff_indices[-10:]]

print("\nTop Features for FAKE News:")
print(top_features_fake)

print("\nTop Features for REAL News:")
print(top_features_real)


# In[ ]:




