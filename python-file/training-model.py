import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

perplexities = [5, 10, 30, 50, 100]

tsne_results = {}

for perplexity in perplexities:
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    X_reduced = tsne.fit_transform(X.toarray())
    tsne_results[perplexity] = X_reduced

clustering_algorithms = {
    'KMeans': KMeans(n_clusters=5, random_state=0),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'AgglomerativeClustering': AgglomerativeClustering(n_clusters=5)
}

results = []

for perplexity, X_reduced in tsne_results.items():
    for algo_name, algo in clustering_algorithms.items():
        if algo_name == 'DBSCAN':
            # DBSCAN does not require fit_predict
            labels = algo.fit_predict(X_reduced)
        else:
            labels = algo.fit_predict(X_reduced)
        
        # Calculate internal metrics
        sil_score = silhouette_score(X_reduced, labels)
        db_index = davies_bouldin_score(X_reduced, labels)
        ch_score = calinski_harabasz_score(X_reduced, labels)
        
        results.append({
            'Perplexity': perplexity,
            'Algorithm': algo_name,
            'Silhouette Score': sil_score,
            'Davies-Bouldin Index': db_index,
            'Calinski-Harabasz Index': ch_score
        })

#file path can use anything, try using little data for getting the best silhuette score
file_path = 'NLP-Emotion-Dataset.csv'
df = pd.read_csv(file_path, delimiter=',', on_bad_lines='skip')

# Automatically identify the text column
text_column = None
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column contains text data
        text_column = column
        break

if text_column is None:
    raise ValueError("No text column found in the DataFrame. Please check your data.")

#process all the dataset before being use as training dataset
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    filtered_words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered_words)

# Apply preprocessing to the text column
df['processed_text'] = df[text_column].fillna('').apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])

#use perplexity algorithm
perplexities = [5, 10, 30, 50, 100]
tsne_results = {}

for perplexity in perplexities:
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    X_reduced = tsne.fit_transform(X.toarray())
    tsne_results[perplexity] = X_reduced

#apply the clustering algorithms that wanted to be test
clustering_algorithms = {
    'KMeans': KMeans(n_clusters=5, random_state=0),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'AgglomerativeClustering': AgglomerativeClustering(n_clusters=5)
}

#start training and find the best model or algorithms 
results = []
for perplexity, X_reduced in tsne_results.items():
    for algo_name, algo in clustering_algorithms.items():
        if algo_name == 'DBSCAN':
            # DBSCAN does not require fit_predict
            labels = algo.fit_predict(X_reduced)
        else:
            labels = algo.fit_predict(X_reduced)
        
        # Calculate internal metrics
        sil_score = silhouette_score(X_reduced, labels)
        db_index = davies_bouldin_score(X_reduced, labels)
        ch_score = calinski_harabasz_score(X_reduced, labels)
        
        results.append({
            'Perplexity': perplexity,
            'Algorithm': algo_name,
            'Silhouette Score': sil_score,
            'Davies-Bouldin Index': db_index,
            'Calinski-Harabasz Index': ch_score
        })

#output of the result as a silhuette score, DB Index, and CH Index
results_df = pd.DataFrame(results)
print(results_df)

#calculate the score and find the best algorithm
best_result = results_df.loc[results_df['Silhouette Score'].idxmax()]

best_perplexity = best_result['Perplexity']
print(f"Best Perplexity: {best_perplexity}")

best_algorithm_name = best_result['Algorithm']
print(f"Best Algorithm: {best_algorithm_name}")

#Use the best perplexity and algorithm to fit the final model
X_reduced = tsne_results[best_perplexity]
best_algorithm = clustering_algorithms[best_algorithm_name]

if best_algorithm_name == 'DBSCAN':
    final_labels = best_algorithm.fit_predict(X_reduced)
else:
    final_labels = best_algorithm.fit_predict(X_reduced)

# Assign cluster labels to DataFrame
df['cluster'] = final_labels

# Create a dictionary mapping cluster labels to emotions
cluster_emotions = {
    0: 'Joy',
    1: 'Sadness',
    2: 'Anger',
    3: 'Love',
    4: 'Surprise'
}

#Map cluster labels to emotions
df['emotion'] = df['cluster'].map(cluster_emotions)

#shows how many text that the AI consider "that" emotion is suit with
emotion_counts = df['emotion'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(emotion_counts.index, emotion_counts.values)
plt.xlabel('Emotion')
plt.ylabel('Number of Data Points')
plt.title('Distribution of Data Points Across Emotions')
# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.show()

#shows the best cluster using t-SNE visualization
plt.figure(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(len(cluster_emotions)):
    cluster_points = X_reduced[df['cluster'] == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors[i], label=cluster_emotions[i])
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title(f't-SNE Visualization of Clusters with Perplexity {best_perplexity} and {best_algorithm_name}')
plt.legend()
plt.show()

#if you want the output of the data that has been labeled by the AI
#output_file_path = 'csv or text file that you had train'
#df.to_csv(output_file_path, index=False)
#print(f'Results saved to {output_file_path}')