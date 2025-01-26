import pandas as pd

df = pd.read_csv("/Users/aldinfazlic/Desktop/projects/fragrance/fra_cleaned.csv", sep=';', encoding="ISO-8859-1", on_bad_lines='skip')

#removing unneccessary columns
df.drop(['Perfumer1','Perfumer2'], axis=1, inplace = True)


#encoding features so ML model can read them 
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output = False)
brand_encoded = encoder.fit_transform(df[['Brand']]) #encode brand

#convert to a df
brand_df = pd.DataFrame(brand_encoded, columns=encoder.get_feature_names_out(['Brand']))

#concatenate with main df
df = pd.concat([df, brand_df], axis=1)
#df.drop('Brand', axis=1, inplace = True)



#TF-IDF encoding for top, middle, base notes
from sklearn.feature_extraction.text import TfidfVectorizer
notes = df[['Top', 'Middle', 'Base']].fillna('')

#combine all notes into one column
notes_combined = notes['Top'] + ' ' + notes['Middle'] + ' ' + notes['Base']

vectorizer = TfidfVectorizer(max_features=100)
notes_tfidf = vectorizer.fit_transform(notes_combined)

notes_df = pd.DataFrame(notes_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

df = pd.concat([df, notes_df], axis = 1)
#print(df.head(5))

#using cosine similarity to build the recommendation system
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(notes_df)
#print(similarities)



def recommend(brand, name, num_recs = 5):

    if name not in df['Perfume'].values:
        return f"Perfume '{name}' not found in the dataset."

    if brand not in df['Brand'].values:
        return f"Perfume '{Brand}' not found in the dataset."

    idx = df[df['Perfume'] == name].index[0]
    
    # Get the similarity scores for the perfume
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort perfumes by similarity scores (excluding the perfume itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recs + 1]  # Top N recommendations
    
    # Get the recommended perfume indices
    perfume_indices = [i[0] for i in sim_scores]
    
    # Return the names of the recommended perfumes
    recommended_perfumes = df.iloc[perfume_indices][['Perfume', 'Brand', 'Rating Value', 'url']]
    return recommended_perfumes

recommendations = recommend("nautica", "voyage", 5)
print(recommendations)