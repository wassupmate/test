# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:27:35 2024

@author: anhtu
"""
import streamlit as st
import pickle
import numpy as np
import pandas as pd


class CB(object):
    def __init__(self, movies_df, n_clusters=8):
        # Set the index to 'Film_title' and ensure it's not duplicated
        self.movies = movies_df.set_index('Film_title')
        self.n_clusters = n_clusters
        self.cosine_sim = None
        self.cluster_labels = None
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)


    def build_model(self):
        # Calculate the cosine similarity matrix, excluding non-numeric columns
        features = self.movies
        self.cosine_sim = cosine_similarity(features)

        # Fit KMeans and assign clusters
        self.cluster_labels = self.kmeans.fit_predict(features)
        self.movies['Cluster'] = self.cluster_labels

        
    def refresh(self):
        self.build_model()

    def fit(self):
        self.refresh()

    def genre_recommendations(self, titles, weights=None, top_x=5):
        # Ensure all titles exist in the DataFrame
        missing_titles = [title for title in titles if title not in self.movies.index]
        if (missing_titles):
            print(f"Titles not found in the dataset: {missing_titles}")
            return [], []

        # Get indices for the titles
        indices = [self.movies.index.get_loc(title) for title in titles]

        # Set default weights if not provided
        if weights is None:
            weights = [1.0 / len(titles)] * len(titles)
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights

        # Calculate the weighted similarity score for the given titles
        
        sim_scores = sum(weight * self.cosine_sim[idx] for idx, weight in zip(indices, weights))
        cluster_scores = sum(weight * self.movies.iloc[idx] for idx, weight in zip(indices,weights))
        
        cluster_scores = cluster_scores.drop("Cluster")
        cluster_scores = cluster_scores.values.reshape(1, -1)
        cluster_categorize = self.kmeans.predict(cluster_scores)
        
        print(cluster_categorize)
        with pd.option_context('display.max_rows', 11, 'display.max_columns', 11):
            cluster = self.movies[self.movies['Cluster']==cluster_categorize[0]]
            print(cluster.mean().sort_values(ascending=False)) 
            print(len(cluster))
        # Create similarity scores with corresponding indices
        sim_scores = list(enumerate(sim_scores))

        # Debug: Print sim_scores before sorting
        #print(f"Similarity scores before sorting: {sim_scores}")

        # Sort similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Debug: Print sim_scores after sorting
        #print(f"Similarity scores after sorting: {sim_scores}")

        # Get the cluster of the first title
        title_clusters = [self.movies.iloc[idx]['Cluster'] for idx in indices]

        # Filter sim_scores to only include movies in the same cluster(s)
        sim_scores = [score for score in sim_scores if self.movies.iloc[score[0]]['Cluster'] in title_clusters]
        cluster = [self.movies.iloc[indice]['Cluster'] for indice in indices]

        # Get top_x results excluding the movies themselves
        sim_scores = [score for score in sim_scores if score[0] not in indices][:top_x]
        
        movie_indices = [i[0] for i in sim_scores]
        movie_cluster = [self.movies.iloc[idx]['Cluster'] for idx in movie_indices]

        # Debug: Print movie_indices
        #print(f"Movie indices: {movie_indices}")
        
        return sim_scores, self.movies.index[movie_indices],cluster, movie_cluster


    
model = pickle.load(open('new.pkl','rb'))

# titles = ["CODA","Avatar"]
# weights = [0.5,0.5]
# top_x = 100
# scores, recommendations,groups,clusters = model.genre_recommendations(titles, weights, top_x)

# # Print recommendations
# print(f"Top {top_x} recommendations for {titles} in {[group for group in groups]}:")
# for score, recommendation,cluster in zip(scores, recommendations,clusters):
#     print(f"{recommendation} (Score: {score[1]}) (cluster:{cluster})")
    
def main():
    st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Forest Fire Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    number_inputs = st.number_input('number of movie', step=1, min_value=1)
    st.write('number of movie ', number_inputs)
    input_values = [st.text_input(f'Movie {i+1}','Movie Name', key=f"text_input_{i}")
          for i in range(number_inputs)]
    weights1 = [st.number_input(f'Weight {i+1}')
          for i in range(number_inputs)]


    # if st.button("Add to df", key="button_update"):
    # # Update dataframe state
    #     st.session_state.df = pd.concat(
    #         [st.session_state.df, pd.DataFrame({'nr': input_values})],
    #         ignore_index=True)
    #     st.text("Updated dataframe")
    # st.dataframe(st.session_state.df)
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Your forest is safe</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
       </div>
    """

    if st.button("Predict"):
        print(input_values)
        titles = input_values
        # titles.append(oxygen)
        # titles.append(humidity)
        # titles.append(temperature)
        weights = weights1
        top_x = 10
        scores, recommendations,groups,clusters = model.genre_recommendations(titles, weights, top_x)
        for score, recommendation,cluster in zip(scores, recommendations,clusters):
            st.success(f"{recommendation} (Score: {score[1]}) (cluster:{cluster})")
        

if __name__=='__main__':
    main()
    


