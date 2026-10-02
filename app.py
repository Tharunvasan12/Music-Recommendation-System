import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Recommendation System")
st.write("Find songs similar to a song you like using Machine Learning.")

@st.cache_data
def load_data():
    return pd.read_csv("spotify-tracks-dataset-detailed.csv")

df = load_data()

features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

music_data = df.dropna(subset=features).copy()

scaler = StandardScaler()

feature_matrix = scaler.fit_transform(
    music_data[features]
)

model = NearestNeighbors(
    n_neighbors=6,
    metric="cosine",
    algorithm="brute"
)

model.fit(feature_matrix)

song_list = (
    music_data["track_name"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)

selected_song = st.selectbox(
    "🎧 Select a song",
    song_list
)

def recommend_songs(song_name):

    song_indices = music_data[
        music_data["track_name"] == song_name
    ].index

    if len(song_indices) == 0:
        return pd.DataFrame()

    selected_index = song_indices[0]

    position = music_data.index.get_loc(
        selected_index
    )

    song_features = feature_matrix[position].reshape(1, -1)

    distances, indices = model.kneighbors(
        song_features,
        n_neighbors=6
    )

    recommendations = []

    for distance, index in zip(
        distances[0][1:],
        indices[0][1:]
    ):

        song = music_data.iloc[index]

        similarity = (1 - distance) * 100

        recommendations.append({
            "Song": song["track_name"],
            "Artist": song["artists"],
            "Genre": song["track_genre"],
            "Similarity": round(similarity, 2)
        })

    return pd.DataFrame(recommendations)

if st.button("🎶 Recommend Songs"):

    recommendations = recommend_songs(
        selected_song
    )

    st.subheader(
        f"Recommended Songs Similar to '{selected_song}'"
    )

    if not recommendations.empty:

        for i, row in recommendations.iterrows():

            st.write(
                f"### {i + 1}. {row['Song']}"
            )

            st.write(
                f"🎤 Artist: {row['Artist']}"
            )

            st.write(
                f"🎼 Genre: {row['Genre']}"
            )

            st.write(
                f"📊 Similarity: {row['Similarity']}%"
            )

            st.divider()

    else:
        st.error("No recommendations found.")