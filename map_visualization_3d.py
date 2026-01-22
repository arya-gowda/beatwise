import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# 1. Load Data
df = pd.read_csv('Liked_Songs.csv')

# 2. Select features for UMAP (NO Popularity)
umap_features = [
    'Danceability', 'Energy', 'Loudness', 'Speechiness', 
    'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 
    'Tempo', 'Mode'
]

# Keep popularity for visualization only
df_clean = df.dropna(subset=umap_features + ['Key', 'Popularity']).copy()

# One-hot encode the 'Key' column
key_dummies = pd.get_dummies(df_clean['Key'].astype(int), prefix='Key', dtype=float)

# Combine ONLY the features for UMAP (no Popularity)
final_features = pd.concat([df_clean[umap_features].reset_index(drop=True), 
                            key_dummies.reset_index(drop=True)], axis=1)

# 3. Standardize
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean[umap_features])

# 4. Run UMAP
reducer = UMAP(
    n_components=3, 
    n_neighbors=8,
    min_dist=0.8,         
    spread=2.5,           
    metric='cosine',      
    random_state=42,
    negative_sample_rate=15
)
embedding = reducer.fit_transform(scaled_data)

df_clean['x'] = embedding[:, 0]
df_clean['y'] = embedding[:, 1]
df_clean['z'] = embedding[:, 2]

# # DEBUG: Find specific songs
# ransom = df_clean[df_clean['Track Name'].str.contains('Ransom', case=False, na=False)]
# beabadoobee = df_clean[df_clean['Artist Name(s)'].str.contains('beabadoobee', case=False, na=False)]

# print("=== RANSOM BY LIL TECCA ===")
# if not ransom.empty:
#     print(ransom[['Track Name', 'Artist Name(s)', 'Popularity'] + umap_features].head())
#     print(f"UMAP coords: ({ransom['x'].iloc[0]:.2f}, {ransom['y'].iloc[0]:.2f}, {ransom['z'].iloc[0]:.2f})")

# print("\n=== BEABADOOBEE SONGS ===")
# if not beabadoobee.empty:
#     print(beabadoobee[['Track Name', 'Artist Name(s)', 'Popularity'] + umap_features].head())
#     for idx, row in beabadoobee.iterrows():
#         print(f"{row['Track Name']}: ({row['x']:.2f}, {row['y']:.2f}, {row['z']:.2f})")

# # Calculate distance between them if both exist
# if not ransom.empty and not beabadoobee.empty:
#     for idx, bb_song in beabadoobee.iterrows():
#         dist = np.sqrt(
#             (ransom['x'].iloc[0] - bb_song['x'])**2 + 
#             (ransom['y'].iloc[0] - bb_song['y'])**2 + 
#             (ransom['z'].iloc[0] - bb_song['z'])**2
#         )
#         print(f"\nDistance from Ransom to {bb_song['Track Name']}: {dist:.2f}")

# 5. Visualization - COLOR by Popularity (but it wasn't used in UMAP!)
fig = px.scatter_3d(
    df_clean, x='x', y='y', z='z',
    color='Popularity',  # Show popularity for visualization only
    color_continuous_scale='Viridis',
    hover_data=['Track Name', 'Artist Name(s)', 'Energy', 'Acousticness', 'Speechiness', 'Valence'],
    title="3D Music Galaxy - UMAP (Audio Features Only, Colored by Popularity)",
    labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
    opacity=0.7
)

fig.update_traces(marker=dict(size=2.5, line=dict(width=0)))
fig.update_layout(template='plotly_dark', height=800)
fig.show()

print(f"\nTotal songs visualized: {len(df_clean)}")
print(f"\nFeatures used for UMAP: {umap_features + ['Key (one-hot encoded)']}")
print(f"Popularity: Used for COLORING only, not for clustering")