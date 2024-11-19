import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import folium
from folium.features import DivIcon
from collections import defaultdict
import networkx as nx

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, dtype={'TELNUM': str})
    return df.dropna(subset=['Latitude', 'Longitude'])

def get_cluster_centers(df, labels):
    centers = {}
    for label in set(labels) - {-1}:
        mask = labels == label
        centers[label] = (
            df[mask]['Latitude'].mean(),
            df[mask]['Longitude'].mean()
        )
    return centers

def cluster_schools(df, max_distance, min_schools=3):
    coords = df[['Latitude', 'Longitude']].values
    
    # Initial clustering
    clustering = DBSCAN(eps=(max_distance*1.5)/69, min_samples=min_schools, metric='euclidean').fit(coords)
    initial_labels = clustering.labels_
    
    # Get initial cluster centers
    centers = get_cluster_centers(df, initial_labels)
    
    # Reassign points based on distance to cluster centers
    final_labels = np.full_like(initial_labels, -1)
    
    # First pass: assign points to their closest valid cluster
    for idx, (lat, lon) in enumerate(coords):
        min_distance = float('inf')
        best_cluster = -1
        
        for cluster_label, (center_lat, center_lon) in centers.items():
            dist = geodesic((lat, lon), (center_lat, center_lon)).miles
            if dist <= max_distance and dist < min_distance:
                min_distance = dist
                best_cluster = cluster_label
        
        if best_cluster != -1:
            final_labels[idx] = best_cluster
    
    # Remove clusters that don't meet minimum size
    cluster_sizes = defaultdict(int)
    for label in final_labels:
        if label != -1:
            cluster_sizes[label] += 1
    
    # Set labels to -1 for clusters that are too small
    for idx, label in enumerate(final_labels):
        if label != -1 and cluster_sizes[label] < min_schools:
            final_labels[idx] = -1
    
    # Renumber remaining clusters consecutively
    unique_labels = sorted(list(set(final_labels) - {-1}))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    final_labels = np.array([label_map.get(label, -1) for label in final_labels])
    
    return final_labels

color_pairs = {
    0: ('#0077BB', '#ffffff'),  # Strong blue
    1: ('#EE7733', '#000000'),  # Orange
    2: ('#009988', '#ffffff'),  # Teal 
    3: ('#CC3311', '#ffffff'),  # Red
    4: ('#33BBEE', '#000000'),  # Light blue
    5: ('#EE3377', '#ffffff'),  # Magenta
    6: ('#BBBBBB', '#000000'),  # Gray
    7: ('#000000', '#ffffff')   # Black
}

def assign_colors_to_clusters(df, labels):
    cluster_colors = {}
    for label in set(labels) - {-1}:
        cluster_colors[label] = color_pairs[label % 8]
    return cluster_colors



def visualize_clusters(df, distance, min_schools=3):
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    labels = cluster_schools(df, distance, min_schools)
    cluster_colors = assign_colors_to_clusters(df, labels)
    
    for idx, row in df.iterrows():
        label = labels[idx]
        if label == -1:
            continue
            
        color, text_color = cluster_colors[label]
        
        address_parts = [str(row[field]) for field in ['ADDRESS1', 'TOWN', 'PCODE'] if pd.notna(row[field]) and str(row[field]).strip()]
        address_string = ', '.join(address_parts)
        
        admpol = 'independent' if pd.isna(row['ADMPOL_PT']) or row['ADMPOL_PT'].strip() == '' else row['ADMPOL_PT']
        
        popup_html = f"""
        <b>{row['SCHNAME']}</b><br>
        {address_string}<br>
        Phone: {row['TELNUM']}<br>
        Admissions Policy: {admpol}<br>
        Age Range: {row['AGERANGE']}<br>
        Gender: {row['GEND1618']}<br>
        Students (16-18): {row['TPUP1618']}<br>
        Avg best 3 A-levels (TB3PTSE): {row['TB3PTSE']:.2f} (<b>{row['TB3PTSE_Grade']}</b>)<br>
        Avg per A-level (TALLPPE_ALEV_1618): {row['TALLPPE_ALEV_1618']:.2f} (<b>{row['TALLPPE_ALEV_1618_Grade']}</b>)
        """
        
        icon = DivIcon(
            icon_size=(30, 30),
            icon_anchor=(15, 30),
            html=f'''
                <div style="width: 30px; height: 30px;">
                    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 0C7.58 0 4 3.58 4 8c0 5.76 8 16 8 16s8-10.24 8-16c0-4.42-3.58-8-8-8z"
                            fill="{color}"
                            stroke="black"
                            stroke-width="1"/>
                        <text x="12" y="9" font-family="Arial" font-size="8" font-weight="bold" 
                            fill="{text_color}" text-anchor="middle" dy=".3em">{round(row['TB3PTSE'])}</text>
                    </svg>
                </div>
            '''
        )
        
        folium.Marker(
            [row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=icon
        ).add_to(m)

    return m

def main(csv_path, distance=3, min_schools=3):
    df = load_and_prepare_data(csv_path)
    labels = cluster_schools(df, distance, min_schools)
    clusters = defaultdict(list)
    
    for idx, label in enumerate(labels):
        if label != -1:
            clusters[label].append(df.iloc[idx])
    
    print(f"\nClusters within {distance} miles (minimum {min_schools} schools):")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Unclustered schools: {sum(1 for l in labels if l == -1)}")
    
    for cluster_id, schools in clusters.items():
        print(f"\nCluster {cluster_id + 1}:")
        for school in schools:
            print(f"- {school['SCHNAME']} ({school['TOWN']})")
    
    map_obj = visualize_clusters(df, distance, min_schools)
    map_obj.save(f'school_clusters_{distance}mi_min{min_schools}.html')

if __name__ == "__main__":
    main("processed_school_data.csv", 2, 2)