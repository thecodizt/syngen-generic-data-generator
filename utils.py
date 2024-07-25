import random
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import networkx as nx
from sklearn.metrics import mean_squared_error
import plotly.express as px

def generate_adjancency_matrix_with_none(num_nodes, density = 0.5, allow_self_edge=False, edge_weight_lower=1, edge_weight_upper=1):
    adj_matrix = [[None for i in range(num_nodes)] for j in range(num_nodes)]
    
    current_density = 0
    
    while current_density < density:
        while True:
            i = random.randint(0,num_nodes-1)
            j = random.randint(0,num_nodes-1)
            
            if not adj_matrix[i][j]:
                adj_matrix[i][j] = edge_weight_lower if (edge_weight_lower == edge_weight_upper) else random.uniform(edge_weight_lower, edge_weight_upper)
                current_density += 1/(num_nodes*num_nodes)
    
                if not allow_self_edge:
                    if i == j:
                        adj_matrix[i][i] = None
                        current_density -= 1/(num_nodes*num_nodes)
                        
                break
                
        
    return adj_matrix

def generate_control_points(num_points = 10):
    control_points = []
    
    while len(control_points) < num_points:
        control_points.append(random.random())
        
    return control_points

def generate_spline(control_points, n_points=100, noise=0):
    # Create x values for control points
    x = np.linspace(0, 1, len(control_points))

    # Create cubic spline
    cs = CubicSpline(x, control_points)

    # Generate N evenly spaced x values between 0 and 1
    x_new = np.linspace(0, 1, n_points)

    # Compute y values for these x values
    y_new = cs(x_new)
    
    # Introduce noise
    if noise != 0:
        noise_amount = np.random.normal(0, noise, size=y_new.shape)
        y_new = y_new + noise_amount

    return y_new

def generate_node_data(num_properties=1, num_records=100, num_control_points=10, noise=0, features=None, node_lower_range=0, node_upper_range=1):
    node_data = dict()
    
    if features is None:
        features = list(range(num_properties))
            
    for i in range(num_properties):
        node_property = generate_spline(generate_control_points(num_control_points), num_records, noise)

        node_property = node_property * (node_upper_range - node_lower_range) + node_lower_range
        
        node_data[features[i]] = node_property
    
    # convert node data to dataframe
    generated_node_data = pd.DataFrame(node_data)
        
    return generated_node_data
        
def flatten_dataframe(df):
    df_flat = df.reset_index().melt(id_vars='index', var_name='column', value_name='value')
    return df_flat

def unflatten_dataframe(df_flat):
    df = df_flat.pivot(index='index', columns='column', values='value')
    df.reset_index(drop=True, inplace=True)
    df.columns.name = None
    return df

def merge_melted_dfs(dfs):
    # Add 'entity' column to each DataFrame and concatenate them
    for i, df in enumerate(dfs):
        df['entity'] = i
    
    df_concat = pd.concat(dfs, ignore_index=True)
    df_concat.columns = ['timestamp', 'feature', 'value', 'node']
    return df_concat

def array_to_dataframe(array_2d):
    df = pd.DataFrame(array_2d)
    return df

def generate_n_node_flat_data_in_range(num_nodes, num_records, lower_num_properties, upper_num_properties, num_control_points, noise, features=None, node_lower_range=0, node_upper_range=1):
    flat_dfs = []
    
    while (len(flat_dfs)) < num_nodes:
        num_properties = random.randint(lower_num_properties, upper_num_properties)
        generated_node_data = generate_node_data(num_properties, num_records, num_control_points, noise, features, node_lower_range=node_lower_range, node_upper_range=node_upper_range)
        flat_df = flatten_dataframe(generated_node_data)
        flat_dfs.append(flat_df)
        
    return merge_melted_dfs(flat_dfs)

def get_node_data_from_merged(merged_data, node_index):
    filtered = merged_data[merged_data["entity"] == node_index]
    filtered.drop(["entity"], axis=1, inplace=True)
    
    unflattened = unflatten_dataframe(filtered)
    
    return unflattened

def adjacency_matrices_to_dataframe(adjacency_matrices, features):
    rows = []
    
    if features is None:
        features = list(range(len(adjacency_matrices)))
    
    # Iterate through each adjacency matrix with its timestamp
    for feature_id, time_matrix in enumerate(adjacency_matrices):
        
        for time, feature_matrix in enumerate(time_matrix):
            
            for i in range(len(feature_matrix)):
                for j in range(len(feature_matrix[i])):
                    rows.append([time, i, j, features[feature_id], feature_matrix[i][j]])

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=['timestamp', 'source', 'target', 'feature', 'value'])
    
    return df

def visualize_node_data(node_data):
    df = node_data
    
    # Get the unique nodes
    nodes = df['node'].unique()

    selected_node = 0
    
    if len(nodes) > 1:
        # Add a slider to select the node
        selected_node = st.selectbox('Select a node', options=nodes)

    # Filter the dataframe based on the selected node
    filtered_df = df[df['node'] == selected_node]

    # Create a line plot for each feature
    fig = px.line(filtered_df, x='timestamp', y='value', color='feature', title=f'Features over Time for Node {selected_node}', width=650, height=500)

    st.plotly_chart(fig)

def visualize_edge_data(edge_data):
    df = edge_data
    
    features = df['feature'].unique()
    timestamps = df['timestamp'].unique()
    
    selected_feature = features[0]
    selected_timestamp = timestamps[0]
    
    if len(features) > 1:
        selected_feature = st.selectbox('Select a feature', options=features)

    if len(timestamps) > 1:
        selected_timestamp = st.select_slider('Select a timestamp', options=timestamps)

    # Filter the dataframe based on selected feature and timestamp
    filtered_df = df[(df['feature'] == selected_feature) & (df['timestamp'] == selected_timestamp)]

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges with value as weight if the value is not None
    for _, row in filtered_df.iterrows():
        if pd.notnull(row['value']):
            G.add_edge(row['source'], row['target'], weight=row['value'], label=row['value'])

    # Draw the graph using NetworkX and Matplotlib
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='->', arrowsize=20)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Title and axes
    plt.title(f'Directed Graph for Feature: {selected_feature} at Timestamp: {selected_timestamp}')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.axis('off')

    # Display the plot
    st.pyplot(plt)