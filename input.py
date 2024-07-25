import streamlit as st

def node_base():
    st.subheader("Node Input")
    
    num_nodes = st.number_input(label="Number of Nodes in Graph", min_value=1, step=1)
        
    is_custom_node_range = st.checkbox("Custom Node Feature Range (defaults to 0 to 1)")
    
    if is_custom_node_range:
        node_upper_range = st.number_input("Upper limit for values")
        node_lower_range = st.number_input("Lower limit for values", max_value=node_upper_range)
    else:
        node_upper_range = 1
        node_lower_range = 0
        
    is_custom_node_feature_names = st.checkbox(label="Custom Node Feature Names")
    
    if is_custom_node_feature_names:
        node_feature_names = st.text_input(label="Enter Node Feature Names (comma separated)")
        node_feature_names = node_feature_names.split(",")
    else:
        node_feature_names = None
        
    num_control_points = st.number_input(label="Number of Control Points in Generation", min_value=2, step=1)
    
    noise = st.number_input(label="Maximum Noise in Values", min_value=0.0, max_value=1.0, step=0.05)
    
    return num_nodes, node_upper_range, node_lower_range, node_feature_names, num_control_points, noise

def edge_base():
    st.subheader("Edge Input")
    
    num_edge_features = st.number_input(label="Number of properties for each edge", min_value=1, step=1)
    
    is_custom_edge_feature_names = st.checkbox(label="Custom Edge Feature Names")
    
    if is_custom_edge_feature_names:
        edge_feature_names = st.text_input("Enter Edge Feature Names (comma seperated)")
        edge_feature_names = edge_feature_names.split(',')
    else:
        edge_feature_names = None
        
    edge_density = st.number_input(label="Edge Density in Adjacency Matrix", min_value=0.0, max_value=1.0, step=0.05)
    
    is_custom_weights = st.checkbox("Custom Edge Weights")
    
    if is_custom_weights:
        edge_weight_upper = st.number_input("Lower range for edge upper")
        edge_weight_lower = st.number_input("Lower range for edge weight", max_value=edge_weight_upper)
    else:
        edge_weight_upper = 1
        edge_weight_lower = 1
    
    return num_edge_features, edge_density, edge_feature_names, edge_weight_lower, edge_weight_upper

def node_homogeneous():
    num_nodes, node_upper_range, node_lower_range, node_feature_names, num_control_points, noise = node_base()

    num_prop = st.number_input(label="Number of properties for each node", min_value=1, step=1)
        
    return num_nodes, node_lower_range, node_upper_range, num_prop, node_feature_names, num_control_points, noise

def node_heterogeneous():
    
    num_nodes, node_upper_range, node_lower_range, node_feature_names, num_control_points, noise = node_base()
    
    upper_num_prop = st.number_input(label="Upper range for number of properties for each node", min_value=1, step=1)
    lower_num_prop = st.number_input(label="Lower range for number of properties for each node", min_value=1, max_value=upper_num_prop, step=1)
        
    return num_nodes, node_lower_range, node_upper_range, lower_num_prop, upper_num_prop, node_feature_names, num_control_points, noise

def edge_static():
    num_edge_features, edge_density, edge_feature_names, edge_weight_lower, edge_weight_upper = edge_base()
    
    return num_edge_features, edge_density, edge_feature_names, edge_weight_lower, edge_weight_upper

def edge_dynamic():
    edge_determination_options = [
            "random",
            "criticality based",
            "community based",
        ]
    
    num_edge_features, edge_density, edge_feature_names, edge_weight_lower, edge_weight_upper = edge_base()
        
    new_edge_likelihood = st.number_input(label="Probabilty of new edge creation", min_value=0.0, max_value=1.0, step=0.05)
    delete_edge_likelihood = st.number_input(label="Probability of edge deletion", min_value=0.0, max_value=1.0, step=0.05)
    
    edge_determination = st.selectbox(label="Algorithm for determining edge changes", options=edge_determination_options)
    
    return num_edge_features, edge_density, edge_feature_names, new_edge_likelihood, delete_edge_likelihood, edge_determination, edge_weight_lower, edge_weight_upper