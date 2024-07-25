import streamlit as st

from variants import StaticHomogenous, DynamicHomogenous, StaticHeterogenous, DynamicHeterogenous
from utils import visualize_node_data, visualize_edge_data

def main():
    st.title("LAM - Generic Graph Time Series Data Generator")
    
    graph_types = [
        "Static",
        "Dynamic"
    ]
    
    node_types = [
        "Homogenous",
        "Heterogenous"
    ]
    
    graph_type = st.selectbox("Graph Type", options=graph_types)
    node_type = st.selectbox("Node Type", options=node_types)
    
    st.header("Inputs")
    
    if graph_type == graph_types[0]:
        if node_type == node_types[0]:
            num_nodes, node_lower_range, node_upper_range, num_records, num_prop, node_feature_names, num_edge_features, edge_density, edge_feature_names ,noise, num_control_points, edge_weight_lower, edge_weight_upper = StaticHomogenous.input()
            
            node_data = StaticHomogenous.generate_node_data(
                num_nodes=num_nodes, 
                num_records=num_records, 
                num_prop=num_prop, 
                noise=noise, 
                num_control_points=num_control_points,
                features=node_feature_names,
                node_lower_range=node_lower_range, 
                node_upper_range=node_upper_range,
            )
            
            edge_data = StaticHomogenous.generate_edge_data(
                edge_density=edge_density,
                num_nodes=num_nodes,
                num_edge_features=num_edge_features,
                features=edge_feature_names,
                edge_weight_lower=edge_weight_lower,
                edge_weight_upper=edge_weight_upper,
            )
            
            st.header("Generated Data")
            
            if len(node_data):
                st.subheader("Node Data")
                st.dataframe(node_data)
                
                st.download_button("Download Node Data", data=node_data.to_csv(), file_name='nodes.csv')
                
            if len(edge_data):
                st.subheader("Edge Data")
                st.dataframe(edge_data)

                st.download_button("Download Edge Data", data=edge_data.to_csv(), file_name='edges.csv')
                
        else:
            num_nodes, node_lower_range, node_upper_range, num_records, lower_num_prop, upper_num_prop, node_feature_names, num_edge_features, edge_density, edge_feature_names, noise, num_control_points, edge_weight_lower, edge_weight_upper = StaticHeterogenous.input()
            
            node_data = StaticHeterogenous.generate_node_data(
                num_nodes=num_nodes, 
                num_records=num_records, 
                noise=noise, 
                num_control_points=num_control_points,
                lower_num_prop=lower_num_prop,
                upper_num_prop=upper_num_prop,
                features=node_feature_names,
                node_lower_range=node_lower_range, 
                node_upper_range=node_upper_range,
            )
            
            edge_data = StaticHeterogenous.generate_edge_data(
                edge_density=edge_density,
                num_nodes=num_nodes,
                num_edge_features=num_edge_features,
                features=edge_feature_names,
                edge_weight_lower=edge_weight_lower,
                edge_weight_upper=edge_weight_upper,
            )
            
            st.header("Generated Data")
            
            if len(node_data):
                st.subheader("Node Data")
                st.dataframe(node_data)
                
                st.download_button("Download Node Data", data=node_data.to_csv(), file_name='nodes.csv')
                
            if len(edge_data):
                st.subheader("Edge Data")
                st.dataframe(edge_data)

                st.download_button("Download Edge Data", data=edge_data.to_csv(), file_name='edges.csv')
                
    else:
        if node_type == node_types[0]:
            num_nodes, node_lower_range, node_upper_range, num_records, num_prop, node_feature_names,  num_edge_features, edge_density, edge_feature_names, new_edge_likelihood, delete_edge_likelihood ,edge_determination, noise, num_control_points, edge_weight_lower, edge_weight_upper = DynamicHomogenous.input()
            
            node_data = DynamicHomogenous.generate_node_data(
                num_nodes=num_nodes, 
                num_records=num_records,
                num_prop=num_prop, 
                noise=noise, 
                num_control_points=num_control_points,
                features=node_feature_names,
                node_lower_range=node_lower_range, 
                node_upper_range=node_upper_range,
            )
            
            edge_data = DynamicHomogenous.generate_edge_data(
               num_nodes = num_nodes, 
               num_records = num_records, 
               edge_density = edge_density, 
               new_edge_likelihood = new_edge_likelihood, 
               delete_edge_likelihood = delete_edge_likelihood, 
               edge_determination = edge_determination,
               num_edge_features=num_edge_features,
               features=edge_feature_names,
               edge_weight_lower=edge_weight_lower,
                edge_weight_upper=edge_weight_upper,
            )
            
            st.header("Generated Data")
            
            if len(node_data):
                st.subheader("Node Data")
                st.dataframe(node_data)
                
                st.download_button("Download Node Data", data=node_data.to_csv(), file_name='nodes.csv')
                
            if len(edge_data):
                st.subheader("Edge Data")
                st.dataframe(edge_data)

                st.download_button("Download Edge Data", data=edge_data.to_csv(), file_name='edges.csv')
        else:
            num_nodes, node_lower_range, node_upper_range, num_records, lower_num_prop, upper_num_prop, node_feature_names, num_edge_features, edge_density, edge_feature_names, new_edge_likelihood, delete_edge_likelihood, edge_determination, noise, num_control_points, edge_weight_lower, edge_weight_upper = DynamicHeterogenous.input()
            
            node_data = DynamicHeterogenous.generate_node_data(
                num_nodes=num_nodes, 
                num_records=num_records,
                noise=noise, 
                num_control_points=num_control_points,
                lower_num_prop=lower_num_prop,
                upper_num_prop=upper_num_prop,
                features=node_feature_names,
                node_lower_range=node_lower_range, 
                node_upper_range=node_upper_range,
            )
            
            edge_data = DynamicHeterogenous.generate_edge_data(
                num_nodes = num_nodes, 
                num_records = num_records, 
                edge_density = edge_density, 
                new_edge_likelihood = new_edge_likelihood, 
                delete_edge_likelihood = delete_edge_likelihood, 
                edge_determination = edge_determination,
                num_edge_features=num_edge_features,
                features=edge_feature_names,
                edge_weight_lower = edge_weight_lower, 
                edge_weight_upper = edge_weight_upper,
            )
            
            st.header("Generated Data")
            
            if len(node_data):
                st.subheader("Node Data")
                st.dataframe(node_data)
                
                st.download_button("Download Node Data", data=node_data.to_csv(), file_name='nodes.csv')
                
            if len(edge_data):
                st.subheader("Edge Data")
                st.dataframe(edge_data)

                st.download_button("Download Edge Data", data=edge_data.to_csv(), file_name='edges.csv')

if __name__ == "__main__":
    main()