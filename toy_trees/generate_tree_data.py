import os
import numpy as np
import torch
import geoopt
import argparse
import h5py
import networkx as nx


def generate_regular_tree(d, depth):
    G = nx.Graph()
    node_counter = 0

    def add_node(parent, current_depth):
        nonlocal node_counter
        current_node = node_counter
        node_counter += 1
        G.add_node(current_node)
        if parent is not None:
            G.add_edge(parent, current_node)
        if current_depth < depth:
            # The root gets d children; every other node gets d-1 children.
            children_count = d if parent is None else d - 1
            for _ in range(children_count):
                add_node(current_node, current_depth + 1)

    add_node(parent=None, current_depth=0)
    return G

def gen_sampled_tree(d, depth, k_perturb=0.1, depth_perturb=0.1):
    G = generate_regular_tree(d, depth)
    pos = nx.spring_layout(G, dim=3, k=k_perturb, scale=10/(depth*(1+depth_perturb)))
    sampled_indices = np.random.choice(len(G.nodes()), size=100, replace=False)
    sampled_nodes = [list(G.nodes())[i] for i in sampled_indices]
    sampled_pos = [pos[node] for node in sampled_nodes]        
    sampled_pos = np.array(sampled_pos)
    return sampled_pos


def main():
    print('Generating data...')
    parser = argparse.ArgumentParser(
        description="Generate manifold Gaussian clusters and save the dataset."
    )
    # n_samples here represents the number of points per cluster.
    parser.add_argument("--n_datapoints", type=int, default=25000,
                        help="Number of distinct distributions to generate (default: 25000)")
    parser.add_argument("--outdir", type=str, default="output_data",
                        help="Output directory to save the data (default: output_data)")
    parser.add_argument("--file_name", type=str, default="manifold_data",
                        help="Output file name (default: manifold_data)")

    
    args = parser.parse_args()

    # Ensure the output directory exists.
    os.makedirs(args.outdir, exist_ok=True)

          
    all_datasets = []
    all_labels = []
    # Generate the graph
    combos = [(3,8),
            (4,5),
            (5,4),
            (5,5),
            (6,4),
            (7,4)]
    combo_key ={(3,8): 0,
                (4,5): 1,
                (5,4): 2,
                (5,5): 3,
                (6,4): 4,
                (7,4): 5}
    for i in range(args.n_datapoints):
        print('Generating data point', i)
        (d, depth) = combos[np.random.choice(len(combos))]
        sampled_pos = gen_sampled_tree(d, depth)
        all_datasets.append(sampled_pos)
        all_labels.append(combo_key[(d, depth)])
        i = i +1
       
    # Convert to single array
    all_datasets = np.stack(all_datasets)
    all_labels = np.array(all_labels)
    # Save the data.
    output_path = os.path.join(args.outdir, args.file_name+'.h5')
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('datasets', data=all_datasets)
        f.create_dataset('labels', data=all_labels)
    print('Data saved to', output_path)

if __name__ == "__main__":
    main()
