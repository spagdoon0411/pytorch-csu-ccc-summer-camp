import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn


def visualize_neural_network(model):
    """
    Visualize a PyTorch neural network as a graph using NetworkX.
    Shows neurons and connections without activation functions.
    """
    G = nx.DiGraph()

    # Get model layers
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers.append((name, module))

    if not layers:
        print("No linear layers found in the model")
        return G

    # Create nodes for each layer
    node_positions = {}
    y_spacing = 1.0

    # Create input layer nodes
    first_layer_name, first_layer = layers[0]
    x_pos = 0
    for neuron_idx in range(first_layer.in_features):
        node_id = f"input_{neuron_idx}"
        y_pos = (neuron_idx - first_layer.in_features / 2) * y_spacing
        node_positions[node_id] = (x_pos, y_pos)
        G.add_node(node_id, layer="input", type="input")

    # Create hidden and output layer nodes
    for layer_idx, (layer_name, layer_module) in enumerate(layers):
        num_neurons = layer_module.out_features
        x_pos = (layer_idx + 1) * 2.0

        for neuron_idx in range(num_neurons):
            node_id = f"{layer_name}_out_{neuron_idx}"
            y_pos = (neuron_idx - num_neurons / 2) * y_spacing
            node_positions[node_id] = (x_pos, y_pos)

            if layer_idx == len(layers) - 1:
                G.add_node(node_id, layer=layer_name, type="output")
            else:
                G.add_node(node_id, layer=layer_name, type="hidden")

    # Add edges between layers
    for i in range(len(layers)):
        layer_name, layer_module = layers[i]

        if i == 0:
            # Connect input nodes to first layer
            for in_neuron in range(layer_module.in_features):
                for out_neuron in range(layer_module.out_features):
                    from_node = f"input_{in_neuron}"
                    to_node = f"{layer_name}_out_{out_neuron}"
                    G.add_edge(from_node, to_node)
        else:
            # Connect previous layer to current layer
            prev_layer_name, prev_layer = layers[i - 1]
            for in_neuron in range(prev_layer.out_features):
                for out_neuron in range(layer_module.out_features):
                    from_node = f"{prev_layer_name}_out_{in_neuron}"
                    to_node = f"{layer_name}_out_{out_neuron}"
                    G.add_edge(from_node, to_node)

    # Create the visualization
    plt.figure(figsize=(12, 8))

    # Draw nodes with different colors based on type
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node]["type"]
        if node_type == "input":
            node_colors.append("lightblue")
        elif node_type == "hidden":
            node_colors.append("lightgreen")
        else:  # output
            node_colors.append("lightcoral")

    # Draw the graph
    nx.draw(
        G,
        pos=node_positions,
        node_color=node_colors,
        node_size=300,
        with_labels=False,
        arrows=True,
        edge_color="gray",
        alpha=0.7,
    )

    # Add layer labels
    # Input layer label
    plt.text(
        0,
        -3,
        "Input",
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Hidden and output layer labels
    for layer_idx, (layer_name, _) in enumerate(layers):
        x_pos = (layer_idx + 1) * 2.0
        layer_type = (
            "Output" if layer_idx == len(layers) - 1 else f"Hidden {layer_idx + 1}"
        )
        plt.text(
            x_pos,
            -3,
            f"{layer_type}\n({layer_name})",
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.title(
        "Neural Network Architecture (Neurons Only)", fontsize=14, fontweight="bold"
    )
    plt.axis("off")
    plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin to prevent label cutoff
    # plt.show()

    return G
