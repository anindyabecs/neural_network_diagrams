import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.patches import Circle, FancyBboxPatch, ConnectionPatch
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set style for beautiful plots
plt.style.use('dark_background')
sns.set_palette("husl")

# =============================================================================
# 1. BEAUTIFUL NEURAL NETWORK DIAGRAM
# =============================================================================

def create_neural_network_diagram():
    """Create a stunning neural network visualization"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define network architecture
    layers = [4, 6, 6, 3, 1]  # nodes per layer
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Hidden\nLayer 3', 'Output\nLayer']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Calculate positions
    x_positions = np.linspace(1, 9, len(layers))
    node_positions = {}
    
    # Draw connections first (so they appear behind nodes)
    for i in range(len(layers) - 1):
        for j in range(layers[i]):
            for k in range(layers[i + 1]):
                x1 = x_positions[i]
                y1 = 4 + (j - (layers[i] - 1) / 2) * 0.8
                x2 = x_positions[i + 1]
                y2 = 4 + (k - (layers[i + 1] - 1) / 2) * 0.8
                
                # Weight-based line thickness and opacity
                weight = np.random.random()
                alpha = 0.1 + 0.4 * weight
                width = 0.5 + 1.5 * weight
                
                ax.plot([x1, x2], [y1, y2], 'white', alpha=alpha, linewidth=width)
    
    # Draw nodes
    for i, (num_nodes, color, name) in enumerate(zip(layers, colors, layer_names)):
        x = x_positions[i]
        
        # Draw layer label
        ax.text(x, 7.5, name, ha='center', va='center', 
               fontsize=12, fontweight='bold', color=color)
        
        for j in range(num_nodes):
            y = 4 + (j - (num_nodes - 1) / 2) * 0.8
            
            # Create gradient effect
            circle = Circle((x, y), 0.25, color=color, alpha=0.8, zorder=10)
            ax.add_patch(circle)
            
            # Add inner highlight
            highlight = Circle((x - 0.08, y + 0.08), 0.08, color='white', alpha=0.6, zorder=11)
            ax.add_patch(highlight)
            
            # Add activation values (simulated)
            if i > 0:  # Skip input layer
                activation = np.random.random()
                ax.text(x, y, f'{activation:.2f}', ha='center', va='center', 
                       fontsize=8, fontweight='bold', color='black', zorder=12)
    
    # Add title and decorations
    ax.text(5, 0.5, 'Deep Neural Network Architecture', 
           ha='center', va='center', fontsize=20, fontweight='bold', 
           color='white', bbox=dict(boxstyle="round,pad=0.3", facecolor='#2C3E50', alpha=0.8))
    
    # Add data flow arrows
    for i in range(len(layers) - 1):
        x1 = x_positions[i] + 0.5
        x2 = x_positions[i + 1] - 0.5
        y = 1.5
        
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                   arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=3, alpha=0.7))
    
    ax.text(5, 1, 'Forward Propagation', ha='center', va='center', 
           fontsize=12, color='#E74C3C', fontweight='bold')
    
    plt.tight_layout()
    return fig

# =============================================================================
# 2. ENTITY RELATIONSHIP DIAGRAM
# =============================================================================

def create_er_diagram():
    """Create a beautiful Entity Relationship Diagram"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define entities and their attributes
    entities = {
        'Customer': {
            'pos': (3, 9),
            'attributes': ['CustomerID*', 'Name', 'Email', 'Phone', 'Address'],
            'color': '#FF6B6B'
        },
        'Order': {
            'pos': (8, 9),
            'attributes': ['OrderID*', 'OrderDate', 'TotalAmount', 'Status'],
            'color': '#4ECDC4'
        },
        'Product': {
            'pos': (13, 9),
            'attributes': ['ProductID*', 'Name', 'Price', 'Category', 'Stock'],
            'color': '#45B7D1'
        },
        'OrderItem': {
            'pos': (8, 5),
            'attributes': ['OrderItemID*', 'Quantity', 'UnitPrice'],
            'color': '#96CEB4'
        },
        'Payment': {
            'pos': (3, 5),
            'attributes': ['PaymentID*', 'Amount', 'PaymentDate', 'Method'],
            'color': '#FFEAA7'
        }
    }
    
    # Define relationships
    relationships = [
        ('Customer', 'Order', '1:N', 'Places', (5.5, 9)),
        ('Order', 'Product', 'M:N', 'Contains', (10.5, 9)),
        ('Order', 'OrderItem', '1:N', 'Has', (8, 7)),
        ('Product', 'OrderItem', '1:N', 'Belongs to', (10.5, 7)),
        ('Order', 'Payment', '1:1', 'Paid by', (5.5, 7))
    ]
    
    # Draw entities
    for entity_name, entity_data in entities.items():
        x, y = entity_data['pos']
        color = entity_data['color']
        attributes = entity_data['attributes']
        
        # Entity box
        entity_box = FancyBboxPatch(
            (x-1.2, y-0.5), 2.4, 1,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='white',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(entity_box)
        
        # Entity name
        ax.text(x, y, entity_name, ha='center', va='center',
               fontsize=14, fontweight='bold', color='white')
        
        # Attributes
        for i, attr in enumerate(attributes):
            attr_y = y - 1.5 - i * 0.4
            
            # Attribute oval
            attr_box = FancyBboxPatch(
                (x-0.8, attr_y-0.15), 1.6, 0.3,
                boxstyle="round,pad=0.05",
                facecolor='lightgray',
                edgecolor=color,
                linewidth=1.5,
                alpha=0.8
            )
            ax.add_patch(attr_box)
            
            # Primary key styling
            if '*' in attr:
                ax.text(x, attr_y, attr, ha='center', va='center',
                       fontsize=10, fontweight='bold', color='black',
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='gold', alpha=0.7))
            else:
                ax.text(x, attr_y, attr, ha='center', va='center',
                       fontsize=10, color='black')
            
            # Connection line
            ax.plot([x, x], [y-0.5, attr_y+0.15], color=color, linewidth=1.5, alpha=0.7)
    
    # Draw relationships
    for rel in relationships:
        entity1, entity2, cardinality, rel_name, rel_pos = rel
        
        pos1 = entities[entity1]['pos']
        pos2 = entities[entity2]['pos']
        
        # Relationship diamond
        diamond = FancyBboxPatch(
            (rel_pos[0]-0.6, rel_pos[1]-0.3), 1.2, 0.6,
            boxstyle="round,pad=0.05",
            facecolor='#E74C3C',
            edgecolor='white',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(diamond)
        
        # Relationship name
        ax.text(rel_pos[0], rel_pos[1], rel_name, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')
        
        # Connection lines
        ax.plot([pos1[0]+1.2, rel_pos[0]-0.6], [pos1[1], rel_pos[1]], 
               'white', linewidth=3, alpha=0.8)
        ax.plot([rel_pos[0]+0.6, pos2[0]-1.2], [rel_pos[1], pos2[1]], 
               'white', linewidth=3, alpha=0.8)
        
        # Cardinality labels
        mid_x1 = (pos1[0] + rel_pos[0]) / 2
        mid_y1 = (pos1[1] + rel_pos[1]) / 2
        mid_x2 = (pos2[0] + rel_pos[0]) / 2
        mid_y2 = (pos2[1] + rel_pos[1]) / 2
        
        card_parts = cardinality.split(':')
        ax.text(mid_x1, mid_y1+0.3, card_parts[0], ha='center', va='center',
               fontsize=12, fontweight='bold', color='yellow',
               bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7))
        ax.text(mid_x2, mid_y2+0.3, card_parts[1], ha='center', va='center',
               fontsize=12, fontweight='bold', color='yellow',
               bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.7))
    
    # Title
    ax.text(8, 11.5, 'Entity Relationship Diagram - E-Commerce System', 
           ha='center', va='center', fontsize=18, fontweight='bold', 
           color='white', bbox=dict(boxstyle="round,pad=0.3", facecolor='#2C3E50', alpha=0.8))
    
    # Legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor='#FF6B6B', label='Customer Entity'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='#4ECDC4', label='Order Entity'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='#45B7D1', label='Product Entity'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='#E74C3C', label='Relationship'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor='gold', label='Primary Key')
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    return fig

# =============================================================================
# 3. ADVANCED NETWORK GRAPH WITH NETWORKX
# =============================================================================

def create_advanced_network():
    """Create an advanced network visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Create neural network graph
    G_nn = nx.DiGraph()
    
    # Add nodes with layers
    layers = [3, 5, 4, 2]
    node_id = 0
    pos = {}
    
    for layer_idx, layer_size in enumerate(layers):
        for node_idx in range(layer_size):
            G_nn.add_node(node_id, layer=layer_idx)
            x = layer_idx * 3
            y = (node_idx - (layer_size - 1) / 2) * 1.5
            pos[node_id] = (x, y)
            node_id += 1
    
    # Add edges between consecutive layers
    current_node = 0
    for layer_idx in range(len(layers) - 1):
        layer_start = current_node
        layer_end = current_node + layers[layer_idx]
        next_layer_start = layer_end
        next_layer_end = next_layer_start + layers[layer_idx + 1]
        
        for i in range(layer_start, layer_end):
            for j in range(next_layer_start, next_layer_end):
                weight = np.random.random()
                G_nn.add_edge(i, j, weight=weight)
        
        current_node = layer_end
    
    # Draw neural network
    ax1.set_title('Neural Network Graph', fontsize=16, fontweight='bold', color='white')
    
    # Get edge weights for coloring
    edges = G_nn.edges()
    weights = [G_nn[u][v]['weight'] for u, v in edges]
    
    # Draw edges with varying thickness and color
    nx.draw_networkx_edges(G_nn, pos, ax=ax1, edge_color=weights, 
                          edge_cmap=plt.cm.viridis, width=[w*3 for w in weights],
                          alpha=0.7, edge_vmin=0, edge_vmax=1)
    
    # Draw nodes with layer-based coloring
    node_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for layer_idx in range(len(layers)):
        layer_nodes = [n for n in G_nn.nodes() if G_nn.nodes[n]['layer'] == layer_idx]
        nx.draw_networkx_nodes(G_nn, pos, nodelist=layer_nodes, 
                              node_color=node_colors[layer_idx], 
                              node_size=800, alpha=0.9, ax=ax1)
    
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Create ER-style graph
    G_er = nx.Graph()
    
    # Add entity nodes
    entities = ['User', 'Post', 'Comment', 'Like', 'Follow']
    entity_pos = {
        'User': (0, 2),
        'Post': (3, 3),
        'Comment': (6, 2),
        'Like': (3, 0),
        'Follow': (0, -1)
    }
    
    for entity in entities:
        G_er.add_node(entity, type='entity')
    
    # Add relationships
    relationships = [
        ('User', 'Post', 'creates'),
        ('User', 'Comment', 'writes'),
        ('User', 'Like', 'gives'),
        ('User', 'Follow', 'has'),
        ('Post', 'Comment', 'receives'),
        ('Post', 'Like', 'gets')
    ]
    
    for u, v, rel_type in relationships:
        G_er.add_edge(u, v, relationship=rel_type)
    
    # Draw ER graph
    ax2.set_title('Entity Relationship Graph', fontsize=16, fontweight='bold', color='white')
    
    # Draw edges
    nx.draw_networkx_edges(G_er, entity_pos, ax=ax2, edge_color='white', 
                          width=2, alpha=0.8)
    
    # Draw entity nodes
    entity_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for i, entity in enumerate(entities):
        nx.draw_networkx_nodes(G_er, entity_pos, nodelist=[entity],
                              node_color=entity_colors[i], node_size=2000, 
                              alpha=0.9, ax=ax2)
    
    # Add labels
    nx.draw_networkx_labels(G_er, entity_pos, ax=ax2, font_size=12, 
                           font_weight='bold', font_color='white')
    
    # Add relationship labels
    edge_labels = {}
    for u, v, d in G_er.edges(data=True):
        edge_labels[(u, v)] = d['relationship']
    
    nx.draw_networkx_edge_labels(G_er, entity_pos, edge_labels, ax=ax2, 
                                font_size=10, font_color='yellow')
    
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

# =============================================================================
# 4. INTERACTIVE NETWORK WITH PLOTLY (Code only - for reference)
# =============================================================================

def create_plotly_network():
    """
    Create interactive network with Plotly (requires plotly)
    """
    code = '''
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np

def plotly_neural_network():
    # Create network
    G = nx.DiGraph()
    layers = [4, 6, 4, 2]
    
    # Add nodes and positions
    pos = {}
    node_id = 0
    node_trace = []
    
    for layer_idx, layer_size in enumerate(layers):
        for node_idx in range(layer_size):
            x = layer_idx * 2
            y = (node_idx - (layer_size - 1) / 2) * 1.5
            pos[node_id] = (x, y)
            G.add_node(node_id, layer=layer_idx, pos=(x, y))
            node_id += 1
    
    # Add edges
    current_node = 0
    edge_x, edge_y = [], []
    
    for layer_idx in range(len(layers) - 1):
        layer_start = current_node
        layer_end = current_node + layers[layer_idx]
        next_layer_start = layer_end
        next_layer_end = next_layer_start + layers[layer_idx + 1]
        
        for i in range(layer_start, layer_end):
            for j in range(next_layer_start, next_layer_end):
                x0, y0 = pos[i]
                x1, y1 = pos[j]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                G.add_edge(i, j, weight=np.random.random())
        
        current_node = layer_end
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(125, 125, 125, 0.5)'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_colors = ['red', 'blue', 'green', 'orange']
    colors = [node_colors[G.nodes[node]['layer']] for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=15,
            color=colors,
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Interactive Neural Network',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Neural Network Visualization",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002 ) ],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='black',
                       paper_bgcolor='black'
                   ))
    
    return fig
'''
    return code

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Creating Beautiful Neural Network and ER Diagrams...")
    
    # 1. Neural Network Diagram
    print("1. Creating Neural Network Diagram...")
    fig1 = create_neural_network_diagram()
    plt.show()
    
    # 2. Entity Relationship Diagram  
    print("2. Creating ER Diagram...")
    fig2 = create_er_diagram()
    plt.show()
    
    # 3. Advanced Network Graphs
    print("3. Creating Advanced Network Graphs...")
    fig3 = create_advanced_network()
    plt.show()
    
    # 4. Print Plotly code for interactive networks
    print("4. Plotly Interactive Network Code:")
    print(create_plotly_network())
    
    print("\nAll diagrams created successfully!")
    print("\nRequired packages:")
    print("- matplotlib")
    print("- networkx") 
    print("- numpy")
    print("- seaborn")
    print("- plotly (for interactive versions)")

# =============================================================================
# USAGE EXAMPLES AND TIPS
# =============================================================================

"""
INSTALLATION:
pip install matplotlib networkx numpy seaborn plotly

CUSTOMIZATION TIPS:

1. Neural Networks:
   - Adjust `layers` list for different architectures
   - Modify colors in `colors` list
   - Change node sizes and connection weights
   - Add dropout visualization or activation functions

2. ER Diagrams:
   - Add more entities in `entities` dict
   - Customize relationship types
   - Add weak entities or identifying relationships
   - Include constraints and business rules

3. Advanced Networks:
   - Use different layout algorithms: spring_layout, circular_layout, etc.
   - Add clustering and community detection
   - Implement force-directed layouts
   - Add animation for dynamic networks

4. Styling:
   - Use different matplotlib styles: 'seaborn', 'ggplot', etc.
   - Create custom color palettes
   - Add gradients and transparency effects
   - Implement 3D visualizations

5. Interactive Features:
   - Use Plotly for web-based interactions
   - Add hover information and clickable nodes
   - Implement zoom and pan functionality
   - Create animated transitions
"""