#!/usr/bin/env python3
"""
Generate visualizations for DSA Chapter 5 - Binary Trees
Creates all required images for binary search trees, heaps, and Huffman trees
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

# Color scheme
NODE_COLOR = '#E3F2FD'  # Light blue
NODE_EDGE_COLOR = '#1976D2'  # Darker blue
TEXT_COLOR = '#000000'  # Black
EDGE_COLOR = '#424242'  # Dark gray
HIGHLIGHT_COLOR = '#FFF59D'  # Light yellow for highlighting


class BinaryTreeVisualizer:
    """Helper class for drawing binary trees"""

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax

    def draw_node(self, x, y, value, radius=0.3, highlight=False):
        """Draw a single node"""
        color = HIGHLIGHT_COLOR if highlight else NODE_COLOR
        circle = plt.Circle((x, y), radius, color=color,
                           ec=NODE_EDGE_COLOR, linewidth=2, zorder=2)
        self.ax.add_patch(circle)
        self.ax.text(x, y, str(value), ha='center', va='center',
                    fontsize=12, fontweight='bold', zorder=3)

    def draw_edge(self, x1, y1, x2, y2, label=None):
        """Draw an edge between two nodes"""
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='-', color=EDGE_COLOR,
                               linewidth=1.5, zorder=1)
        self.ax.add_patch(arrow)

        if label is not None:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            self.ax.text(mid_x, mid_y, label, fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white', edgecolor='none'))

    def setup_axes(self, xlim, ylim, title):
        """Setup axes properties"""
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)


def generate_bst_example():
    """Generate BST example with inorder traversal result"""
    fig, ax = plt.subplots(figsize=(12, 8))
    vis = BinaryTreeVisualizer(fig, ax)

    # Tree structure: 53 is root
    # Left subtree: 09, 45, 23, 17
    # Right subtree: 65, 81, 78, 87, 94

    # Node positions (manually calculated for clarity)
    nodes = {
        53: (6, 7),
        45: (3, 5.5),
        65: (9, 5.5),
        9: (1, 4),
        23: (2.5, 4),
        81: (10, 4),
        17: (2, 2.5),
        78: (9.5, 2.5),
        87: (10.5, 2.5),
        94: (11, 1)
    }

    # Edges
    edges = [
        (53, 45), (53, 65),
        (45, 9), (45, 23),
        (65, 81),
        (23, 17),
        (81, 78), (81, 87),
        (87, 94)
    ]

    # Draw edges first
    for parent, child in edges:
        x1, y1 = nodes[parent]
        x2, y2 = nodes[child]
        vis.draw_edge(x1, y1, x2, y2)

    # Draw nodes (format 9 as "09" for display)
    for value, (x, y) in nodes.items():
        display_val = f"{value:02d}" if value < 100 else str(value)
        vis.draw_node(x, y, display_val)

    # Add inorder traversal result
    inorder = "Inorder Traversal: 09, 17, 23, 45, 53, 65, 78, 81, 87, 94"
    ax.text(6, 0.5, inorder, ha='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4',
                     edgecolor='#F57C00', linewidth=2))

    vis.setup_axes((-0.8, 12.5), (-0.3, 8.3), 'Binary Search Tree (BST) Example')
    plt.tight_layout(pad=1.5)
    plt.savefig('bst-example.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close()
    print("✓ Generated bst-example.png")


def generate_heap_example(is_min=True):
    """Generate min-heap or max-heap example"""
    fig, ax = plt.subplots(figsize=(10, 7))
    vis = BinaryTreeVisualizer(fig, ax)

    if is_min:
        # Min heap: parent <= children
        values = [12, 14, 15, 19, 20, 17, 18, 24, 22, 26]
        title = 'Min Heap Example'
        filename = 'min-heap-example.png'
    else:
        # Max heap: parent >= children
        values = [26, 24, 18, 22, 20, 17, 15, 19, 14, 12]
        title = 'Max Heap Example'
        filename = 'max-heap-example.png'

    # Complete binary tree layout
    positions = [
        (5, 6),      # 0: root
        (2.5, 4.5),  # 1: left child of 0
        (7.5, 4.5),  # 2: right child of 0
        (1, 3),      # 3: left child of 1
        (4, 3),      # 4: right child of 1
        (6, 3),      # 5: left child of 2
        (9, 3),      # 6: right child of 2
        (0.5, 1.5),  # 7: left child of 3
        (1.5, 1.5),  # 8: right child of 3
        (3.5, 1.5)   # 9: left child of 4
    ]

    # Draw edges
    edges = [(0,1), (0,2), (1,3), (1,4), (2,5), (2,6), (3,7), (3,8), (4,9)]
    for parent, child in edges:
        x1, y1 = positions[parent]
        x2, y2 = positions[child]
        vis.draw_edge(x1, y1, x2, y2)

    # Draw nodes
    for i, value in enumerate(values):
        x, y = positions[i]
        vis.draw_node(x, y, value)

    # Add array representation
    array_str = f"Array: [{', '.join(map(str, values))}]"
    ax.text(5, 0.5, array_str, ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9',
                     edgecolor='#4CAF50', linewidth=2))

    vis.setup_axes((-0.8, 10.5), (-0.3, 7.3), title)
    plt.tight_layout(pad=1.5)
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close()
    print(f"✓ Generated {filename}")


def generate_huffman_wpl():
    """Generate Huffman WPL comparison with 4 different trees"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Huffman Tree - Weighted Path Length (WPL) Comparison',
                 fontsize=16, fontweight='bold')

    trees = [
        # Tree (a) - balanced
        {
            'nodes': {1:(1,2), 3:(3,2), 5:(5,2), 7:(7,2)},
            'edges': [],
            'wpl': 32,
            'calc': 'WPL = 1×2 + 3×2 + 5×2 + 7×2 = 32'
        },
        # Tree (b)
        {
            'nodes': {7:(4,4), 1:(2,2.5), 3:(3,2), 5:(4,2)},
            'edges': [(7,1), (7,3), (7,5)],
            'wpl': 33,
            'calc': 'WPL = 1×2 + 3×3 + 5×3 + 7×1 = 33'
        },
        # Tree (c)
        {
            'nodes': {7:(2,4), 5:(3,3), 3:(2,2), 1:(3,1)},
            'edges': [(7,5), (5,3), (3,1)],
            'wpl': 43,
            'calc': 'WPL = 7×3 + 5×3 + 3×2 + 1×1 = 43'
        },
        # Tree (d) - Huffman optimal
        {
            'nodes': {7:(3,4), 1:(1,2), 3:(2,2), 5:(4,3)},
            'edges': [(7,1), (7,5), (5,3)],
            'wpl': 29,
            'calc': 'WPL = 1×3 + 3×3 + 5×2 + 7×1 = 29 (Optimal!)'
        }
    ]

    for idx, (ax, tree_data) in enumerate(zip(axes.flat, trees)):
        vis = BinaryTreeVisualizer(fig, ax)

        # Draw edges
        for parent, child in tree_data['edges']:
            x1, y1 = tree_data['nodes'][parent]
            x2, y2 = tree_data['nodes'][child]
            vis.draw_edge(x1, y1, x2, y2)

        # Draw nodes
        highlight = (idx == 3)  # Highlight optimal tree
        for value, (x, y) in tree_data['nodes'].items():
            vis.draw_node(x, y, value, radius=0.25, highlight=highlight)

        # Add WPL calculation
        ax.text(2.5, 0.5, tree_data['calc'], ha='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='#FFF59D' if highlight else '#E3F2FD',
                        edgecolor='#F57C00' if highlight else '#1976D2'))

        vis.setup_axes((-0.8, 5.5), (-0.3, 5.3), f'Tree ({chr(97+idx)})')

    plt.tight_layout(pad=1.5)
    plt.savefig('huffman-tree-wpl.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close()
    print("✓ Generated huffman-tree-wpl.png")


def generate_huffman_construction():
    """Generate step-by-step Huffman tree construction"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Huffman Tree Construction Process (Weights: 2, 4, 5, 7)',
                 fontsize=16, fontweight='bold')

    steps = [
        # Step 1: Initial
        {'title': 'Step 1: Initial', 'nodes': {2:(1,1), 4:(3,1), 5:(5,1), 7:(7,1)}, 'edges': []},
        # Step 2: Merge 2,4 -> 6
        {'title': 'Step 2: Merge {2,4}', 'nodes': {2:(1,1), 4:(2,1), 6:(1.5,2), 5:(4,1), 7:(6,1)},
         'edges': [(6,2), (6,4)]},
        # Step 3: Merge 5,6 -> 11
        {'title': 'Step 3: Merge {5,6}', 'nodes': {2:(0.5,1), 4:(1.5,1), 6:(1,2), 5:(3,1), 11:(2,3), 7:(5,1)},
         'edges': [(6,2), (6,4), (11,6), (11,5)]},
        # Step 4: Merge 7,11 -> 18
        {'title': 'Step 4: Merge {7,11}', 'nodes': {2:(0.5,1), 4:(1.5,1), 6:(1,2), 5:(3,1), 11:(2,3), 7:(5,1), 18:(3.5,4)},
         'edges': [(6,2), (6,4), (11,6), (11,5), (18,11), (18,7)]},
        # Step 5: Final tree with labels
        {'title': 'Step 5: Final Tree', 'nodes': {2:(1,1), 4:(2,1), 6:(1.5,2.5), 5:(4,1), 11:(2.5,3.5), 7:(6,1), 18:(4,5)},
         'edges': [(6,2), (6,4), (11,6), (11,5), (18,11), (18,7)],
         'edge_labels': {(18,11):'0', (18,7):'1', (11,6):'0', (11,5):'1', (6,2):'0', (6,4):'1'}}
    ]

    for ax, step_data in zip(axes.flat, steps):
        vis = BinaryTreeVisualizer(fig, ax)

        # Draw edges
        for parent, child in step_data['edges']:
            x1, y1 = step_data['nodes'][parent]
            x2, y2 = step_data['nodes'][child]
            label = step_data.get('edge_labels', {}).get((parent, child))
            vis.draw_edge(x1, y1, x2, y2, label=label)

        # Draw nodes
        for value, (x, y) in step_data['nodes'].items():
            vis.draw_node(x, y, value, radius=0.25)

        vis.setup_axes((-0.8, 7.5), (-0.3, 6.3), step_data['title'])

    # Hide unused subplot
    axes.flat[-1].axis('off')

    plt.tight_layout(pad=1.5)
    plt.savefig('huffman-tree-construction.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close()
    print("✓ Generated huffman-tree-construction.png")


def generate_huffman_encoding():
    """Generate Huffman encoding example with frequencies"""
    fig, ax = plt.subplots(figsize=(12, 8))
    vis = BinaryTreeVisualizer(fig, ax)

    # Character frequencies: a:15, b:2, c:6, d:5, e:20, f:10, g:18
    # Tree structure (pre-calculated)
    nodes = {
        'root': (8, 7),
        '33': (4, 5.5),
        '43': (11, 5.5),
        '13': (2, 4),
        'e': (6, 4),
        'g': (10, 4),
        '23': (13, 4),
        '7': (1, 2.5),
        'a': (3, 2.5),
        'f': (12, 2.5),
        'c': (0.5, 1),
        'd': (1.5, 1),
        'b': (14, 2.5)
    }

    edges = [
        ('root', '33', '0'), ('root', '43', '1'),
        ('33', '13', '0'), ('33', 'e', '1'),
        ('43', 'g', '0'), ('43', '23', '1'),
        ('13', '7', '0'), ('13', 'a', '1'),
        ('7', 'c', '0'), ('7', 'd', '1'),
        ('23', 'f', '0'), ('23', 'b', '1')
    ]

    # Draw edges with labels
    for parent, child, label in edges:
        x1, y1 = nodes[parent]
        x2, y2 = nodes[child]
        vis.draw_edge(x1, y1, x2, y2, label=label)

    # Draw nodes (highlight leaf nodes)
    leaves = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    for node, (x, y) in nodes.items():
        vis.draw_node(x, y, node, radius=0.25, highlight=(node in leaves))

    # Add encoding table
    codes = [
        ('a', '010', 15), ('b', '111', 2), ('c', '0000', 6), ('d', '0001', 5),
        ('e', '01', 20), ('f', '110', 10), ('g', '10', 18)
    ]

    table_text = "Huffman Codes:\n"
    for char, code, freq in codes:
        table_text += f"{char}:{code:5s} (freq:{freq:2d})  "
        if char == 'd':
            table_text += "\n"

    ax.text(8, 0.3, table_text, ha='center', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9',
                     edgecolor='#4CAF50', linewidth=2))

    vis.setup_axes((-1.5, 15.5), (-0.8, 8.3),
                   'Huffman Encoding (frequencies: a:15, b:2, c:6, d:5, e:20, f:10, g:18)')
    plt.tight_layout(pad=1.5)
    plt.savefig('huffman-encoding-example.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close()
    print("✓ Generated huffman-encoding-example.png")


def generate_huffman_cast():
    """Generate simple CAST Huffman tree example"""
    fig, ax = plt.subplots(figsize=(10, 7))
    vis = BinaryTreeVisualizer(fig, ax)

    # CAST: C:2, A:7, S:4, T:5
    # Correct structure: merge(2,4)=6, merge(5,6)=11, merge(7,11)=18
    nodes = {
        18: (5, 5),
        'A': (2, 3.5),
        11: (7, 3.5),
        'T': (5.5, 2),
        6: (8, 2),
        'C': (7, 0.5),
        'S': (9, 0.5)
    }

    edges = [
        (18, 'A', '0'), (18, 11, '1'),
        (11, 'T', '0'), (11, 6, '1'),
        (6, 'C', '0'), (6, 'S', '1')
    ]

    # Draw edges
    for parent, child, label in edges:
        x1, y1 = nodes[parent]
        x2, y2 = nodes[child]
        vis.draw_edge(x1, y1, x2, y2, label=label)

    # Draw nodes
    leaves = ['A', 'C', 'S', 'T']
    for node, (x, y) in nodes.items():
        vis.draw_node(x, y, node, radius=0.3, highlight=(node in leaves))

    # Add comparison table
    comparison = """Huffman Encoding: A:0  T:10  C:110  S:111
Length: 7×1 + 5×2 + (2+4)×3 = 35

Fixed-Length Encoding: A:00  T:10  C:01  S:11
Length: (2+7+4+5)×2 = 36

Savings: 1 bit (2.8%)"""

    ax.text(5, -1.5, comparison, ha='center', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4',
                     edgecolor='#F57C00', linewidth=2))

    vis.setup_axes((-0.5, 9.5), (-2.8, 6.3),
                   'CAST Example (C:2, A:7, S:4, T:5)')
    plt.tight_layout(pad=1.5)
    plt.savefig('huffman-cast-example.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close()
    print("✓ Generated huffman-cast-example.png")


if __name__ == '__main__':
    print("Generating binary tree visualizations...")
    print("=" * 60)

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Generate all images
    generate_bst_example()
    generate_heap_example(is_min=True)
    generate_heap_example(is_min=False)
    generate_huffman_wpl()
    generate_huffman_construction()
    generate_huffman_encoding()
    generate_huffman_cast()

    print("=" * 60)
    print("✓ All images generated successfully!")
    print("\nGenerated files:")
    print("  - bst-example.png")
    print("  - min-heap-example.png")
    print("  - max-heap-example.png")
    print("  - huffman-tree-wpl.png")
    print("  - huffman-tree-construction.png")
    print("  - huffman-encoding-example.png")
    print("  - huffman-cast-example.png")
