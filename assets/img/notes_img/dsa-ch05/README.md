# DSA Chapter 5 - Binary Tree Images

## Existing Images (Old)

1. ✅ `binary-tree-definition.jpg` - Binary tree definition
2. ✅ `binary-tree-forms.jpg` - Five basic forms
3. ✅ `binary-tree-properties.jpg` - Properties illustration
4. ✅ `binary-tree-traversal-example.jpg` - Traversal example
5. ✅ `complete-binary-tree.jpg` - Complete binary tree
6. ✅ `full-binary-tree.jpg` - Full binary tree

## Required New Images (Generated ✓)

### Binary Search Tree (BST)

- ✅ `bst-example.png` - BST example with values 09, 17, 23, 45, 53, 65, 78, 81, 87, 94
  - Referenced in: line 433
  - Source: **Python generated** (matplotlib)
  - Shows: Inorder traversal result demonstrating sorted order

### Heap & Priority Queue

- ✅ `min-heap-example.png` - Min heap example

  - Referenced in: line 542
  - Source: **Python generated** (matplotlib)
  - Shows: Complete binary tree with values {12, 14, 15, 19, 20, 17, 18, 24, 22, 26}

- ✅ `max-heap-example.png` - Max heap example
  - Referenced in: line 545
  - Source: **Python generated** (matplotlib)
  - Shows: Same values organized as max heap

### Huffman Tree

- ✅ `huffman-tree-wpl.png` - Weighted path length examples

  - Referenced in: line 672
  - Source: **Python generated** (matplotlib)
  - Shows: Four binary trees (a), (b), (c), (d) with weights {1, 3, 5, 7} and WPL calculations

- ✅ `huffman-tree-construction.png` - Huffman tree construction steps

  - Referenced in: line 689
  - Source: **Python generated** (matplotlib)
  - Shows: Step-by-step merging process from {2, 4, 5, 7} to final tree

- ✅ `huffman-encoding-example.png` - Huffman encoding with frequencies

  - Referenced in: line 716
  - Source: **Python generated** (matplotlib)
  - Shows: Complete tree with frequencies {a:15, b:2, c:6, d:5, e:20, f:10, g:18} and binary codes

- ✅ `huffman-cast-example.png` - CAST example
  - Referenced in: line 752
  - Source: **Python generated** (matplotlib)
  - Shows: Simple Huffman tree for {C:2, A:7, S:4, T:5} with encoding comparison

## Generation Instructions

All new images are generated using Python (matplotlib) for better quality and control.

### To regenerate all images:

```bash
cd assets/img/notes_img/dsa-ch05/
python3 generate_images.py
```

### Generation Script Features:

- High-resolution output (300 DPI)
- Consistent visual style
- English-only labels
- Clean, academic presentation
- Proper node spacing and edge routing

### Customization:

Edit `generate_images.py` to modify:

- Node colors and sizes
- Font sizes and styles
- Tree layouts and spacing
- Edge styles and labels

## Image Requirements

- **Format**: PNG (transparency support)
- **Resolution**: ≥800px width
- **Labels**: English only
- **Quality**: Clear, no blur or artifacts
- **Background**: White or transparent

## Notes

- All images should be extracted from official PKU course materials
- Maintain consistent visual style across all diagrams
- Binary tree node layout should be clear and readable
- Color coding: nodes in light blue/gray, text in black, edges in gray
