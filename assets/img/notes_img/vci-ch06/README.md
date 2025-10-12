# VCI Chapter 6 - Image Representation Images (Optimized)

## Current Images (Essential Only)

After optimization, only the following **3 essential images** are retained:

1. `inpainting_input.png` - Image inpainting input example ✅ **Extracted from PDF page 28**

   - Shows the region to be repaired (black area marked as Ω)
   - Essential for demonstrating the inpainting problem setup

2. `inpainting_result.png` - Image inpainting result ✅ **Extracted from PDF page 30**

   - Shows the repaired result after applying Laplace editing
   - Essential for comparing before/after inpainting effects

3. `poisson_cloning.png` - Poisson editing complete workflow ✅ **Extracted from PDF page 45**
   - Shows the complete image cloning process
   - Demonstrates source image, target scene, guidance field, and final result
   - Essential for understanding Poisson editing applications

## Removed Images (Replaced with Text Descriptions)

The following images were removed and replaced with concise text explanations:

### Conceptual Diagrams (Can be described in text)

- `image_definition.png` - Image formation process
- `framebuffer.png` - Framebuffer working principle
- `rgb_storage.png` - RGB color storage method
- `colormap_lut.png` - Color lookup table (LUT)
- `alpha_compositing.png` - Alpha channel compositing

### Comparison Diagrams (Redundant)

- `vector_vs_raster.png` - Vector vs Raster comparison
- `raster_vector_comparison.png` - Zoomed comparison (duplicate)
- `raster_vector_display.png` - Display technologies (unused)

### Example Images (Pure demonstrations)

- `blur_example.png` - Blur filter example
- `blur_filter.png` - Blur filter effects
- `edge_filter_example.png` - Edge filter example (unused)
- `edge_detection.png` - Edge detection results
- `gradient_sobel.png` - Sobel edge detection
- `edge_result.png` - Final edge detection result

### Formula Images (Already in LaTeX)

- `inpainting_formula.png` - Mathematical formulas (redundant with LaTeX)

### Cloning Examples (Redundant)

- `cloning_example1.png` - Cloning example 1
- `cloning_example2.png` - Cloning example 2 (multiple scenes)

## Optimization Summary

- **Original:** 21 image files (~15.5 MB)
- **Optimized:** 3 image files (~10.5 MB)
- **Reduction:** 85.7% fewer images, ~32% size reduction
- **Approach:** Keep only visual effects that cannot be described in text; replace conceptual diagrams with concise descriptions

## Extraction Info

- All remaining images extracted from `06-image.pdf` using `extract_pdf_images.py`
- Images renamed to descriptive names for better organization
- All images use English labels (no Chinese text)

## Image Usage in Notes

All images are inserted using Bootstrap responsive grid layout with:

- `loading="eager"` - Enable fast loading
- `zoomable=true` - Enable image zoom functionality
- `class="img-fluid rounded z-depth-1"` - Responsive, rounded corners, shadow effect
- Descriptive titles for accessibility

## Notes

- Only truly essential images retained: inpainting input/output and Poisson cloning workflow
- Conceptual diagrams replaced with clear text descriptions
- Mathematical formulas already expressed in LaTeX
- Filter effects described in text rather than shown visually
