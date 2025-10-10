# Tools for Academic Note Generation

This directory contains utility scripts for creating and managing academic notes.

## Available Tools

### 1. extract_pdf_images.py

Universal PDF image extractor for academic notes. Extracts images from PDF files for use in Jekyll blog posts.

**Requirements:**

```bash
pip install PyMuPDF Pillow
# or
conda install -c conda-forge pymupdf pillow
```

**Usage:**

```bash
# Extract all embedded images from PDF
python3 _tools/extract_pdf_images.py lecture.pdf assets/img/notes_img/chapter/

# Extract specific pages as images (1-indexed)
python3 _tools/extract_pdf_images.py lecture.pdf output/ --pages 5,10,15

# Extract with custom DPI
python3 _tools/extract_pdf_images.py lecture.pdf output/ --pages 5 --dpi 600

# Interactive mode - browse and select images
python3 _tools/extract_pdf_images.py lecture.pdf output/ --interactive

# List all pages with embedded images
python3 _tools/extract_pdf_images.py lecture.pdf output/ --list
```

**Examples:**

```bash
# Example 1: Extract images from CV Chapter 7 PDF
python3 _tools/extract_pdf_images.py \
    /path/to/07_Epipolar_Geometry.pdf \
    assets/img/notes_img/cv-ch07/temp/

# Example 2: Extract specific diagram pages
python3 _tools/extract_pdf_images.py \
    textbook.pdf \
    assets/img/notes_img/cs231n-ch03/ \
    --pages 12,25,47

# Example 3: Interactive browsing
python3 _tools/extract_pdf_images.py lecture.pdf output/ -i
```

**Workflow:**

1. Extract images to a temporary directory
2. Review extracted images
3. Rename with descriptive names (e.g., `epipolar_constraint.png`)
4. Move selected images to the target chapter directory
5. Delete temporary extraction directory
6. Update chapter README.md to document image sources

## Best Practices

### Image Naming Convention

Use descriptive English names with lowercase and underscores:

- ✅ `stereo_camera_baseline.png`
- ✅ `epipolar_constraint_geometry.png`
- ✅ `fundamental_matrix_relationship.png`
- ❌ `图1.png` (Chinese)
- ❌ `img001.png` (non-descriptive)
- ❌ `screenshot.png` (too generic)

### Image Quality Standards

- **Resolution**: ≥ 300 DPI for generated images
- **Width**: ≥ 800px for extracted images
- **Format**: PNG preferred (supports transparency)
- **Clarity**: No blur, artifacts, or truncation
- **Labels**: English only, no Chinese text

### Scientific Accuracy Verification

**Always verify images for:**

- [ ] Correctness of geometric relationships
- [ ] Accuracy of mathematical formulas
- [ ] Completeness of labels and legends
- [ ] Consistency with course materials
- [ ] Appropriate for academic context

See `CLAUDE.md` for complete image processing workflow.

## Dependencies

- **PyMuPDF (fitz)**: PDF processing and image extraction
- **Pillow (PIL)**: Image manipulation and metadata reading
- **Python 3.7+**: Required Python version

## Troubleshooting

**Issue: `ModuleNotFoundError: No module named 'fitz'`**

```bash
pip install --user PyMuPDF
# or
conda install -c conda-forge pymupdf
```

**Issue: Images are low quality**

- Use `--dpi 600` for higher resolution page rendering
- For embedded images, quality depends on source PDF

**Issue: Too many extracted images**

- Use `--list` first to see which pages have images
- Use `--pages` to extract only specific pages
- Use `--interactive` mode for selective extraction

## Contributing

When adding new tools:

1. Add comprehensive docstring and comments
2. Include usage examples in the script
3. Update this README with tool description
4. Update `CLAUDE.md` if it affects the note-taking workflow

## License

These tools are part of the stibiums.github.io academic website repository.
