#!/usr/bin/env python3
"""
Universal PDF Image Extractor for Academic Notes

This tool extracts images from PDF files for use in Jekyll blog posts.
Supports both extracting embedded images and rendering pages as images.

Usage:
    python3 extract_pdf_images.py <pdf_path> <output_dir> [options]

Requirements:
    pip install PyMuPDF Pillow

Example:
    # Extract all embedded images
    python3 extract_pdf_images.py input.pdf output_dir/

    # Extract specific pages as images
    python3 extract_pdf_images.py input.pdf output_dir/ --pages 5,10,15

    # Extract with interactive mode
    python3 extract_pdf_images.py input.pdf output_dir/ --interactive
"""

import fitz  # PyMuPDF
from pathlib import Path
import io
from PIL import Image
import argparse
import sys

class PDFImageExtractor:
    """Extract images from PDF files"""

    def __init__(self, pdf_path, output_dir):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self.pdf_document = fitz.open(str(self.pdf_path))
        self.total_pages = len(self.pdf_document)

    def extract_embedded_images(self, page_range=None):
        """
        Extract all embedded images from PDF

        Args:
            page_range: Optional tuple (start, end) for page range
        """
        image_count = 0

        print("="*60)
        print(f"Extracting embedded images from: {self.pdf_path.name}")
        print(f"Total pages: {self.total_pages}")
        print("="*60)

        start_page = page_range[0] if page_range else 0
        end_page = page_range[1] if page_range else self.total_pages

        for page_num in range(start_page, end_page):
            page = self.pdf_document[page_num]
            image_list = page.get_images()

            if image_list:
                print(f"\nPage {page_num + 1}: Found {len(image_list)} image(s)")

                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = self.pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Save the image
                    image_filename = f"page_{page_num+1:02d}_img_{img_index+1}.{image_ext}"
                    image_path = self.output_dir / image_filename

                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)

                    # Get image dimensions
                    img_obj = Image.open(io.BytesIO(image_bytes))
                    width, height = img_obj.size

                    print(f"  ✓ Extracted: {image_filename} ({width}x{height})")
                    image_count += 1

        print("\n" + "="*60)
        print(f"Total images extracted: {image_count}")
        print(f"Output directory: {self.output_dir}")
        print("="*60)

        return image_count

    def extract_page_as_image(self, page_num, output_filename=None, dpi=300, crop_box=None):
        """
        Extract a specific page or region as an image

        Args:
            page_num: Page number (0-indexed)
            output_filename: Output filename (default: page_XX.png)
            dpi: Resolution for rendering (default: 300)
            crop_box: Optional (x0, y0, x1, y1) to crop specific region
        """
        if page_num < 0 or page_num >= self.total_pages:
            raise ValueError(f"Invalid page number: {page_num + 1}")

        page = self.pdf_document[page_num]

        # Set zoom factor based on DPI
        zoom = dpi / 72  # 72 is the default DPI
        mat = fitz.Matrix(zoom, zoom)

        if crop_box:
            # Crop to specific region
            rect = fitz.Rect(crop_box)
            pix = page.get_pixmap(matrix=mat, clip=rect)
        else:
            # Full page
            pix = page.get_pixmap(matrix=mat)

        # Generate filename
        if output_filename is None:
            output_filename = f"page_{page_num+1:02d}.png"

        output_path = self.output_dir / output_filename
        pix.save(str(output_path))

        print(f"✓ Saved page {page_num + 1} to {output_filename}")
        return output_path

    def extract_pages_as_images(self, page_numbers, dpi=300):
        """
        Extract multiple pages as images

        Args:
            page_numbers: List of page numbers (0-indexed)
            dpi: Resolution for rendering
        """
        print("="*60)
        print(f"Extracting pages as images from: {self.pdf_path.name}")
        print(f"Pages to extract: {[p+1 for p in page_numbers]}")
        print("="*60)

        for page_num in page_numbers:
            self.extract_page_as_image(page_num, dpi=dpi)

        print("\n" + "="*60)
        print(f"Total pages extracted: {len(page_numbers)}")
        print(f"Output directory: {self.output_dir}")
        print("="*60)

    def interactive_mode(self):
        """Interactive mode to preview and select images"""
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        print(f"PDF: {self.pdf_path.name}")
        print(f"Total pages: {self.total_pages}")
        print("\nOptions:")
        print("  1. Extract all embedded images")
        print("  2. Extract specific pages as images")
        print("  3. List pages with embedded images")
        print("  4. Exit")
        print("="*60)

        choice = input("\nSelect option (1-4): ").strip()

        if choice == "1":
            self.extract_embedded_images()
        elif choice == "2":
            page_input = input("Enter page numbers (comma-separated, 1-indexed): ").strip()
            try:
                page_numbers = [int(p.strip()) - 1 for p in page_input.split(",")]
                dpi = input("Enter DPI (default 300): ").strip() or "300"
                self.extract_pages_as_images(page_numbers, dpi=int(dpi))
            except ValueError:
                print("❌ Invalid input")
        elif choice == "3":
            self.list_pages_with_images()
        elif choice == "4":
            print("Exiting...")
        else:
            print("❌ Invalid option")

    def list_pages_with_images(self):
        """List all pages that contain embedded images"""
        print("\n" + "="*60)
        print("PAGES WITH EMBEDDED IMAGES")
        print("="*60)

        for page_num in range(self.total_pages):
            page = self.pdf_document[page_num]
            image_list = page.get_images()

            if image_list:
                print(f"Page {page_num + 1}: {len(image_list)} image(s)")

        print("="*60)

    def close(self):
        """Close the PDF document"""
        self.pdf_document.close()

def main():
    parser = argparse.ArgumentParser(
        description="Extract images from PDF files for academic notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all embedded images
  python3 extract_pdf_images.py lecture.pdf output_dir/

  # Extract specific pages as images
  python3 extract_pdf_images.py lecture.pdf output_dir/ --pages 5,10,15

  # Interactive mode
  python3 extract_pdf_images.py lecture.pdf output_dir/ --interactive

  # Extract with custom DPI
  python3 extract_pdf_images.py lecture.pdf output_dir/ --pages 5 --dpi 600
        """
    )

    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("output_dir", help="Output directory for images")
    parser.add_argument("--pages", help="Comma-separated page numbers to extract (1-indexed)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for page rendering (default: 300)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--list", "-l", action="store_true", help="List pages with images")

    args = parser.parse_args()

    try:
        extractor = PDFImageExtractor(args.pdf_path, args.output_dir)

        if args.interactive:
            extractor.interactive_mode()
        elif args.list:
            extractor.list_pages_with_images()
        elif args.pages:
            page_numbers = [int(p.strip()) - 1 for p in args.pages.split(",")]
            extractor.extract_pages_as_images(page_numbers, dpi=args.dpi)
        else:
            # Default: extract all embedded images
            extractor.extract_embedded_images()

        extractor.close()

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
