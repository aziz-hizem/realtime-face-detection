"""
Face detection on image files, from the command line.

Usage:
    python detect_images.py --input photo.jpg --show
    python detect_images.py --input photos/ --output results/
"""
import argparse
import os

import cv2
from face_detector import FaceDetector
from utils import load_image, convert_rgb_to_bgr, draw_face_box, save_image
from config import OUTPUT_DIR, BBOX_COLOR_FACE, BBOX_THICKNESS


def detect_image(image_path, output_path=None, show_result=False):
    """
    Detect faces in a single image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save annotated image (optional)
        show_result: Display result window
    
    Returns:
        List of face locations
    """
    detector = FaceDetector()
    
    # Load image
    print(f"Loading image: {image_path}")
    try:
        image = load_image(image_path)
    except Exception as e:
        print(f"ERROR: Could not load image: {e}")
        return None
    
    # Detect faces
    print("Detecting faces...")
    face_locations = detector.detect_faces(image)
    
    # Display results
    print(f"\nFound {len(face_locations)} face(s):")
    for i, location in enumerate(face_locations, 1):
        print(f"  Face {i}:")
        print(f"    Location: top={location[0]}, right={location[1]}, "
              f"bottom={location[2]}, left={location[3]}")
    
    # Draw results on image
    display_image = convert_rgb_to_bgr(image)
    
    for location in face_locations:
        draw_face_box(display_image, location, BBOX_COLOR_FACE, BBOX_THICKNESS)
    
    # Save annotated image
    if output_path:
        save_image(display_image, output_path)
        print(f"\nAnnotated image saved to: {output_path}")
    
    # Show result
    if show_result:
        cv2.imshow('Face Recognition Result', display_image)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return face_locations


def detect_batch(input_dir, output_dir=None):
    """
    Detect faces in all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save annotated images
    
    Returns:
        Dictionary mapping filenames to face locations
    """
    detector = FaceDetector()
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    if len(image_files) == 0:
        print(f"No image files found in {input_dir}")
        return None
    
    print(f"Processing {len(image_files)} images...")
    
    all_results = {}
    
    for i, filename in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {filename}...")
        
        input_path = os.path.join(input_dir, filename)
        
        try:
            # Load and process image
            image = load_image(input_path)
            face_locations = detector.detect_faces(image)

            all_results[filename] = face_locations

            print(f"  Found {len(face_locations)} face(s)")
            
            # Save annotated image if output directory specified
            if output_dir and len(face_locations) > 0:
                display_image = convert_rgb_to_bgr(image)

                for location in face_locations:
                    draw_face_box(display_image, location, BBOX_COLOR_FACE, BBOX_THICKNESS)
                
                output_path = os.path.join(output_dir, f"annotated_{filename}")
                save_image(display_image, output_path)
        
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[filename] = []
    
    return all_results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Face Detection for Static Images'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input image file or directory'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file or directory for annotated images'
    )
    parser.add_argument(
        '--show', '-s',
        action='store_true',
        help='Display result window (single image only)'
    )
    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Process all images in input directory'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FACE DETECTION SYSTEM - IMAGE MODE")
    print("=" * 60 + "\n")
    
    if args.batch or os.path.isdir(args.input):
        # Batch processing
        output_dir = args.output or os.path.join(OUTPUT_DIR, 'batch_results')
        results = detect_batch(args.input, output_dir)
        
        if results:
            print("\n" + "=" * 60)
            print("BATCH PROCESSING COMPLETE")
            print("=" * 60)
            print(f"Processed {len(results)} images")
            print(f"Results saved to: {output_dir}")
    else:
        # Single image processing
        output_path = args.output
        if not output_path:
            basename = os.path.basename(args.input)
            name, ext = os.path.splitext(basename)
            output_path = os.path.join(OUTPUT_DIR, f"detected_{name}{ext}")
        
        results = detect_image(args.input, output_path, args.show)
        
        if results is not None:
            print("\n" + "=" * 60)
            print("PROCESSING COMPLETE")
            print("=" * 60)


if __name__ == "__main__":
    main()
