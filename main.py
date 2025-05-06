import argparse
import os
from preprocess import preprocess_image
from generate_3d import image_to_3d, text_to_3d
from visualize import visualize_3d_model

def main():
    parser = argparse.ArgumentParser(description="Convert photo or text to 3D model")
    parser.add_argument("--input", required=True, help="Path to input image or text file")
    parser.add_argument("--output", default="output/model.obj", help="Output 3D model path (.obj/.stl)")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Check if input is image or text
    if args.input.lower().endswith(('.jpg', '.png')):
        # Process image
        processed_image = preprocess_image(args.input)
        model = image_to_3d(processed_image)
    elif args.input.lower().endswith('.txt'):
        # Process text
        with open(args.input, 'r') as f:
            prompt = f.read().strip()
        model = text_to_3d(prompt)
    else:
        raise ValueError("Input must be a .jpg/.png image or .txt file")

    # Save 3D model
    model.export(args.output)
    print(f"3D model saved to {args.output}")

    # Visualize
    visualize_3d_model(args.output)

if __name__ == "__main__":
    main()
