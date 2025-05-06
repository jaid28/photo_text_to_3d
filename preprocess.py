from rembg import remove
import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess image: remove background and prepare for 3D conversion.
    Returns: Processed image (numpy array).
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Try to remove background using rembg
    try:
        image_no_bg = remove(image)
        # Convert to grayscale and check if rembg produced a valid result
        gray_no_bg = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY)
        if np.mean(gray_no_bg) > 10:  # Threshold to detect if the image is mostly black
            print("Background removed successfully with rembg.")
            return gray_no_bg
        else:
            print("rembg failed to remove background; trying OpenCV fallback.")
    except Exception as e:
        print(f"rembg failed with error: {e}. Trying OpenCV fallback.")

    # Fallback: Use OpenCV for background removal
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to create a binary mask
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours to isolate the main object
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour (assumed to be the object)
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Apply the mask to the grayscale image
        result = cv2.bitwise_and(gray, gray, mask=mask)
        return result

    # If all else fails, return the grayscale image
    print("OpenCV fallback failed; using original grayscale image.")
    return gray
