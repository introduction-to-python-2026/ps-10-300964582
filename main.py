import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball

# import functions
from image_utils import load_image, edge_detection

def main():
    # 1. Load the Image
    
    image_path = 'my_image.jpg' 
    original_image = load_image(image_path)
    
    
    if original_image is None:
        print("Error loading image")
        return

    print(f"Image loaded with shape: {original_image.shape}")

    # 2. Suppress Noise (Median Filter)
    
    clean_image = median(original_image, ball(3))

    # 3. Detect Edges
    
    edge_mag = edge_detection(clean_image)

    # 4. Thresholding (Binary Image)
    
    threshold_value = 100 
    binary_edges = edge_mag > threshold_value

    # 5. Save and Display
   
  
    edge_image_to_save = Image.fromarray((binary_edges * 255).astype(np.uint8))
    edge_image_to_save.save('my_edges.png')
    
    print("Process completed. Result saved as 'my_edges.png'")

  
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(edge_image_to_save, cmap='gray')
    plt.title("Edges Detected")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
