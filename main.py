import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve as convolve2D 

from skimage.filters import median
from skimage.morphology import disk   


image_path = "/content/barbie.jpg"   
img = load_image(image_path)       
print("Loaded:", img.shape, img.dtype)

plt.imshow(img)
plt.axis("off")
plt.title("Original")
plt.show()


gray = img.astype(np.float32).mean(axis=2)      
clean_image = median(gray, disk(3))             

plt.imshow(clean_image, cmap="gray")
plt.axis("off")
plt.title("Clean Image (Median Filter)")
plt.show()


clean_rgb = np.stack([clean_image]*3, axis=2).astype(np.float32)  
edgeMAG = edge_detection(clean_rgb)                               

plt.imshow(edgeMAG, cmap="gray")
plt.axis("off")
plt.title("Edge Magnitude")
plt.show()

plt.hist(edgeMAG.ravel(), bins=100)
plt.title("Histogram of edgeMAG")
plt.xlabel("Edge strength")
plt.ylabel("Count")
plt.show()

threshold = 50   
edge_binary = edgeMAG > threshold   

plt.imshow(edge_binary, cmap="gray")
plt.axis("off")
plt.title(f"Binary Edges (threshold={threshold})")
plt.show()

edge_uint8 = (edge_binary * 255).astype(np.uint8)
edge_image = Image.fromarray(edge_uint8)
edge_image.save("my_edges.png")

print("Saved: my_edges.png")
