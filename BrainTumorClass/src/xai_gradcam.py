import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model

def build_gradcam_model(autoencoder, conv_layer_name='conv2d_2', bottleneck_layer_name='bottleneck'):
    """
    Extracts the feature maps and the bottleneck representations.
    For an Autoencoder + SVM pipeline, we use the bottleneck activations 
    as the target to compute gradients, simulating Grad-CAM for the extracted features.
    """
    grad_model = Model(
        inputs=autoencoder.input,
        outputs=[
            autoencoder.get_layer(conv_layer_name).output,
            autoencoder.get_layer(bottleneck_layer_name).output
        ]
    )
    return grad_model

def generate_heatmap(image_array, grad_model):
    """
    Produces a spatial heatmap indicating which regions of the image 
    caused the greatest activation in the bottleneck layer.
    """
    # Ensure image_array is a tensor and requires gradient (though it's the model weights that matter)
    img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, bottleneck = grad_model(img_tensor)
        
        # In a standard CNN, this is the class score. 
        # Here, it's the mean activation of the bottleneck features.
        target_activation = tf.reduce_mean(bottleneck)
        
    # Get gradients of the target activation with respect to the feature maps
    grads = tape.gradient(target_activation, conv_outputs)
    
    # Global average pooling of gradients (Weights for each feature map channel)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Squeeze batch dimension
    conv_outputs = conv_outputs[0]
    
    # Compute the weighted sum of feature maps
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Apply ReLU to keep only positive influences
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalize the heatmap between 0 and 1
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
        
    return heatmap.numpy()

def create_overlay(original_image_path, heatmap_array, img_size=128, alpha=0.5):
    """
    Applies the colormap to the heatmap and overlays it on the original image.
    """
    # Read and resize original image
    original_img = cv2.imread(original_image_path)
    if original_img is None:
        return None
    original_img = cv2.resize(original_img, (img_size, img_size))
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap_array, (img_size, img_size))
    
    # Convert to 8-bit color
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Overlay using alpha blending
    overlay = cv2.addWeighted(heatmap_color, alpha, original_img, 1 - alpha, 0)
    
    return overlay
