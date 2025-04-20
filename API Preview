import gradio as gr
from keras.models import load_model
from PIL import Image
import numpy as np
import keras.backend as K
from tensorflow.keras.layers import Layer

# Define CapsuleLayer class with a default value for dim_capsules
class CapsuleLayer(Layer):
    def __init__(self, num_capsules, dim_capsules=16, routings=3, **kwargs):
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules  # Default value for dim_capsules
        self.routings = routings
        super(CapsuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weight initialization
        self.W = self.add_weight(name="capsule_weight",
                                 shape=(input_shape[1], self.num_capsules, self.dim_capsules),
                                 initializer="glorot_uniform",
                                 trainable=True)
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs):
        # Capsule forward pass (simplified)
        u_hat = K.batch_dot(inputs, self.W, [2, 1])  # Apply the transformation matrix
        return u_hat

# Load the Capsule Network model
try:
    model = load_model("capsule_model.h5", custom_objects={"CapsuleLayer": CapsuleLayer}, compile=False)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Define class labels
class_labels = [
    'Melanocytic nevi',
    'Melanoma',
    'Benign keratosis-like lesions',
    'Basal cell carcinoma',
    'Actinic keratoses',
    'Vascular lesions',
    'Dermatofibroma'
]

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the correct model's expected input size
    image = np.array(image)  # Convert image to numpy array
    image = image.astype('float32')  # Convert to float32
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the image
    return image

# Prediction function
def predict_skin_cancer(image):
    # Preprocess the image
    image = preprocess_image(image)

    # Predict the class probabilities
    predictions = model.predict(image)

    # Get the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)

    # Get the corresponding label
    predicted_label = class_labels[predicted_class[0]]

    # Get the confidence score (probability of the predicted class)
    confidence_score = np.max(predictions) * 100

    return predicted_label, confidence_score

# Custom CSS for purple theme and title box with frame
css = """
body {
    background-color: #7A4E96;
    color: white;
    font-family: Arial, sans-serif;
}

h1 {
    font-size: 36px;
    text-align: center;
    padding: 20px;
    background-color: #5d3377;
    border-radius: 10px;
    border: 2px solid #7A4E96;
}

h3 {
    font-size: 28px;
}

.gradio-container {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 20px;
}



button {
    background-color: #7A4E96;
    color: white;
    font-size: 18px;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
}

button:hover {
    background-color: #5d3377;
}


"""

# Create Gradio interface
interface = gr.Interface(
    fn=predict_skin_cancer,
    inputs=gr.Image(type="pil", label="Upload a Dermoscopic Image", interactive=True),
    outputs=[
        gr.Textbox(label="Predicted Skin Cancer Type"),
        gr.Textbox(label="Confidence Score (%)")
    ],
    title="Skin Cancer Detection System",
    description="This system uses a Capsule Network to detect 7 types of skin cancer from dermoscopic images.",
    theme="default",  # Use the default theme
    allow_flagging="never",  # Optional: hide flagging
    css=css  # Apply custom CSS
)

# Launch the interface
interface.launch(share=True)
