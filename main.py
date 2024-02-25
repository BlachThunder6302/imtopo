import os
from PIL import Image as PILImage
import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.vision_models import ImageCaptioningModel, Image

# Set up paths
image_path = "path/to/your/image.jpg"
caption_model_path = "path/to/your/caption_model"
text_model_path = "path/to/your/text_model"

# Function to generate poem from image
def generate_poem_from_image(image_path):
    # Open the image
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    vertex_image = Image(image_bytes)

    # Generate captions from the image bytes
    caption_model = ImageCaptioningModel.from_pretrained(caption_model_path)
    captions = caption_model.get_captions(
        image=vertex_image,
        number_of_results=1,
        language="en"
    )

    # Generate a poem from the image captions
    text_model = TextGenerationModel.from_pretrained(text_model_path)
    poem = text_model.predict(
        prompt=f"Write a poem about the following image caption: {captions[0]}",
        temperature=1.0,
        max_output_tokens=256
    )

    return captions[0], poem.text

# Example usage
if __name__ == "__main__":
    caption, poem = generate_poem_from_image(image_path)
    print("Caption:", caption)
    print("Poem:", poem)
