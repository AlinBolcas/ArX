from PIL import Image
import os
from pathlib import Path

def create_mood_board(directory, output_file, board_size, img_size):
    images = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Create a new image for the mood board
    mood_board = Image.new('RGB', board_size, (255, 255, 255))  # white background

    x_offset = 0
    y_offset = 0
    for img_path in images:
        # Open the image and resize it
        img = Image.open(img_path)
        img = img.resize(img_size)

        # Paste the image into the mood board
        mood_board.paste(img, (x_offset, y_offset))

        # Update the x_offset and y_offset for the next image
        x_offset += img_size[0]
        if x_offset >= mood_board.width:
            x_offset = 0
            y_offset += img_size[1]

    # Save the mood board
    mood_board.save(output_file)

# Example usage
output_dir = Path(__file__).parent / 'output'
create_mood_board(str(output_dir), 'mood_board.png', (512, 512), (256, 256))


# graph network + particle system = attaching text & images to it - make it 3dx