from PIL import Image

# Load images
original = Image.open('girl-2.png').convert('RGBA')
original_size = original.size
mask = Image.open('mask.png').convert('L')  # grayscale mask
mask = mask.resize(original_size)

def double_and_clip(p):
    return (p-128)*4

alpha = mask.point(double_and_clip)

r_o, g_o, b_o, _ = original.split()
result = Image.merge('RGBA', (r_o, g_o, b_o, alpha))

result.save('result_with_doubled_red_alpha.png')
