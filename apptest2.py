from pdf2image import convert_from_path
images = convert_from_path('temp.pdf', poppler_path=r'C:\poppler-24.08.0\Library\bin')
for i, image in enumerate(images):
    image.save(f'page_{i}.png', 'PNG')