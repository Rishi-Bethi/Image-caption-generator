from flask import Flask, render_template, request, url_for
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def generate_captions(image_file, num_captions, max_length=1024):
    # Load the image
    image = Image.open(image_file).convert('RGB')

    # Preprocess the image
    inputs = processor(image, padding=True, max_length=max_length, return_tensors="pt")

    # Generate the captions
    outputs = model.generate(**inputs, num_return_sequences=num_captions, max_length=max_length, do_sample=True)

    # Decode the captions
    captions = []
    for output in outputs:
        caption = processor.decode(output, skip_special_tokens=True)
        captions.append(caption)

    return captions


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    num_captions = int(request.form.get('num_captions'))
    file = request.files['file']
    image_path = 'static/' + file.filename
    file.save(image_path)
    captions = generate_captions(image_path, num_captions=num_captions)
    image_url = url_for('static', filename=file.filename)
    return render_template('result.html', image_url=image_url, captions=captions)


if __name__ == '__main__':
    app.run(debug=True)
