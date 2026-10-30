import shutil
import sys
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageFilter
import numpy as np
import subprocess
import os
import uuid

# Serve the built Vite frontend (dist/) from the same Flask app so the
# whole thing deploys as a single service. Falls back gracefully when
# dist/ doesn't exist (e.g. local dev with `vite` running separately).
_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dist')
app = Flask(__name__, static_folder=_DIST, static_url_path='')
CORS(app)

@app.route('/')
def index():
    index_file = os.path.join(_DIST, 'index.html')
    if os.path.exists(index_file):
        return app.send_static_file('index.html')
    return 'Frontend not built. Run `pnpm build` first.', 404

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_MODELS = ['u2net', 'u2netp', 'u2net_human_seg']
ACCEPTED_EXTS = ['.jpg', '.jpeg', '.png', '.webp']  # what the API accepts
# backgroundremover CLI only supports these input formats — anything else
# (e.g. webp) is converted to PNG with PIL before processing
CLI_EXTS = ['.jpg', '.jpeg', '.png', '.heic', '.heif']
MAX_DIM = 1200  # keep payload reasonable


def find_backgroundremover():
    """Locate the backgroundremover CLI, trying PATH first, then the
    interpreter's own site-packages (handles the case where the server was
    launched from an environment with a different PATH than pip used)."""
    path = shutil.which('backgroundremover')
    if path:
        return [path]
    # Fallback: run as a module with the same interpreter running this server
    return [sys.executable, '-m', 'backgroundremover']


def clean_alpha(png_path):
    """Post-process the cutout: the u2net models leave a soft gray halo
    around the subject and often erode thin structures (bridges, temples).
    1) Morphological closing on alpha reconnects thin parts without growing
       the silhouette.
    2) Push low alpha values to fully transparent and high ones to fully
       opaque, removing the halo.
    3) Crop to the content bounding box."""
    img = Image.open(png_path).convert('RGBA')
    alpha_img = img.getchannel('A')

    # Closing (dilate via MaxFilter, erode via MinFilter) reconnects the
    # thin frame parts the AI segmentation tends to erode
    k = 5
    closed = alpha_img.filter(ImageFilter.MaxFilter(k))
    closed = closed.filter(ImageFilter.MinFilter(k))

    # Combine: take the max of the original and the closed mask so we only
    # ever ADD coverage (recovered thin parts), never enlarge the silhouette
    a_raw = np.array(alpha_img, dtype=np.float32)
    a_closed = np.array(closed, dtype=np.float32)
    alpha = np.maximum(a_raw, a_closed)

    # Halo removal: map alpha [0..255] -> 0 below 40, ramp 40..160, opaque above
    alpha = np.clip((alpha - 40.0) * (255.0 / (160.0 - 40.0)), 0, 255)

    a = np.array(img)
    a[:, :, 3] = alpha.astype(np.uint8)
    img = Image.fromarray(a)

    # Crop to the bounding box of visible content (removes empty padding
    # so the glasses can be scaled to the user's face precisely)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def downscale(img):
    if max(img.size) > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
    return img


def to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'data:image/png;base64,{b64}'


@app.route('/remove-background', methods=['POST'])
def remove_background():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Tweak options: which segmentation model and whether to run alpha matting
    model = request.form.get('model', 'u2netp')
    if model not in ALLOWED_MODELS:
        model = 'u2netp'
    matting = request.form.get('matting') == '1'

    file_id = str(uuid.uuid4())
    input_ext = os.path.splitext(file.filename)[1].lower() or '.png'
    if input_ext not in ACCEPTED_EXTS:
        input_ext = '.png'

    input_path = os.path.join(UPLOAD_FOLDER, f'{file_id}_input{input_ext}')
    output_path = os.path.join(UPLOAD_FOLDER, f'{file_id}_output.png')

    try:
        file.save(input_path)

        # The CLI rejects webp — convert unsupported formats to PNG first
        cli_input = input_path
        if input_ext not in CLI_EXTS:
            png_path = os.path.join(UPLOAD_FOLDER, f'{file_id}_conv.png')
            with Image.open(input_path) as im:
                im.save(png_path, format='PNG')
            cli_input = png_path

        cmd = find_backgroundremover()
        args = ['-i', cli_input, '-o', output_path, '-m', model]
        if matting:
            args.append('-a')
        subprocess.run(cmd + args, check=True, capture_output=True)

        # The CLI can exit 0 and write an empty file on failure
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return jsonify({'error': 'Processing failed: the model produced no output. Try another model or image.'}), 500

        cutout = clean_alpha(output_path)
        original = downscale(Image.open(input_path).convert('RGBA'))

        # Return both so the client can offer a manual touch-up editor
        return jsonify({
            'image': to_data_url(cutout),
            'original': to_data_url(original),
        })
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else 'Unknown error'
        return jsonify({'error': f'Processing failed: {stderr}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        for p in (input_path, output_path,
                  os.path.join(UPLOAD_FOLDER, f'{file_id}_conv.png')):
            if os.path.exists(p):
                os.remove(p)


if __name__ == '__main__':
    # Render injects PORT; default to 5010 for local dev
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5010)), debug=False)
