import base64
import contextlib
import io
import os
import socket

import dash
import numpy as np
import tensorflow as tf
from dash import Input, Output, State, dcc, html
from PIL import Image

from convolution_patterns.config.config import Config
from convolution_patterns.config.transform_config import TransformConfig
from convolution_patterns.services.image_dataset_service import ImageDatasetService
from convolution_patterns.services.transform_service import TransformService

print(dash.__version__)
# 🔧 Setup
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# 📁 Constants
dataset_modes = ["Raw", "Transformed", "Augmented"]
splits = ["train", "val", "test"]
config = Config()
transform_config = TransformConfig.from_yaml(config.transform_config_path)


# 🧠 Utilities
def image_to_base64(image):
    image = np.array(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image)
        min_val, max_val = np.min(image), np.max(image)
        if max_val - min_val > 0:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image.squeeze(-1)
    try:
        img_pil = Image.fromarray(image)
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print("❌ image_to_base64 failed:", e)
        return ""


def load_dataset(split, mode):
    ds, class_names = ImageDatasetService().get_dataset(split=split, print_stats=False)
    if mode == "Raw":
        return ds, class_names
    transform_fn = TransformService(transform_config).get_pipeline(
        mode="train" if mode == "Augmented" else "val"
    )
    return ds.map(lambda x, y: (transform_fn(x), y)), class_names


# 🖼️ Layout
app.layout = html.Div(
    [
        html.H2("🧪 Convolution CV Debug Viewer"),
        html.Div(
            [
                html.Label("Split"),
                dcc.Dropdown(splits, value="train", id="split-dropdown"),
                html.Label("Mode"),
                dcc.Dropdown(dataset_modes, value="Raw", id="mode-dropdown"),
                html.Label("Batch Index"),
                dcc.Input(id="batch-index", type="number", min=0, value=0, step=1),
                html.Label("Filter by Class"),
                dcc.Dropdown(
                    [],
                    value=None,
                    id="class-filter",
                    placeholder="(optional)",
                    clearable=True,
                ),
                html.Button("Reload Config", id="reload-button", n_clicks=0),
            ],
            style={"width": "25%", "display": "inline-block", "verticalAlign": "top"},
        ),
        html.Div(
            id="image-grid",
            style={"display": "flex", "flexWrap": "wrap", "gap": "10px"},
        ),
        dcc.Store(id="current-classnames-store"),
        html.Div(
            id="modal-container",
            style={"display": "none"},
            children=[
                html.Div(
                    id="modal",
                    style={
                        "position": "fixed",
                        "top": 0,
                        "left": 0,
                        "width": "100%",
                        "height": "100%",
                        "backgroundColor": "rgba(0,0,0,0.8)",
                        "zIndex": 9999,
                        "display": "flex",
                        "justifyContent": "center",
                        "alignItems": "center",
                    },
                    children=[
                        html.Div(
                            [
                                html.Img(
                                    id="modal-img",
                                    style={"maxHeight": "60vh", "maxWidth": "60vw"},
                                ),
                                html.Pre(
                                    id="modal-info",
                                    style={
                                        "color": "white",
                                        "fontSize": "14px",
                                        "marginTop": "10px",
                                    },
                                ),
                                html.Button(
                                    "Close",
                                    id="close-modal",
                                    style={"marginTop": "10px"},
                                ),
                            ],
                            style={"textAlign": "center"},
                        )
                    ],
                )
            ],
        ),
    ]
)


# 🔄 Load class names
@app.callback(
    Output("current-classnames-store", "data"),
    Output("class-filter", "options"),
    Input("mode-dropdown", "value"),
    Input("split-dropdown", "value"),
    Input("reload-button", "n_clicks"),
)
def reload_config_and_get_classes(mode, split, _):
    global transform_config
    transform_config = TransformConfig.from_yaml(config.transform_config_path)
    _, class_names = load_dataset(split, mode)
    options = [{"label": name, "value": i} for i, name in enumerate(class_names)]
    return class_names, options


# 🖼️ Display batch
@app.callback(
    Output("image-grid", "children"),
    Input("batch-index", "value"),
    Input("mode-dropdown", "value"),
    Input("split-dropdown", "value"),
    Input("current-classnames-store", "data"),
    Input("class-filter", "value"),
)
def update_image_grid(batch_idx, mode, split, class_names, class_filter):
    if class_names is None:
        return [html.P("⚠️ No dataset loaded.")]

    ds, _ = load_dataset(split, mode)
    ds_batch = ds.skip(batch_idx).take(1)

    try:
        images, labels = next(iter(ds_batch))
        images = tf.convert_to_tensor(images).numpy()
        labels = np.argmax(tf.convert_to_tensor(labels).numpy(), axis=-1)
    except Exception as e:
        return [html.P(f"⚠️ Failed to load batch {batch_idx}: {str(e)}")]

    children = []
    for i, (img, label_idx) in enumerate(zip(images, labels)):
        if class_filter is not None and label_idx != class_filter:
            continue
        img_id = f"Image {i:02d}"
        encoded = image_to_base64(img)
        label_text = class_names[label_idx]
        min_val, max_val = float(np.min(img)), float(np.max(img))
        shape = img.shape

        tooltip = f"{img_id} – {label_text}\nShape: {shape}\nMin: {min_val:.2f}, Max: {max_val:.2f}"
        children.append(
            html.Div(
                [
                    html.Img(
                        src=f"data:image/png;base64,{encoded}",
                        title=tooltip,
                        id={"type": "thumbnail", "index": i},
                        style={"height": "150px", "cursor": "pointer"},
                    ),
                    html.P(f"{img_id} – {label_text}", style={"fontSize": "12px"}),
                ],
                style={"textAlign": "center", "width": "180px"},
            )
        )
    return children


# 🧲 Open zoom modal + info
@app.callback(
    Output("modal-container", "style"),
    Output("modal-img", "src"),
    Output("modal-info", "children"),
    Input({"type": "thumbnail", "index": dash.ALL}, "n_clicks"),
    State("batch-index", "value"),
    State("mode-dropdown", "value"),
    State("split-dropdown", "value"),
    State("current-classnames-store", "data"),
    prevent_initial_call=True,
)
def show_modal(thumbnail_clicks, batch_idx, mode, split, class_names):
    triggered_idx = next((i for i, v in enumerate(thumbnail_clicks) if v), None)
    if triggered_idx is None:
        return {"display": "none"}, "", ""

    ds, _ = load_dataset(split, mode)
    ds_batch = ds.skip(batch_idx).take(1)
    images, labels = next(iter(ds_batch))
    image = tf.convert_to_tensor(images[triggered_idx]).numpy()
    label_idx = int(np.argmax(labels[triggered_idx].numpy()))
    label_name = class_names[label_idx]

    encoded = image_to_base64(image)
    info = f"""📷 {triggered_idx:02d}
Label: {label_name}
Class Index: {label_idx}
Shape: {image.shape}
Min: {np.min(image):.2f}, Max: {np.max(image):.2f}"""

    return {"display": "block"}, f"data:image/png;base64,{encoded}", info


# ❌ Close modal
@app.callback(
    Output("modal-container", "style", allow_duplicate=True),
    Input("close-modal", "n_clicks"),
    prevent_initial_call=True,
)
def hide_modal(_):
    return {"display": "none"}


def find_free_port(start=8050, end=9000):
    for port in range(start, end):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("❌ No free port found in range.")


# 🚀 Launch
if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8050  # ===find_free_port()
    ip = socket.gethostbyname(socket.gethostname())

    print("\n🚀 You can now view your Dash app in your browser.\n")
    print(f"  Local URL:   http://localhost:{port}")
    print(f"  Network URL: http://{ip}:{port}\n")

    app.run(debug=True, host=host, port=port)
