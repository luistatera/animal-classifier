from flask import Flask, request, render_template
from model_loader import predict_label, get_available_models
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    available_models = get_available_models()
    
    if request.method == "POST":
        file = request.files["file"]
        model_name = request.form.get("model", "best_cnn_model.keras")
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        label, confidence = predict_label(filepath, model_name)
        return render_template("index.html", 
                             label=label, 
                             confidence=confidence, 
                             image=filepath,
                             available_models=available_models,
                             selected_model=model_name)
    
    return render_template("index.html", 
                         available_models=available_models,
                         selected_model="best_cnn_model.keras")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

