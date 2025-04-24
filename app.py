from flask import Flask, request, render_template
from model_loader import predict_label, get_available_models
import os
import logging

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        available_models = get_available_models()
        
        if request.method == "POST":
            if 'file' not in request.files:
                return render_template("index.html", 
                                    error="No file uploaded",
                                    available_models=available_models)
            
            file = request.files["file"]
            if file.filename == '':
                return render_template("index.html", 
                                    error="No file selected",
                                    available_models=available_models)
            
            model_name = request.form.get("model", "best_cnn_model.keras")
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            try:
                label, confidence = predict_label(filepath, model_name)
                return render_template("index.html", 
                                    label=label, 
                                    confidence=confidence, 
                                    image=filepath,
                                    available_models=available_models,
                                    selected_model=model_name)
            except Exception as e:
                app.logger.error(f"Prediction error: {str(e)}")
                return render_template("index.html", 
                                    error=f"Error during prediction: {str(e)}",
                                    available_models=available_models)
        
        return render_template("index.html", 
                            available_models=available_models,
                            selected_model="best_cnn_model.keras")
    
    except Exception as e:
        app.logger.error(f"Application error: {str(e)}")
        return render_template("index.html", error=f"Application error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

