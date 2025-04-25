from flask import Flask, request, render_template, session
from model_loader import predict_label, get_available_models, load_model
import os
import logging

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
# Set a secret key for session management
app.secret_key = 'your-secret-key-123'  # Change this to a secure secret key in production

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model display name mapping
MODEL_DISPLAY_NAMES = {
    "best_cnn_model_luis.pth": "Simple CNN Model",
    "best_resnet_model.pth": "ResNet 50 Model"
}

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            if 'file' in request.files and request.files['file'].filename != '':
                file = request.files["file"]
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                session['current_image'] = filepath
            
            filepath = session.get('current_image')
            
            if not filepath:
                return render_template("index.html", error="Please upload an image first")
            
            try:
                model_results = {}
                available_models = get_available_models()
                
                for model_name in available_models:
                    model, model_type = load_model(model_name)
                    label, confidence = predict_label(model, filepath, model_type)
                    
                    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
                    
                    if label is None:
                        model_results[display_name] = {
                            "warning": f"Image not recognized as any of the 10 animals (confidence: {confidence * 100:.2f}%)"
                        }
                    else:
                        model_results[display_name] = {
                            "label": label,
                            "confidence": confidence
                        }
                
                return render_template("index.html", 
                                    image=filepath,
                                    model_results=model_results)
            except Exception as e:
                app.logger.error(f"Prediction error: {str(e)}")
                return render_template("index.html", 
                                    error=f"Error during prediction: {str(e)}")
        
        session.clear()
        return render_template("index.html")
    
    except Exception as e:
        app.logger.error(f"Application error: {str(e)}")
        return render_template("index.html", error=f"Application error: {str(e)}")

@app.route("/project", methods=["GET", "POST"])
def project():
    try:
        if request.method == "POST":
            if 'file' in request.files and request.files['file'].filename != '':
                file = request.files["file"]
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                session['current_image'] = filepath
            
            filepath = session.get('current_image')
            
            if not filepath:
                return render_template("project.html", error="Please upload an image first")
            
            try:
                model_results = {}
                available_models = get_available_models()
                
                for model_name in available_models:
                    model, model_type = load_model(model_name)
                    label, confidence = predict_label(model, filepath, model_type)
                    
                    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
                    
                    if label is None:
                        model_results[display_name] = {
                            "warning": f"Image not recognized as any of the 10 animals (confidence: {confidence * 100:.2f}%)"
                        }
                    else:
                        model_results[display_name] = {
                            "label": label,
                            "confidence": confidence
                        }
                
                return render_template("project.html", 
                                    image=filepath,
                                    model_results=model_results)
            except Exception as e:
                app.logger.error(f"Prediction error: {str(e)}")
                return render_template("project.html", 
                                    error=f"Error during prediction: {str(e)}")
        
        session.clear()
        return render_template("project.html")
    
    except Exception as e:
        app.logger.error(f"Application error: {str(e)}")
        return render_template("project.html", error=f"Application error: {str(e)}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)


