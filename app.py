from flask import Flask, request, render_template, session
from model_loader import predict_label, get_available_models
import os
import logging

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
# Set a secret key for session management
app.secret_key = 'your-secret-key-123'  # Change this to a secure secret key in production

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        available_models = get_available_models()
        
        if request.method == "POST":
            model_name = request.form.get("model", "best_cnn_model.keras")
            
            # Handle new image upload
            if 'file' in request.files and request.files['file'].filename != '':
                file = request.files["file"]
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                session['current_image'] = filepath
            
            # Use existing image if available
            filepath = session.get('current_image')
            
            if not filepath:
                return render_template("index.html", 
                                    error="Please upload an image first",
                                    available_models=available_models,
                                    selected_model=model_name)
            
            try:
                label, confidence = predict_label(filepath, model_name)
                if label is None:
                    return render_template("index.html",
                                        warning=f"Image not recognized as any of the 10 animals (confidence: {confidence * 100:.2f}%)",
                                        image=filepath,
                                        available_models=available_models,
                                        selected_model=model_name)
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
                                    available_models=available_models,
                                    selected_model=model_name)
        
        # Clear the session on fresh page load (GET request)
        session.clear()
        return render_template("index.html", 
                            available_models=available_models,
                            selected_model="best_cnn_model.keras")
    
    except Exception as e:
        app.logger.error(f"Application error: {str(e)}")
        return render_template("index.html", error=f"Application error: {str(e)}")

@app.route("/project", methods=["GET", "POST"])
def project():
    try:
        project_models = ["prj_best_cnn_model.keras", "prj_best_resnet_model.keras"]
        
        if request.method == "POST":
            model_name = request.form.get("model", "prj_best_cnn_model.keras")
            
            # Handle new image upload
            if 'file' in request.files and request.files['file'].filename != '':
                file = request.files["file"]
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                session['current_image'] = filepath
            
            # Use existing image if available
            filepath = session.get('current_image')
            
            if not filepath:
                return render_template("project.html", 
                                    error="Please upload an image first",
                                    selected_model=model_name)
            
            try:
                label, confidence = predict_label(filepath, model_name)
                if label is None:
                    return render_template("project.html",
                                        warning=f"Image not recognized as any of the 10 animals (confidence: {confidence * 100:.2f}%)",
                                        image=filepath,
                                        selected_model=model_name)
                return render_template("project.html", 
                                    label=label, 
                                    confidence=confidence, 
                                    image=filepath,
                                    selected_model=model_name)
            except Exception as e:
                app.logger.error(f"Prediction error: {str(e)}")
                return render_template("project.html", 
                                    error=f"Error during prediction: {str(e)}",
                                    selected_model=model_name)
        
        # Clear the session on fresh page load (GET request)
        session.clear()
        return render_template("project.html", 
                            selected_model="prj_best_cnn_model.keras")
    
    except Exception as e:
        app.logger.error(f"Application error: {str(e)}")
        return render_template("project.html", error=f"Application error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

