from flask import Flask, request, render_template, session
from model_loader import predict_label, get_available_models, load_model
import os
import logging
from io import BytesIO
import base64

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
app.secret_key = 'your-secret-key-123'  # Simple secret key

# Model display name mapping
MODEL_DISPLAY_NAMES = {
    "best_cnn_model_luis.pth": "Simple CNN Model",
    "best_resnet_model.pth": "ResNet 50 Model"
}

@app.after_request
def add_security_headers(response):
    """Add security headers to each response"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    
    # Cache control for static resources
    if request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    else:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    
    return response

def get_image_data_url(file):
    """Convert image file to base64 data URL"""
    file_content = file.read()
    file.seek(0)  # Reset file pointer for subsequent reads
    encoded = base64.b64encode(file_content).decode('utf-8')
    return f"data:{file.content_type};base64,{encoded}"

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            if 'file' in request.files and request.files['file'].filename != '':
                file = request.files["file"]
                
                # Read the file content once and store it
                file_content = file.read()
                file.seek(0)  # Reset for data URL creation
                
                # Convert image to data URL and store in session
                image_data_url = get_image_data_url(file)
                session['current_image'] = image_data_url
                
                try:
                    model_results = {}
                    available_models = get_available_models()
                    
                    # Create a new BytesIO object for each model prediction
                    for model_name in available_models:
                        model, model_type = load_model(model_name)
                        # Create a fresh BytesIO object for each prediction
                        img_bytes = BytesIO(file_content)
                        label, confidence = predict_label(model, img_bytes, model_type)
                        
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
                                        image=image_data_url,
                                        model_results=model_results)
                except Exception as e:
                    app.logger.error(f"Prediction error: {str(e)}")
                    return render_template("index.html", 
                                        error=f"Error during prediction: {str(e)}")
            
            return render_template("index.html", error="Please upload an image first")
        
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
                
                # Read the file content once and store it
                file_content = file.read()
                file.seek(0)  # Reset for data URL creation
                
                # Convert image to data URL and store in session
                image_data_url = get_image_data_url(file)
                session['current_image'] = image_data_url
                
                try:
                    model_results = {}
                    available_models = get_available_models()
                    
                    # Create a new BytesIO object for each model prediction
                    for model_name in available_models:
                        model, model_type = load_model(model_name)
                        # Create a fresh BytesIO object for each prediction
                        img_bytes = BytesIO(file_content)
                        label, confidence = predict_label(model, img_bytes, model_type)
                        
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
                                        image=image_data_url,
                                        model_results=model_results)
                except Exception as e:
                    app.logger.error(f"Prediction error: {str(e)}")
                    return render_template("project.html", 
                                        error=f"Error during prediction: {str(e)}")
            
            return render_template("project.html", error="Please upload an image first")
        
        session.clear()
        return render_template("project.html")
    
    except Exception as e:
        app.logger.error(f"Application error: {str(e)}")
        return render_template("project.html", error=f"Application error: {str(e)}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)


