from flask import Flask, request, render_template, session
from model_loader import predict_label, get_available_models, load_model
import os
import logging
from io import BytesIO
import base64
from google.cloud import secretmanager

def get_secret(secret_id):
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/chiaraguru/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        app.logger.error(f"Error fetching secret: {e}")
        return None

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

# Get the secret key from GCP Secret Manager
secret_key = get_secret('FLASK_SECRET_KEY')
if secret_key:
    app.secret_key = secret_key
else:
    app.logger.warning("Using fallback secret key - not recommended for production")
    app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-123')

# Model display name mapping
MODEL_DISPLAY_NAMES = {
    "best_cnn_model_luis.pth": "Simple CNN Model",
    "best_resnet_model.pth": "ResNet 50 Model"
}

@app.after_request
def add_security_headers(response):
    """Add security headers to each response"""
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline';"
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    # Content type headers
    if not response.headers.get('Content-Type'):
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
    
    # Cache control headers
    if request.path.startswith('/static/'):
        # Cache static resources for 1 year with cache busting
        response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    else:
        # No caching for dynamic content
        response.headers['Cache-Control'] = 'no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    
    return response

def get_image_data_url(file):
    """Convert image file to base64 data URL"""
    file_content = file.read()
    file.seek(0)  # Reset file pointer for subsequent reads
    encoded = base64.b64encode(file_content).decode('utf-8')
    mime_type = file.content_type or 'application/octet-stream'
    return f"data:{mime_type};base64,{encoded}"

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

# Do not change this
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)


