from flask import Flask, request, render_template, session, url_for
from model_loader import predict_label, get_available_models, load_model
import os
import logging
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

# Get the secret key from environment variable
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-123')

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model display name mapping
MODEL_DISPLAY_NAMES = {
    "best_cnn_model_luis.pth": "Simple CNN Model",
    "best_resnet_model.pth": "ResNet 50 Model"
}

def save_uploaded_file(file):
    """Save uploaded file and return its path"""
    # Generate a unique filename
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    # Save the file
    file.save(file_path)
    
    # Return the relative path for web access
    return os.path.join('uploads', unique_filename)

@app.after_request
def add_security_headers(response):
    """Add security headers to each response"""
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; font-src 'self' https://cdnjs.cloudflare.com;"
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
                
                # Save the file and get its path
                image_path = save_uploaded_file(file)
                full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(image_path))
                
                try:
                    model_results = {}
                    available_models = get_available_models()
                    
                    # Use the saved file path for prediction
                    for model_name in available_models:
                        model, model_type = load_model(model_name)
                        label, confidence = predict_label(model, full_image_path, model_type)
                        
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
                            app.logger.debug(f"Model: {display_name}, Label: {label}, Confidence: {confidence}")
                    
                    return render_template("index.html", 
                                        image=image_path,
                                        model_results=model_results)
                except Exception as e:
                    app.logger.error(f"Prediction error: {str(e)}")
                    return render_template("index.html", 
                                        error=f"Error during prediction: {str(e)}")
            
            return render_template("index.html", error="Please upload an image first")
        
        return render_template("index.html")
    
    except Exception as e:
        app.logger.error(f"Application error: {str(e)}")
        return render_template("index.html", error=f"Application error: {str(e)}")

@app.route("/project", methods=["GET", "POST"])
def project():
    try:
        if request.method == "POST":
            app.logger.debug("POST request received")
            app.logger.debug(f"Files in request: {request.files}")
            
            if 'file' not in request.files:
                app.logger.error("No file part in the request")
                return render_template("project.html", error="No file part in the request")
                
            file = request.files['file']
            app.logger.debug(f"File received: {file.filename}")
            
            if file.filename == '':
                app.logger.error("No selected file")
                return render_template("project.html", error="No selected file")
            
            if not file:
                app.logger.error("File is None")
                return render_template("project.html", error="File is None")
                
            # Save the file and get its path
            image_path = save_uploaded_file(file)
            full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(image_path))
            
            try:
                model_results = {}
                available_models = get_available_models()
                
                # Use the saved file path for prediction
                for model_name in available_models:
                    model, model_type = load_model(model_name)
                    label, confidence = predict_label(model, full_image_path, model_type)
                    
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
                        app.logger.debug(f"Model: {display_name}, Label: {label}, Confidence: {confidence}")
                
                return render_template("project.html", 
                                    image=image_path,
                                    model_results=model_results)
            except Exception as e:
                app.logger.error(f"Prediction error: {str(e)}")
                return render_template("project.html", 
                                    error=f"Error during prediction: {str(e)}")
        
        return render_template("project.html")
    
    except Exception as e:
        app.logger.error(f"Application error: {str(e)}")
        return render_template("project.html", error=f"Application error: {str(e)}")

# Do not change this
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)


