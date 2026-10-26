from flask import Flask
try:
    from flask_cors import CORS
except ImportError:
    CORS = lambda app: None

from flask_sqlalchemy import SQLAlchemy
import os
import logging

# Initialize SQLAlchemy
db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Basic settings
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key')
    app.config['JSON_SORT_KEYS'] = False
    
    # ✅ ADDED: Database configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # ✅ ADDED: Initialize database with app
    db.init_app(app)

    # Register all blueprints (routes)
    try:
        from app.routes.precare import precare_bp
        app.register_blueprint(precare_bp)
    except Exception as e:
        app.logger.error(f"Failed to import precare blueprint: {e}")

    return app
