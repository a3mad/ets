from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'TEST@1234567890'
    app.config['DEBUG_MODE'] = True
    from .routes import main
    app.register_blueprint(main)

    return app
