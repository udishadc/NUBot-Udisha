
from flask import Flask
from flask_restx import Api, Resource # type: ignore
from src.logger import logging


app =Flask(__name__)

# Initialize Flask-RESTX API with Swagger support
api =Api(app, version="1.0" , title="NuBot Backend", description="Backend for NuBot")

# create a namespace 
ns=api.namespace("NuBot",descrption="namespace for Backend")

@ns.route("/")
class Main(Resource):
    def get(self):
        """Return a simple message"""
        logging.info("Get api called")
        return {'message': 'Hello, World!'}
    

if __name__=="__main__":
    app.run(debug=True)