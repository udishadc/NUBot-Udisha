
from flask import Flask
from flask_restx import Api, Resource # type: ignore
from backend.logger import logging
from dotenv import load_dotenv
from backend.exception import CustomException
import sys
load_dotenv()


app =Flask(__name__)

# Initialize Flask-RESTX API with Swagger support
api =Api(app, version="1.0" , title="NuBot Backend", description="Backend for NuBot")

# create a namespace 
ns=api.namespace("NuBot",descrption="namespace for Backend")

@ns.route("/")
class Main(Resource):
    def get(self):
        """Return a simple message"""
        try:
            logging.info("Get api called")
            return {'message': 'Hello, World!'}
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    app.run(debug=True)