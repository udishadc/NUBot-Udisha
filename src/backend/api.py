
from flask import Flask
from flask_restx import Api, Resource,fields # type: ignore
from backend.logger import logging
from dotenv import load_dotenv
from flask_restx import reqparse


from backend.exception import CustomException
import sys

from src.model.rag_model import generate_response
load_dotenv()


app =Flask(__name__)

# Initialize Flask-RESTX API with Swagger support
api =Api(app, version="1.0" , title="NuBot Backend", description="Backend for NuBot")

# create a namespace 
ns=api.namespace("NuBot",descrption="namespace for Backend")
parser = reqparse.RequestParser()
# Define a request model for Swagger UI
query_model = api.model('QueryModel', {
    'query': fields.String(required=True, description="User's input query")
})
@ns.route("/")
class Main(Resource):
    @api.expect(query_model)
    def post(self):
        """Return a simple message"""
        try:
            parser.add_argument('query', required=True, help="Name cannot be blank!")
            args=parser.parse_args()
            logging.info("Get api called")
            query=args['query']
            response=generate_response(query)
            return response
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    app.run(debug=True)