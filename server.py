from waitress import serve

from traceability_api_backend.wsgi import application

if __name__ == '__main__':
    serve(application, port='8000')
