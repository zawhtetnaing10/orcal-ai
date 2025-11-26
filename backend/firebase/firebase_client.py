import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("private_keys/service_account_key.json")
firebase_app = firebase_admin.initialize_app(cred)

firestore_client = firestore.client(firebase_app)
