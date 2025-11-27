import firebase_admin
from firebase_admin import credentials, firestore_async, firestore

cred = credentials.Certificate("private_keys/service_account_key.json")
firebase_app = firebase_admin.initialize_app(cred)

firestore_async = firestore_async.client(firebase_app)
