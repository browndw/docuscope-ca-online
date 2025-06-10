# Copyright (C) 2025 David West Brown

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import pathlib
import sys

import streamlit as st
from datetime import datetime, timedelta
from google.cloud import firestore
from google.oauth2 import service_account

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from webapp.utilities.handlers import import_options_general  # noqa: E402

OPTIONS = str(project_root.joinpath("webapp/options.toml"))

# import options
_options = import_options_general(OPTIONS)
DESKTOP = _options['global']['desktop_mode']

if DESKTOP is False:
    # Set up the Google Cloud Firestore credentials
    try:
        key_dict = st.secrets["firestore"]["key_dict"]
        creds = service_account.Credentials.from_service_account_info(key_dict)
    except FileNotFoundError:
        creds = None


# Functions for handling states and files.
def persistent_hash(input_string,
                    algorithm='sha256'):
    """
    Generates a persistent hash of a string using the specified algorithm.

    Parameters
    ----------
        input_string: The string to hash.
        algorithm: The hashing algorithm to use (e.g., 'md5', 'sha256').

    Returns
    -------
        A hexadecimal string representing the hash of the input string.
    """
    hasher = hashlib.new(algorithm)
    hasher.update(input_string.encode('utf-8'))
    return hasher.hexdigest()


def add_message(user_id: str,
                session_id: str,
                assistant_id: int,
                role: str,
                message_idx: int,
                message: str):
    """
    Adds a message to the Firestore database.

    Parameters
    ----------
        user_id: str
            The ID of the user sending or receiving the message.
        session_id: str
            The ID of the session associated with the message.
        assistant_id: int
            The ID of the assistant involved in the conversation.
        role: str
            The role of the message sender ('user' or 'assistant').
        message: str
            The content of the message.

    Returns
    -------
        None
    """
    timestamp = datetime.now()
    user_id = persistent_hash(user_id)

    # Generate a unique document ID based on user_id, timestamp, and role
    # Note: The role is converted to an integer (0 or 1) for the ID
    # to ensure uniqueness for nearly simultaneous assignment to Firestore
    if role == "user":
        type = 0
    else:
        type = 1
    doc_id = (
        user_id[:12] +
        "-" +
        timestamp.strftime("%Y%m%d%H%M%S") +
        "-" +
        str(type)
        )

    # Create a Firestore client and add the message
    try:
        db = firestore.Client(credentials=creds, project="docuscope-ca-data")
        doc_ref = db.collection('messages').document(doc_id)
        doc_ref.set({
            'user_id': user_id,
            'session_id': session_id,
            'time_stamp': timestamp,
            'assistant_id': assistant_id,
            'role': role,
            'message_idx': message_idx,
            'message': message
        })
    except Exception:
        pass


def add_plot(user_id: str,
             session_id: str,
             assistant_id: int,
             message_idx: int,
             plot_library: str,
             plot_svg: str) -> None:
    """
    Adds a plot arry to the Firestore database.

    Parameters
    ----------
        user_id: str
            The ID of the user sending or receiving the message.
        session_id: str
            The ID of the session associated with the message.
        assistant_id: int
            The ID of the assistant involved in the conversation.
        role: str
            The role of the message sender ('user' or 'assistant').
        message: str
            The content of the message.

    Returns
    -------
        None
    """
    timestamp = datetime.now()
    user_id = persistent_hash(user_id)
    type = 1

    # Generate a unique document ID based on user_id, timestamp, and role
    # Note: The role is converted to an integer (0 or 1) for the ID
    # to ensure uniqueness for nearly simultaneous assignment to Firestore

    doc_id = (
        user_id[:12] +
        "-" +
        timestamp.strftime("%Y%m%d%H%M%S") +
        "-" +
        str(type)
        )

    # Create a Firestore client and add the message
    try:
        db = firestore.Client(credentials=creds, project="docuscope-ca-data")
        doc_ref = db.collection('plots').document(doc_id)
        doc_ref.set({
            'user_id': user_id,
            'session_id': session_id,
            'time_stamp': timestamp,
            'assistant_id': assistant_id,
            'message_idx': message_idx,
            'plot_library': plot_library,
            'plot_svg': plot_svg
        })
    except Exception as e:
        # Handle the exception (e.g., log it, print it, etc.)
        print(f"Error adding plot to Firestore: {e}")
        pass


def add_login(user_id: str,
              session_id: str):
    """
    Adds a user login instance to the Firestore database.

    Parameters
    ----------
        user_id: The ID of the user.
        session_id: The ID of the session.

    Returns
    -------
        None
    """
    timestamp = datetime.now()
    from_cmu = user_id.endswith(".cmu.edu")
    user_id = persistent_hash(user_id)

    doc_id = (
        user_id[:12] +
        "-" +
        timestamp.strftime("%Y%m%d%H%M%S")
        )
    try:
        db = firestore.Client(credentials=creds, project="docuscope-ca-data")
        doc_ref = db.collection('users').document(doc_id)
        doc_ref.set({
            'user_id': user_id,
            'from_cmu': from_cmu,
            'session_id': session_id,
            'time_stamp': timestamp
        })
    except Exception:
        pass


def get_query_count(user_id):
    try:
        db = firestore.Client(credentials=creds, project="docuscope-ca-data")
        collection_ref = db.collection("messages")
        timestamp = datetime.now()
        user_id = persistent_hash(user_id)

        # Calculate the timestamp for 24 hours ago
        last_24_hours = timestamp - timedelta(hours=24)

        # Create a query to filter documents by user_id and time_stamp
        query = (
            collection_ref
            .where("user_id", "==", user_id)
            .where("role", "==", "user")
            .where("time_stamp", ">=", last_24_hours)
            )
        docs = query.get()
        return len(docs)

    except Exception:
        return 0
