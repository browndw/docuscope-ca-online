"""
User Management Utilities

Simple utilities for managing user authorization that can be used
from the command line or integrated into admin interfaces.
"""

import sys
from webapp.utilities.auth import (
    add_authorized_user,
    remove_authorized_user,
    list_authorized_users,
    initialize_authorization_db
)


def add_user_cli():
    """Command line utility to add a user."""
    if len(sys.argv) < 4:
        print("Usage: python -m webapp.utilities.auth.manage add <email> <role>")
        print("Roles: user, instructor, admin")
        return

    email = sys.argv[2]
    role = sys.argv[3]

    if role not in ['user', 'instructor', 'admin']:
        print("Invalid role. Must be: user, instructor, or admin")
        return

    initialize_authorization_db()

    if add_authorized_user(email, role, added_by='cli'):
        print(f"Successfully added {email} with role '{role}'")
    else:
        print(f"Failed to add user {email}")


def list_users_cli():
    """Command line utility to list users."""
    initialize_authorization_db()
    users = list_authorized_users()

    if not users:
        print("No authorized users found.")
        return

    print(f"{'Email':<30} {'Role':<12} {'Added By':<15} {'Active'}")
    print("-" * 70)

    for user in users:
        active_str = "Yes" if user['active'] else "No"
        added_by = user['added_by'] or 'N/A'
        print(f"{user['email']:<30} {user['role']:<12} {added_by:<15} {active_str}")


def remove_user_cli():
    """Command line utility to remove a user."""
    if len(sys.argv) < 3:
        print("Usage: python -m webapp.utilities.auth.manage remove <email>")
        return

    email = sys.argv[2]

    initialize_authorization_db()

    if remove_authorized_user(email):
        print(f"Successfully removed {email}")
    else:
        print(f"Failed to remove user {email}")


def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        print("Usage: python -m webapp.utilities.auth.manage <command>")
        print("Commands:")
        print("  add <email> <role>  - Add a new user")
        print("  remove <email>      - Remove a user")
        print("  list                - List all users")
        return

    command = sys.argv[1]

    if command == "add":
        add_user_cli()
    elif command == "remove":
        remove_user_cli()
    elif command == "list":
        list_users_cli()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: add, remove, list")


if __name__ == "__main__":
    main()
