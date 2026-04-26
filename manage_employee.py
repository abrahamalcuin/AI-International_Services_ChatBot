import argparse

from auth_db import create_employee, generate_invite, init_db

BASE_URL = "https://ai-international-services-chatbot.onrender.com/onboarding"


def add_employee(args):
    create_employee(args.employee_code, args.first_name, args.last_name, args.email, args.role)
    print("Employee added.")


def make_invite(args):
    token, expires_at = generate_invite(args.email, args.hours)
    print(f"Invite link: {BASE_URL}?token={token}")
    print(f"Expires at (UTC): {expires_at}")


def main():
    init_db()

    parser = argparse.ArgumentParser(description="Manage employee onboarding records")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add-employee")
    add_parser.add_argument("employee_code")
    add_parser.add_argument("first_name")
    add_parser.add_argument("last_name")
    add_parser.add_argument("email")
    add_parser.add_argument("--role", default="user", choices=["user", "admin"])
    add_parser.set_defaults(func=add_employee)

    invite_parser = subparsers.add_parser("generate-invite")
    invite_parser.add_argument("email")
    invite_parser.add_argument("--hours", type=int, default=72)
    invite_parser.set_defaults(func=make_invite)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
