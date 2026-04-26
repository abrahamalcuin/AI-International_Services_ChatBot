# Login and onboarding

This backend now includes:
- `/login` UI
- `/logout`
- `/onboarding?token=...`
- `/SEO` protected route that redirects to `/login` if the user is not signed in

## Setup

Environment variables:
- `SESSION_SECRET` - required in production
- `AUTH_DB_PATH` - optional, defaults to `auth.db`

## Render note

SQLite files on Render need persistent disk/storage if you want accounts to survive redeploys or restarts.
Without persistent storage, the database can reset.

## Manage employees

```bash
python manage_employee.py add-employee EMP001 Alice Smith alice@example.com
python manage_employee.py generate-invite alice@example.com --hours 72
```
