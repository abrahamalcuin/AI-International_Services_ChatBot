# SEO Project Management module

This repo now includes a protected SEO project management workspace under `/SEO`.

## What was added
- Session-protected SEO dashboard pages: `/SEO`, `/SEO/clients`, `/SEO/tasks`
- JSON API endpoints for clients, templates, and task status updates
- SQLite schema extensions inside `auth.db`
- Auto-seeded tier templates for Tier 1, Tier 2, and Tier 3
- Automatic task generation when a client is created

## SQLite tables
- `seo_clients`
- `seo_task_templates`
- `seo_client_tasks`

## Notes
- Credentials are stored as plain text in SQLite because that was part of the requested fields and there was no existing secrets vault in this repo. In production, move that field to encrypted storage.
- Startup now tolerates missing Gemini configuration so the SEO module can still load while chat features remain unavailable until `GEMINI_API_KEY` is configured.
