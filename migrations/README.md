# Control-plane migrations

Alembic is the authoritative schema upgrade path for PostgreSQL deployments.
The connection URL comes from `DATABASE_URL`; the URL in `alembic.ini` is only a
local fallback.

## New database

```bash
alembic upgrade head
```

## Existing database created with `Base.metadata.create_all()`

The baseline revision represents the schema that existed before migration
versioning was introduced. Stamp that already-existing schema, then apply the
corrective migrations:

```bash
alembic stamp 20260608_0001
alembic upgrade head
```

Back up PostgreSQL before the first migration. Revision `20260805_01` consolidates
any duplicate public artifact identities, repoints their jobs, and replaces null
public ownership with the reserved `__public__` principal before enforcing the
non-null uniqueness constraint.

The Compose `migrate` service runs `alembic upgrade head`; the application and
worker depend on its successful completion. Non-Compose deployments must use
the same ordering explicitly. `Base.metadata.create_all()` remains available
for isolated tests and local compatibility; it is not a replacement for
upgrades.
