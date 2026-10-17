-- Runs once when the PostgreSQL container is first created.
-- The API creates the tables on start-up; this file prepares a read-only role for the read endpoints.
CREATE ROLE eg_readonly LOGIN PASSWORD 'eg_readonly';
GRANT CONNECT ON DATABASE evidenceguard TO eg_readonly;
GRANT USAGE ON SCHEMA public TO eg_readonly;
ALTER DEFAULT PRIVILEGES FOR ROLE evidenceguard IN SCHEMA public GRANT SELECT ON TABLES TO eg_readonly;
