---
name: create-docker-tiered-configuration
description: Use this skill when you need to manage environment variables for a Docker Compose project with multiple services and deployment targets. This method is appropriate when you have multiple services that require different environment variables, deploy to multiple hardware/platform configurations, want clear separation between Docker Compose configuration and runtime environment variables, or need to audit or review configuration changes with clear ownership.
modeSlugs:
  - architect
  - code
---

# Create Docker Tiered Configuration

## When to use this skill

Use this skill when you need to manage environment variables for a Docker Compose project with multiple services and deployment targets. This method is appropriate when:

- You have multiple services (e.g., router, database, cache) that require different environment variables
- You deploy to multiple hardware/platform configurations (e.g., x86_64, ARM64, GPU vs CPU)
- You want clear separation between Docker Compose configuration and runtime environment variables
- You need to audit or review configuration changes with clear ownership

Do NOT use this skill when:
- Your project has only one service with simple configuration
- You don't need multiple deployment targets
- You prefer all configuration in a single `.env` file

## Inputs required from the user

1. List of services in your Docker Compose deployment
2. List of deployment targets (e.g., hardware platforms, cloud providers)
3. Which variables belong to which service
4. Which variables are deployment-target-specific

## Workflow

### 1. Create the directory structure

```
profiles/
├── <service1>.env          # Service 1 runtime variables
├── <service2>.env          # Service 2 runtime variables
└── <target>/.env           # Target-specific overrides (optional)
```

### 2. Define Docker Compose variables in `.env.example`

Variables that control Docker Compose behavior:
- Image selection (`IMAGE_NAME`)
- Resource limits (`MEMORY_LIMIT`, `CPU_COUNT`)
- Profile/target selection (`TARGET_PROFILE`)

### 3. Create service-specific env files

Each file contains only variables consumed by that service:

**`profiles/<service1>.env`**:
- Variables used by service1 container
- No Docker Compose variables (no `image:`, no `deploy:`)

**`profiles/<service2>.env`**:
- Variables used by service2 container
- No Docker Compose variables

### 4. Create target-specific overrides (optional)

If you have multiple deployment targets:

**`profiles/<target>/.env`**:
- Overrides for target-specific values
- Only variables that differ from defaults

### 5. Update docker-compose.yml

Reference env files using `env_file` directive:

```yaml
services:
  service1:
    env_file:
      - profiles/service1.env
      - profiles/<TARGET_PROFILE:-default>/.env
  service2:
    env_file:
      - profiles/service2.env
      - profiles/<TARGET_PROFILE:-default>/.env
```

### 6. Validate configuration

```bash
docker compose config --quiet && echo "Configuration is valid"
```

## Examples

### Example: Multi-service application

**Directory structure:**
```
profiles/
├── router.env
├── database.env
└── redis.env
```

**`profiles/router.env`**:
```bash
# Router-specific variables
API_PORT=8080
DATABASE_URL=postgres://db:5432/app
REDIS_URL=redis://redis:6379
```

**`profiles/database.env`**:
```bash
# Database-specific variables
POSTGRES_USER=app
POSTGRES_PASSWORD=secret
POSTGRES_DB=app
```

**`profiles/redis.env`**:
```bash
# Redis-specific variables
REDIS_MAXMEMORY=256mb
REDIS_MAXMEMORY_POLICY=allkeys-lru
```

### Example: Multi-target deployment

**Directory structure:**
```
profiles/
├── router.env
├── database.env
├── x86_64/.env
└── arm64/.env
```

**`profiles/x86_64/.env`**:
```bash
# x86_64-specific overrides
DATABASE_IMAGE=postgres:15-x86_64
```

**`profiles/arm64/.env`**:
```bash
# arm64-specific overrides
DATABASE_IMAGE=postgres:15-arm64v8
```

## Troubleshooting

### Issue: Variable not being picked up

**Check**:
1. File exists in correct location
2. `env_file` path is correct in docker-compose.yml
3. Variable name matches exactly (case-sensitive)

### Issue: Wrong value being used

**Check**:
1. File loading order (later files override earlier)
2. Default values in docker-compose.yml
3. Environment variable precedence

### Issue: Configuration validation fails

**Run**: `docker compose config --quiet` to see detailed error

## Best practices

1. **One responsibility per file**: Each env file should contain only variables for one service
2. **Document each variable**: Add comments explaining purpose and valid values
3. **Use defaults in docker-compose.yml**: Set fallback values for critical variables
4. **Validate before committing**: Run `docker compose config --quiet` after changes
5. **Keep .env.example minimal**: Only include Docker Compose variables, not runtime variables

## References

- [Docker Compose env_file documentation](https://docs.docker.com/compose/reference/env-file/)
- [Docker Compose environment variable precedence](https://docs.docker.com/compose/environment-variables/)
