# Global Validation Order

Validate the narrowest relevant thing first.

## Rules

- Prefer behavior-scoped validation before broad validation.
- After producing an artifact, perform the cheapest falsifiable check before more edits.
- Use contract validation before transition when a handoff is emitted.
