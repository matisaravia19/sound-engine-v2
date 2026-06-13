# Code Comments

- Document public and crate-visible APIs: structs, enums, traits, type aliases, constants, methods, and fields.
- For resource wrappers, mention ownership, lifetime, synchronization, or safety assumptions when relevant.
- Add inline comments only for non-trivial logic, especially Vulkan barriers, descriptor writes, staging, command ordering, SBT layout, and device addresses.
- Explain why code exists, not what obvious syntax does.
- Keep comments brief, precise, and update them when behavior changes.

# Documentation

The existing documentation in /docs are outdated, just use them to understand the general idea for the project, but don't
take them as a reference for the code. Also don't mind updating them, they are not meant to be updated.