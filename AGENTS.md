# Code Comments

- Document public and crate-visible APIs: structs, enums, traits, type aliases, constants, methods, and fields.
- For resource wrappers, mention ownership, lifetime, synchronization, or safety assumptions when relevant.
- Add inline comments only for non-trivial logic, especially Vulkan barriers, descriptor writes, staging, command ordering, SBT layout, and device addresses.
- Explain why code exists, not what obvious syntax does.
- Keep comments brief, precise, and update them when behavior changes.
