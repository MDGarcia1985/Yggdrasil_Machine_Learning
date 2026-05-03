
<!--
Copyright (c) 2026 Michael Garcia, M&E Design
Contact: michael@mandedesign.studio
License: MIT
Version: v0.1.0
-->

# Commenting Policy

---

## Principle

- Comments must enable a capable but less experienced engineer to understand what the code does and why it is structured that way.
- Comments are part of the teaching surface of the codebase, especially in shared libraries, core interfaces, and non-obvious system logic.
- Comments must support understanding without turning the code into a line-by-line tutorial.
- Comments must remain accurate. Incorrect comments are worse than missing comments.

---

## What Comments Must Do

- Explain the purpose of modules, classes, and functions.
- Clarify non-obvious design decisions and tradeoffs.
- Provide context for patterns, shorthand, or language features when first required for understanding.
- Define how inputs, outputs, and side effects should be interpreted.
- Call out constraints involving safety, timing, protocol, hardware, or architectural boundaries.

---

## Major Function and Interface Expectations

- Public functions, methods, and shared interfaces must include docstrings or structured comments.
- Comments must explain:
  - What the function does
  - Why it exists
  - How it fits into the system
- Core abstractions must document their contract, not just their signature.
- Functions participating in critical workflows must state their role in that workflow.
- When using non-obvious patterns, explain the pattern once at the point of first meaningful use.

---

## Explaining Shorthand and Advanced Features

- Do not explain common syntax repeatedly.
- Explain advanced constructs when they first affect understanding.
- Prefer a single clear explanation over repeated inline commentary.

Examples that may require explanation:

- Decorators  
- Enums  
- Dataclasses  
- Protocols / interfaces  
- Async control flow  
- Vectorized operations  
- Bit manipulation  
- Hardware register logic  

---

## Intent vs Implementation

### Intent Comments (Why)

Use when explaining:

- Purpose of a module or function  
- Design decisions and tradeoffs  
- System responsibilities  
- Architectural role  

### Implementation Comments (How)

Use when explaining:

- Non-obvious control flow  
- Data transformations  
- Ordering or sequencing constraints  
- Language or syntax that may not be immediately clear  

Rules:

- Prefer intent comments over implementation comments  
- Do not explain trivial implementation details  
- Place comments above blocks or functions instead of annotating every line  

---

## Good Uses of Comments

- Module headers describing system role  
- Docstrings for public APIs  
- Notes explaining abstraction or data structure choices  
- Definitions of units, coordinate systems, and state meanings  
- Clarification of timing assumptions or protocol expectations  
- Explanation of return values or mappings when not self-evident  

---

## Poor Uses of Comments

- Repeating what clearly named code already expresses  
- Explaining basic syntax or obvious operations  
- Casual or conversational commentary  
- TODOs without ownership or follow-up path  
- Using comments to compensate for unclear design  

---

## Style Rules

- Use direct, plain language  
- Prefer complete thoughts when explaining important behavior  
- Keep comments close to the code they describe  
- Use docstrings where supported  
- Use inline comments sparingly and only when necessary  

---

## Maintenance Rules

- Update comments in the same change as related code  
- Remove comments that no longer improve understanding  
- If code requires excessive commenting, refactor the code  
- Review comments from the perspective of a capable but inexperienced engineer  
- Agent-generated comments must be reviewed and approved by a human before merging  

---

## Decision Record References

- If a concept cannot be explained clearly in code comments, create a companion markdown file  
- Place the file alongside the relevant code unless a stronger repository convention applies  
- The companion file must include:
  - Design context  
  - Key decisions  
  - Constraints  
- The file must:
  - Link to the corresponding Decision record  
  - Link back to the relevant source code  

---

## Enforcement

- Missing required comments are defects  
- Incorrect comments are defects  
- Over-commented trivial code is a design signal and should be refactored  

This policy is enforceable through code review.
