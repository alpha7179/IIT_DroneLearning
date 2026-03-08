# REQUEST_TEMPLATE.md - Ask Codex Effectively

Use this template when requesting implementation/design support.

## 1) Request Format
`goal | stage | inputs | constraints | done criteria`

Example:
`Implement Stage1 pursuer config | Stage1 | ray + self state | Unity 6.0.57f1, ML-Agents 4.0.x | run launches and logs metrics`

## 2) Include These Constraints
- Branch: `work/pursuer`
- No destructive git operations
- Keep one capability per change set
- Write/update experiment log entry

## 3) Ask for Outputs Explicitly
- files to create/update
- commands to run
- metrics expected to move
- rollback plan if training regresses
