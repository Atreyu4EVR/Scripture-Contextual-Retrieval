# Permission Correspondence

Scanned or saved permission correspondence lives here, one file per grant.
A record's `permission_ref` field points at a path in this directory.

Conventions:

- Name files `<source-name>-<yyyy-mm-dd>.<ext>` using the date permission was granted.
- Save the full exchange (request and reply), not just the approval line.
- A source whose `permission_status` is `granted` must reference a file here;
  M3 forum adapters are blocked until that file exists (CLAUDE.md, Build Order).
